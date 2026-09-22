# command_v17.md — uni_section_v16 → v17 개선 지시서 (idea_v17.md 5개 판정 구현)

> 출처: /synod idea 세션 (Claude Judge + Gemini flash/high Architect + OpenAI gpt4o Explorer, Solver 라운드
> 실 병렬 교차검증, conf 95(can_exit=true) / 85(can_exit=false) — 옵티마이저 재구성 방식 쟁점은 Judge가
> PyTorch 동작 원리 근거로 해소)
> 기반: `docs/idea/idea_v17.md`(5개 아이디어 채택/조건부채택/기각 판정 완료). 베이스는 `uni_section_v16.py`.

## 0. 범위 선언

idea_v17.md에서 채택/조건부채택된 5개 항목을 전부 구현한다. "완전 리셋"(가중치 재초기화)은 idea_v17.md에서
이미 기각되었으므로 이번 범위에 포함하지 않는다.

| idea_v17.md 항목 | 판정 | 이번 v17에 포함? |
|---|---|---|
| 1. 후보군 축소 | 채택 | 포함 |
| 2. w_sparse 상향+Mp err 게이팅 | 채택(파라미터 제약) | 포함 |
| 3. 엔트로피 정규화 | 조건부 채택 | 포함(gate_active 구간 한정) |
| 4. 더블쇼크 분리 | 채택 | 포함 |
| 5. State-Only Reset | 수정 채택 | 포함 |
| (참고) 완전 리셋 | 기각 | 미포함 |

> **Solver 라운드 쟁점 및 해소**: 아이디어5의 옵티마이저 재구성 방식에서 Gemini는 `optim.AdamW(...)`를
> **새로 생성**하며 기존 `optimizer.param_groups`에서 `lr`을 동적으로 읽어와 보존하는 방식을 제시했고,
> OpenAI는 기존 optimizer 객체의 `param_groups[i]['params']`만 재대입하는 방식을 제시했다. **Judge 판정**:
> PyTorch AdamW의 모멘텀은 `optimizer.state`(파라미터 id를 key로 하는 dict)에 저장되며, `param_groups`의
> `params` 리스트만 바꿔치기해도 `optimizer.state`에 남아있는 기존 항목은 지워지지 않는다 — 즉 OpenAI 방식은
> **모멘텀을 전혀 flush하지 못해 idea_v17.md가 채택한 "State-Only Reset"의 핵심 목적(모멘텀 오염 제거)을
> 달성하지 못하는 결함이 있다.** Gemini의 "새 AdamW 인스턴스 생성 + lr 동적 보존" 방식을 채택한다.

---

## 1. 상수 변경

```python
# --- v17 변경 ---
STAGE2_EPOCH = 128                      # 유지(두께 언락 시점, 변경 없음)
GATE_ACTIVE_EPOCH = STAGE2_EPOCH + 30   # [아이디어4] 게이트/프루닝 활성화를 두께 언락과 분리

S_PROTECT = [0, 1, 2]        # [아이디어1] Inner Hat(#06, part_id=2) 추가 보호
CANDIDATE_PARTS = [3, 4]     # [아이디어1] Patch1(#07), Patch2(#08)만 후보

PRUNING_COOLDOWN_EPOCHS = 20  # [아이디어5] 삭제 확정 직후 트리거 판정 일시 정지 기간

# [아이디어2] 희소화 게이팅
W_SPARSE = 0.10        # 기존 0.05 → 상향(idea_v17.md 권장 범위 0.05~0.2 내 중간값)
TAU_GATE = 0.05        # Mp err 5% 미만일 때만 희소화 압력 강해짐
SPARSE_K = 50.0        # 시그모이드 전환 기울기

# [아이디어3] 엔트로피 정규화
ALPHA_ENT = 0.01       # 엔트로피 항 스케일(게이팅과 별도로 한 번 더 감쇠됨, §3 참조)
ENTROPY_EPS = 1e-7
```

`S_PROTECT`, `CANDIDATE_PARTS`는 `CGDN.__init__`의 `self.s_protect`에도 동일하게 반영되어야 한다(v16의
`self.s_protect = S_PROTECT` 라인이 이미 모듈 상수를 참조하므로, 상수만 바꾸면 자동 반영됨 — 별도 코드
변경 불필요, 확인만 할 것).

---

## 2. Stage 스케줄 재정의 (아이디어4)

v16은 `gate_active = epoch >= STAGE2_EPOCH`로 두께 언락과 게이트 활성화가 동일 시점이었다. v17은 이를
분리한다.

```
Stage 1 (epoch < 128):            두께 동결(thickness_gate≈0), 게이트 비활성(z_gate=1 강제)
Stage 2 (128 <= epoch < 158):     두께 언락(thickness_gate 램프), 게이트는 아직 비활성 — 두께가 먼저 안정화
Stage 3 (epoch >= 158):           게이트/희소화/엔트로피/프루닝 트리거 전부 활성화
```

`train_step`/`run_training` 양쪽에서 `gate_active` 계산식을 다음으로 교체:

```python
# train_step 내부 (기존: gate_active = epoch >= STAGE2_EPOCH)
gate = thickness_gate_value(epoch)             # 유지 — 두께 언락은 STAGE2_EPOCH 그대로
gate_active = epoch >= GATE_ACTIVE_EPOCH        # [아이디어4] 별도 시점
```

`train_step` 시그니처 자체는 변경할 필요 없음(`epoch`을 이미 인자로 받으므로 내부에서
`GATE_ACTIVE_EPOCH` 모듈 상수를 참조하면 됨).

---

## 3. 희소화 게이팅 + 엔트로피 정규화 (아이디어2/3) — `train_step` 내부

기존(v16):
```python
l_sparse = z_open[CANDIDATE_PARTS].mean()
contrib_sparse = weights['w_sparse'] * l_sparse
```

v17로 교체:
```python
# [아이디어2] Mp err 기반 시그모이드 게이팅
gate_multiplier = torch.sigmoid(torch.tensor(SPARSE_K * (TAU_GATE - mp_rel_err))).item()
w_sparse_effective = weights['w_sparse'] * gate_multiplier

l_sparse = z_open[CANDIDATE_PARTS].mean()
contrib_sparse = w_sparse_effective * l_sparse

# [아이디어3] 엔트로피 정규화 — gate_active 구간에서만, gate_multiplier로 추가 감쇠
contrib_entropy = torch.tensor(0.0, device=x.device)
if gate_active:
    z_cand = z_open[CANDIDATE_PARTS]
    entropy = -torch.mean(z_cand * torch.log(z_cand + ENTROPY_EPS)
                           + (1.0 - z_cand) * torch.log(1.0 - z_cand + ENTROPY_EPS))
    contrib_entropy = (ALPHA_ENT * gate_multiplier) * entropy
```

`mp_rel_err`는 `train_step` 내부에서 이미 계산되는 변수(섹션별 Mp 합산 오차, `.item()` 스칼라)를 그대로
재사용한다 — 별도 계산 불필요.

최종 loss 조립부(v16 대비 `contrib_entropy` 항 추가):
```python
loss = (contrib_phys + contrib_smooth + L_alda_effective
        + contrib_order + contrib_anchor + contrib_sat
        + contrib_sparse + contrib_entropy)
```

`train_step`의 반환 dict에 `"gate_multiplier": gate_multiplier`를 추가해 `run_training`에서 로깅에
사용할 수 있게 한다.

---

## 4. State-Only Reset (아이디어5) — `run_training` 내부

### 4.1 쿨다운 상태 관리

`pruning_state` dict(이미 v16에 존재)에 쿨다운 카운터를 추가로 저장한다(별도 변수로 관리해도 무방하나,
체크포인트 저장 시 함께 다루기 위해 `pruning_state`에 통합 권장):

```python
# make_pruning_state() 반환 dict에 추가
pruning_state['cooldown_remaining'] = 0   # [아이디어5]
```

### 4.2 epoch 루프 내 삽입 위치 — `update_pruning_state` 호출부 교체

기존(v16):
```python
if info['gate_active']:
    newly_deleted = update_pruning_state(pruning_state, info['z_gate'])
    for pid in newly_deleted:
        model.log_alpha.data[pid] = -10.0
        print(f"[Pruning] epoch {epoch}: part {pid} DELETED ...")
```

v17로 교체:
```python
if info['gate_active']:
    if pruning_state['cooldown_remaining'] > 0:
        pruning_state['cooldown_remaining'] -= 1
        newly_deleted = []
    else:
        newly_deleted = update_pruning_state(pruning_state, info['z_gate'])

    for pid in newly_deleted:
        model.log_alpha.data[pid] = -10.0
        print(f"[Pruning] epoch {epoch}: part {pid} DELETED — triggering State-Only Reset")

        # [아이디어5] (a) 옵티마이저 재인스턴스화 — 반드시 새 AdamW 객체 생성
        #   (기존 param_groups의 'params' 리스트만 바꿔치기하면 optimizer.state가 그대로 남아
        #    모멘텀이 flush되지 않는다 — Solver 라운드에서 기각된 방식, 재도입 금지)
        current_lrs = {g['name']: g['lr'] for g in optimizer.param_groups}
        optimizer = optim.AdamW(
            [{'params': main_params,  'name': 'main',              'lr': current_lrs['main']},
             {'params': thick_params, 'name': 'thickness_decoder', 'lr': current_lrs['thickness_decoder']},
             {'params': gate_params,  'name': 'gate_params',       'lr': current_lrs['gate_params']}],
            weight_decay=1e-4)

        # (b) ALDA 승수 초기화(rho는 유지)
        alda_state['mu_Mp'] = 1.0
        alda_state['mu_col'] = 1.0
        print(f"[Reset] optimizer 재인스턴스화 완료, mu_Mp/mu_col → 1.0 (rho_Mp/rho_col 유지)")

        # (c) 쿨다운 시작 — 이후 PRUNING_COOLDOWN_EPOCHS 동안 트리거 판정 정지
        pruning_state['cooldown_remaining'] = PRUNING_COOLDOWN_EPOCHS
```

**주의**: `main_params`, `thick_params`, `gate_params`는 `run_training` 시작부에서 이미 정의된 리스트를
그대로 재사용한다(모델 파라미터 객체 자체는 삭제 이벤트로 바뀌지 않으므로 — soft-zero라 파라미터 shape가
불변임을 재확인). `current_lrs`를 옵티마이저 생성 직전에 매번 새로 읽어오므로, Stage2/gate_params lr 조정
등 그 시점까지의 스케줄 변경 사항이 자동으로 보존된다(Solver 라운드 Gemini 제안, 하드코딩 금지).

### 4.3 10epoch ALDA 승수 갱신 주기와의 상호작용 (리스크 대응)

옵티마이저 재구성/ALDA mu 리셋이 10epoch 주기의 갱신 시점과 우연히 겹칠 수 있다. 이 경우 특별 처리는
불필요하다 — mu 리셋 직후 10epoch 갱신 로직(`alda_state['mu_Mp'] = min(..., max(0.0, mu_Mp + rho*g_Mp))`)
은 리셋된 `mu_Mp=1.0`을 기준으로 정상적으로 다시 누적을 시작하므로 별도 동기화 로직이 필요 없다. 단,
`rho_freeze_epochs=30` 로직(§ALDA 적응형 rho)이 삭제 이벤트 직후에도 카운트다운 중이라면 그대로 진행되게
둔다(리셋과 무관한 별도 타이머).

---

## 5. 검증 기준 (Gate)

1. **실제 프루닝 발동**: `CANDIDATE_PARTS=[3,4]` 중 최소 1개가 300epoch 이내 `DELETED` 상태에 도달(v16의
   "미발동" 문제 해소 확인).
2. **Mp/충돌 유지**: `Mp err`, `l_collision`이 v16 베이스라인(최종 0.96%/0.0000) 수준에서 크게 후퇴하지
   않을 것.
3. **Reset 후 안정성**: State-Only Reset 직후 5epoch 이내에 loss가 폭주(NaN/Inf 또는 이전 대비 10배 이상
   급증)하지 않고 완만하게 재수렴할 것.
4. **쿨다운 준수**: 한 번의 삭제 이벤트 이후 `PRUNING_COOLDOWN_EPOCHS`(20) 이내에 추가 삭제가 발생하지
   않을 것(로그로 확인).
5. **더블쇼크 해소 확인**: `GATE_ACTIVE_EPOCH`(158) 근방에서 v16의 epoch100~119 구간 같은 급격한
   MpErr/area 스파이크(예: 10%p 이상 급등)가 재현되지 않을 것.

---

## 6. 리스크 (5개 아이디어 동시 적용 시 조합 리스크)

| ID | 리스크 | 메커니즘 | 완화책 |
|----|--------|----------|--------|
| R-1 | 옵티마이저 재구성 시 lr 하드코딩 | 삭제가 Stage2 lr 조정 이전/이후 등 다양한 시점에 발생 가능 | §4.2처럼 `optimizer.param_groups`에서 **매번 동적으로** 현재 lr을 읽어와 보존(하드코딩 금지) |
| R-2 | 엔트로피-희소화 상쇄(command_v13.md §3에서 지적된 유형과 동일 계열) | `contrib_entropy`와 `contrib_sparse`가 동시에 강하게 걸리면 z가 중간값에 재고착 가능 | 둘 다 동일한 `gate_multiplier`(Mp err 기반)로 감쇠되므로 Mp err가 높을 땐 둘 다 약함 — 이미 완화책 내장(§3) |
| R-3 | 쿨다운 중 EWMA 계속 갱신되어 쿨다운 해제 즉시 재트리거 | `update_pruning_state` 자체를 건너뛰므로 `ewma_z`가 쿨다운 동안 갱신되지 않음 — 쿨다운 해제 직후 직전 값 그대로 판정 재개 | 의도된 동작(§4.2에서 `update_pruning_state` 호출 자체를 skip). 필요시 후속 버전에서 쿨다운 중 EWMA만 별도로 계속 갱신할지 재검토 |
| R-4 | GATE_ACTIVE_EPOCH(158)와 쿨다운(20epoch)이 학습 후반부에 몰릴 경우 프루닝 판정 실질 가동 기간이 짧아짐 | 300epoch 중 158~300(142epoch)에서만 게이트 활성, 삭제 1회당 20epoch 쿨다운 소모 | 2개 후보(Patch1/Patch2) 기준 이론상 여유 있음(142epoch ÷ 20epoch 쿨다운 ≫ 2). 문제 시 `max_epochs` 확장 또는 `GATE_ACTIVE_EPOCH` 하향 조정 |
| R-5 | State-Only Reset이 optimizer.state를 완전히 새로 만들면서 AdamW의 `weight_decay` 등 하이퍼파라미터를 재확인 없이 고정값(1e-4)으로 재설정 | 원래 `run_training(lr=...)` 인자로 받은 `weight_decay`가 있다면 재구성 시 다를 수 있음 | §4.2 코드에서 `weight_decay=1e-4`를 `run_training`의 원래 인자와 동일한 값으로 명시적으로 맞출 것(하드코딩 값 그대로 두되 실제 구현 시 원본 인자 재확인) |

---

## 부록: 숙의 과정

- **Solver 라운드**: Gemini(flash — pro는 이번에도 rate-limit되어 폴백, thinking high, conf 95,
  can_exit=true) — 5개 아이디어 전체를 정확한 코드 삽입 지점과 함께 제시, 특히 옵티마이저 재구성 시
  `current_lrs`를 동적으로 읽어와 보존하는 방식과 완전한 새 AdamW 인스턴스 생성을 명시. OpenAI(gpt4o,
  conf 85, can_exit=false) — 대체로 동일한 구조에 도달했으나, 옵티마이저 재구성을 `param_groups`의
  `params` 리스트만 바꿔치기하는 방식으로 제안(모멘텀 미flush 결함).
- **쟁점 해소(Critic 라운드 생략, Judge 직접 판정)**: PyTorch AdamW의 `optimizer.state`는 파라미터 id를
  key로 하는 dict이며 `param_groups`의 `params` 리스트 교체만으로는 지워지지 않는다는 사실에 근거해
  Gemini 방식(새 인스턴스 생성)을 채택 — OpenAI의 `reset_optimizer` 함수는 idea_v17.md가 채택한
  "State-Only Reset"의 핵심 목적(모멘텀 오염 제거)을 달성하지 못하므로 재도입 금지 항목으로 명시(§4.2 주석).
- **공통 합의**: 후보군 축소, Mp err 게이팅된 희소화/엔트로피, GATE_ACTIVE_EPOCH 분리, 쿨다운 메커니즘
  전부 이견 없이 채택.
