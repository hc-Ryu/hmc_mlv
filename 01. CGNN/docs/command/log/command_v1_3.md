# command_v1_3.md — AI_design_v1_2.py → AI_design_v1_3.py: 질량 역설·게이트 미수렴 수정 지시서

작성일: 2026-08-08 (`/synod idea` 세션 `synod-20260808-idea-cgnn13` 합의안)
개정: 2026-08-08 (`/synod review` 세션 `synod-20260808-review-cmdv13` + 사용자 결정 반영 — §5 전면 개정)
개정2: 2026-08-08 (`/synod review` 세션 `synod-20260808-review-ideav4-vs-cmdv13` + 사용자 결정 ①②③ 반영
— §2.4/§7 롤백 기준을 **전역 사망 원장 방식**으로 교체, §5.1 Stage4 시작 epoch 변수 정의, §3.5-4 조치 사다리 신설)
개정3: 2026-08-08 (`/synod review` 세션 `synod-20260808-review-final-cmdv13-vs-v1_2py` +
`AI_design_v1_2.py` 실측 대조 반영 — §2.3 호출부 diff 추가, §5.1 Stage4 진입 변수를
`stage4_start_epoch`로 개명(기존 리포트용 `final_epoch`과의 이름 충돌 제거), §5.9 반환
시그니처·`__main__` 언패킹 diff 신설, §2.6.2 `contrib_mass` 삽입 지점 명시)
참조: `docs/idea/idea_v4.md`(설계 근거), `reports/v1_2/AI_design_v1_2.md`(1500 epoch 실행 로그),
`AI_design_v1_2.py`(기반 코드), `uni-section/code/uni_section_v18.py`(**불변 — import만**)

> **앵커 규칙**: 본 문서의 모든 위치 지정은 **함수/변수 이름 기준**이다. 라인 번호는 참고용일 뿐
> 코드 변경으로 어긋날 수 있으므로, 구현 에이전트는 반드시 이름 매칭으로 위치를 찾을 것.

> **불변 규칙**: `uni_section_v18.py`는 **한 줄도 수정하지 않는다.** `W_SPARSE`, `ALPHA_ENT`,
> `T_MIN`, `T_MAX`는 모두 `AI_design_v1_3.py` 안에서 **로컬 오버라이드/서브클래스 속성**으로 덮는다.
> monkey-patch(`usv18.ALPHA_ENT = ...`)는 **금지** — 근거는 §2.3.

> **제조 제약 (사용자 확정, 2026-08-08)**: 연속 부재(Outer Hat / Inner / Plate)는 전 섹션에 존재하는
> **단일 판재**이므로 **부재당 두께는 항상 하나**다. 두께 그룹 분할(run pooling)은 **Patch1 / Patch2에만**
> 적용한다. 상세 및 금지 근거는 §5.0.

---

## §0. 목표 및 산출물

`AI_design_v1_2.py`를 복사해 **`AI_design_v1_3.py`** 를 만들고, 1500 epoch 실행에서 관측된
세 가지 병리를 제거한다.

| 병리 | 관측값 | 원인 (idea_v4.md) | 대응 |
|---|---|---|---|
| **질량 역설** | 단면적 24,056.9 → 26,497.8 mm² (**+10.1%**) | 면적과 무관한 상수 페널티 `l_sparse`(`w_sparse=0.5`)가 이득 없는 패치 삭제를 강제 → Mp 보상으로 연속 파트가 두꺼워짐 | §2 |
| **게이트 미수렴** | Patch1 평균 게이트 0.926(ep680) → 0.584(ep1499), 단조 하강 | 이진화 압력이 `ALPHA_ENT`(0.01)가 아니라 단방향 `W_SPARSE`(0.5)에 실려 있음 | §2, §5 |
| **조용한 죽음의 구간** | sec 11 hard/target **95.0%** 미달 | 프루닝 임계 0.08 ≠ 이진화 임계 0.5. 구간 [0.08, 0.5]의 게이트가 학습에는 참여하고 제조에선 삭제됨 | §4, §5 |

**추가 목표 (사용자 확정 2026-08-08)**: 병리 제거에 그치지 않고 **초기 대비 5% 경량화**를 달성한다.
목표 단면적 **22,854.1 mm²** (= 24,056.9 × 0.95). 구현 §2.6, 판정 §7.1.

**산출물**

- `AI_design_v1_3.py`
- `tests/test_thickness_range_v1_3.py` (학습 불필요, §6.1)
- `tests/test_sparse_entropy_v1_3.py` (학습 불필요, §6.2)
- `tests/test_stage4_binarize_v1_3.py` (학습 불필요, §6.3) — **[개정 신규]**
- 기존과 동일 경로의 리포트 + `reports/AI_design_v1_3_report.md`

**적용 순서(반드시 준수)** — idea_v4.md §7

```
[1단계] §2 (W_SPARSE→0, ALPHA_ENT↑, 감량 목표 95%) → 재학습 → 면적 감소·게이트 포화 확인
[2단계] §3 (T_MIN 0.7 / T_MAX 2.3 + DELTA_SCALE 10.0) → 재학습 → 전 파트 두께 가동성 확인
[3단계] §4 (BINARIZE_THRESHOLD 0.3)  → 리포트에서 죽음의 구간 잔존 게이트 수 확인
[4단계] §5 (Stage 4) — 3단계 종료 시 게이트가 0/1로 수렴하지 않았으면 자동 진입
```

> §5(Stage 4)만 먼저 만드는 것은 **금지**한다. 증상 수습일 뿐이며 다음 실행에서 같은 결과가 반복된다.

---

## §1. 헤더 및 상수 블록

`AI_design_v1_2.py` 최상단 docstring을 v1.3용으로 갱신하고, v1.3 전용 상수 블록을 신설한다.

> ### ★ 배치 위치 (개정 — 초안의 치명적 오류 수정)
>
> 상수 블록은 **`import` 문 직후, `class CGDN17(usv18.CGDN)` 정의보다 반드시 앞에** 둔다.
>
> **초안은 `ASWC_STAGE1_END = 100` 정의 블록 바로 아래(≈L907)에 두라고 지시했으나 이는 오류다.**
> `CGDN17`은 **≈L135**에 정의되고 §3.1에 따라 `T_MIN_V13` / `T_MAX_V13` / `DELTA_SCALE_V13`을
> **클래스 속성으로 참조**한다. 클래스 바디는 import 시점에 실행되므로, 상수가 770여 줄 뒤에 있으면
> **import 즉시 `NameError`** 로 아무것도 실행되지 않는다.
>
> `ASWC_STAGE*_END` 블록은 기존 위치에 그대로 두고, **v1.3 상수만 파일 상단으로 올린다.**

```python
# ══════════════════════════════════════════════════════════════════
# [v1.3] 로컬 오버라이드 상수 (uni_section_v18.py 불변, 여기서만 덮는다)
# 근거: docs/idea/idea_v4.md, docs/command/command_v1_3.md
# ★ 배치: import 직후 / class CGDN17 정의보다 앞 (§1)
# ══════════════════════════════════════════════════════════════════

# §2. 질량 역설 — 면적과 무관한 상수 페널티 제거, 이진화 압력을 엔트로피로 이관
W_SPARSE_V13  = 0.0     # usv18.W_SPARSE(0.5) 대체. 0.0 = idea_v4 제안 A안(권장)
ALPHA_ENT_V13 = 0.05    # usv18.ALPHA_ENT(0.01) 대체. 대칭 이진화 압력
USE_AREA_WEIGHTED_SPARSE = False   # §2.2 B안 스위치. 기본 False

# §2.6 질량 감량 목표 (사용자 확정 2026-08-08) — 초기 단면적의 95%
AREA_TARGET_RATIO = 0.95   # target_area = ratio × 초기 단면적 24,056.9 → 22,854.1 mm²
W_MASS_V13        = 10.0   # ★ l_mass를 loss에 편입할 때의 가중치. 0.0이면 예산 제약 없음

# §3. 두께 한계 (제조 제약) — 요구: 최종 두께가 [0.7, 2.3] 전 구간에서 자유롭게 결정될 것
T_MIN_V13 = 0.7         # usv18.CGDN.T_MIN(0.1) 대체
T_MAX_V13 = 2.3         # usv18.CGDN.T_MAX(2.5) 대체
FRAC_CLAMP_HI = 0.99    # 앵커 logit 상한 (+4.60). 상세 §3.2
FRAC_CLAMP_LO = 0.01    # 앵커 logit 하한 (-4.60)
DELTA_SCALE_V13 = 10.0  # ★ usv18.CGDN.DELTA_SCALE(1.35) 대체. 전 구간 도달의 핵심 노브. §3.2

# §4. 이진화 임계 — 프루닝 임계(0.08)와의 "조용한 죽음의 구간" 축소
BINARIZE_THRESHOLD = 0.3

# §5. Stage 4 — 존재 확정 후 형상·두께 재수렴
ENABLE_STAGE4   = None  # None = §5.1 자동 판정 / True = 강제 진입 / False = 강제 비활성
STAGE4_EPOCHS   = 200
STAGE4_LR_SCALE = 0.1

# §7. 전역 게이트 사망 감시 (개정2) — 특정 섹션 하드코딩 금지, 전 섹션 × 전 후보에 적용
DEATH_MP_THRESHOLD   = 0.98   # 사망 시점 해당 섹션 hard/target 이 이 값 미만이면 SUSPECT
DEATH_BURST_WINDOW   = 100    # R2: 연쇄 붕괴 감시 창 (epoch)
DEATH_BURST_RATIO    = 0.20   # R2: 창 내 신규 0-포화가 ALIVE 대비 이 비율 이상이면 중단
AREA_REGRESS_WINDOW  = 200    # R3: 질량 역설 감시 창 (epoch)

# ★ §2.2 B안 가드 — 스위치만 켜고 가중치를 0으로 두면 침묵 no-op이 된다
assert not (USE_AREA_WEIGHTED_SPARSE and W_SPARSE_V13 == 0.0), \
    "B안(면적 가중 l_sparse) 활성 시 W_SPARSE_V13는 0.02~0.05여야 한다 (command_v1_3.md §2.2)"
```

---

## §2. 질량 역설 제거 — `l_sparse` 무력화 + 엔트로피 이관

### §2.1 `weights['w_sparse']` 오버라이드

- **대상**: `run_training_multi()` 내부 `weights = {...}` 기본값 블록
- **AS-IS**

```python
weights = {'w_phys': 10.0, 'w_order': 1.0, 'w_smooth': 0.5, 'w_anchor': 0.02, 'w_sat': 0.01,
           'w_sparse': usv18.W_SPARSE}
```

- **TO-BE**

```python
weights = {'w_phys': 10.0, 'w_order': 1.0, 'w_smooth': 0.5, 'w_anchor': 0.02, 'w_sat': 0.01,
           'w_sparse': W_SPARSE_V13,      # [v1.3] usv18.W_SPARSE(0.5) → 0.0
           'w_mass':   W_MASS_V13}        # [v1.3] §2.6 감량 목표 예산 제약. 여기서 한 번에 넣는다
```

> **개정2**: `'w_mass'` 키를 이 블록에 **함께** 넣는다. 초판은 §2.6.2에서 산문으로 "추가하라"고만 해
> 같은 dict를 두 절에서 갱신하게 되어 있었고, §2만 먼저 적용하면 `KeyError: 'w_mass'`가 난다.

**근거**: 미분 가능한 질량 목적함수가 이미 존재한다.

> ### ⚠️ 근거 정정 (개정 — 초안 및 idea_v4.md §2.1의 오류)
>
> 초안과 `idea_v4.md`는 근거로 `usv18.compute_mass_loss()`가 반환하는 **`l_mass`** 를 지목했다.
> **이는 틀렸다.** 코드 확인 결과 `l_mass`는 **계산·로깅만 되고 `loss`에 더해지지 않는다.**
>
> ```python
> :1024  area, l_mass = usv18.compute_mass_loss(new_coords, t_final, edge_index, edge_attr, target_area)
> :1048  loss = (contrib_phys + contrib_smooth + L_alda_effective
>                + contrib_order + contrib_anchor + contrib_sat
>                + contrib_sparse + contrib_entropy + contrib_continuity)   # ← l_mass 없음
> :1071  "l_mass": l_mass.item(),      # 로깅 전용
> ```
>
> **진짜 질량 항은 `usv18.compute_alda_loss()` 안의 선형항 `f = area / target_area` 다.**
>
> ```python
> f = area / (target_area + 1e-12)          # relu 없음 → 상시 활성, 계수 1.0
> L_alda = f + term_Mp + term_col
> # train_step_multi: L_alda_effective = L_alda * max(gate, 0.05)
> ```
>
> **제안 A의 결론은 그대로 유효하다.** `area = Σ seg_len · t_src`이고 `t_final = t_raw · z_gate`이므로
> 면적은 게이트에 대해 미분 가능하며, `∂f/∂z_k = A_k / target_area`가 상시 하강 압력을 공급한다.
>
> **`l_mass`는 §2.6에서 감량 목표(예산 제약)의 수단으로 되살린다.** 그때까지는 로깅 전용이다.

`l_sparse = z_open[:, CANDIDATE_PARTS].mean()`은 파트 크기와 무관한 **상수 페널티**여서,
면적 기여가 작은 패치일수록 `w_sparse ≫ A_k`가 되어 이득 없는 삭제를 강제한다. 정량 비교:

| 항 | 게이트 1개당 그래디언트 |
|---|---|
| 물리 면적 (`f`) | `A_k / target_area` ≈ 200/24,057 ≈ **0.0083** (패치 면적 200 mm² 가정) |
| `l_sparse` (현행) | `w_sparse·gate_mult / (17·n_cand)` ≈ **0.0147**·gate_mult |

작은 패치에서 인공 페널티가 물리 신호를 **압도**한다. 이것이 +10.1%의 메커니즘이다.

> **금지**: rollback / PENDING_DELETE 스냅샷 비교 상태 기계. idea_v4.md §2.4에서 Critic·Defense
> 양쪽이 기각했다. PENDING 진입 시점엔 이미 연속 파트가 두꺼워진 뒤라 counterfactual이 아니며,
> 이력현상과 "삭제→보상→롤백→재삭제" 루프 위험이 있다.

### §2.2 (보류안 B) 면적 가중 `l_sparse`

**1단계 재학습에서 게이트가 중간값(0.3~0.7)에 정체하면**에만 적용한다. 기본은 §2.1(A안)이다.

- **대상**: `train_step_multi()`의 `l_sparse = z_open[:, CANDIDATE_PARTS].mean()`

```python
# [v1.3-B, 조건부] 각 게이트를 자기 파트의 실제 면적 기여로 가중
if USE_AREA_WEIGHTED_SPARSE:
    with torch.no_grad():
        cand_idx = torch.tensor(CANDIDATE_PARTS, device=x.device)
        a_k = _per_candidate_area(new_coords, t_final, z_gate, part_ids, section_ids,
                                  edge_index, edge_attr, cand_idx)      # (17, n_cand)
        a_k = a_k / (a_k.sum() + 1e-9)
    l_sparse = (z_open[:, CANDIDATE_PARTS] * a_k).sum()
else:
    l_sparse = z_open[:, CANDIDATE_PARTS].mean()
```

- `_per_candidate_area()`는 신규 헬퍼로 `@torch.no_grad()` 데코레이트할 것. 정규화된 가중치이므로
  autograd 그래프에 들어가면 안 된다.
- B안 적용 시 `W_SPARSE_V13`는 0.0이 아니라 **0.02~0.05**로 둔다(0이면 항 자체가 사라짐).
  **§1의 가드 assert가 이 조합을 강제한다** — 스위치만 켜고 가중치를 0으로 두면 `contrib_sparse`가
  통째로 0이 되어 아무 에러 없이 B안이 무효화되고, §6.2의 `test_sparse_contribution_is_zero`가
  그 고장난 설정을 **통과시켜 은폐한다**(개정2에서 §6.2에 대칭 테스트 추가).

#### §2.2.1 `_per_candidate_area()` 구현 명세 (개정2 — 초안은 shape만 있고 계산식이 없었다)

네 가지를 반드시 아래대로 고정한다. 구현자 재량에 맡기면 자기강화 루프가 생긴다.

| 항목 | 확정 | 이유 |
|---|---|---|
| 두께 인자 | **`t_raw`** (`t_final` 금지) | `t_final = t_raw · z_gate`이므로 `t_final`을 쓰면 **가중치 자체가 게이트에 의존**한다. 게이트↓ → 가중치↓ → 페널티↓ → 게이트↓ 의 자기강화 루프 |
| `z_gate` 곱셈 | **하지 않는다** | 같은 이유. 시그니처로 받더라도 면적 계산에 쓰지 말 것 |
| 정규화 | **전역** `a_k / (a_k.sum() + 1e-9)` | 섹션별 정규화하면 "면적 기여에 비례"라는 의미가 사라진다 |
| 면적 식 | `usv18.compute_mass_loss`와 동일하게 `Σ seg_len · t_src`, 후보 파트 에지만 마스킹해 **섹션별 합산** | 물리 면적과 단위를 일치시키기 위함 |

```python
@torch.no_grad()
def _per_candidate_area(new_coords, t_raw, part_ids, section_ids,
                        edge_index, cand_idx):
    """(17, n_cand) — 후보 파트별·섹션별 실제 단면적 기여. z_gate를 곱하지 않는다."""
    src, dst = edge_index[0], edge_index[1]
    seg_len  = torch.norm(new_coords[src] - new_coords[dst], dim=1)
    contrib  = seg_len * t_raw[src].squeeze(-1)              # 에지별 면적
    out = torch.zeros(NUM_SECTIONS, len(cand_idx), device=new_coords.device)
    for c, pid in enumerate(cand_idx.tolist()):
        m = (part_ids[src] == pid)
        out[:, c].index_add_(0, section_ids[src][m].long(), contrib[m])
    return out
```

**참고 수치**: 전역 정규화 + `W_SPARSE_V13 = 0.03`이면 게이트 1개당 그래디언트가
`0.03 × (1/34) ≈ 0.00088`로 물리 신호(`0.0083`)의 **약 1/10**이다. B안은 의도적으로 순한 처방이므로
정상이다. 다만 "B안을 켰는데 변화가 미미하다"는 관측이 나올 때 **정상 동작인지 위의 no-op 버그인지
구분**할 수 있어야 하므로, B안 활성 시 `contrib_sparse` 값을 반드시 로그에 찍을 것.

### §2.3 `ALPHA_ENT` 오버라이드 — monkey-patch 금지

- **대상**: `train_step_multi()`의 `contrib_entropy = (usv18.ALPHA_ENT * gate_multiplier) * entropy`
- **TO-BE**: 모듈 상수를 직접 참조하지 말고 **인자로 받는다** (§5에서 0으로 꺼야 하므로).

```python
def train_step_multi(..., alpha_ent=ALPHA_ENT_V13, stage4_active=False):
    ...
    contrib_entropy = (alpha_ent * gate_multiplier) * entropy   # [v1.3] usv18.ALPHA_ENT(0.01) → 0.05
```

> **`usv18.ALPHA_ENT = 0.05` 형태의 monkey-patch는 금지한다.** 숙의 과정에서 "usv18 내부 함수가
> 전역 `ALPHA_ENT`를 참조하므로 로컬 섀도잉은 불일치를 낳는다"는 주장이 나왔으나, 코드 확인 결과
> **거짓**이다. `uni_section_v18.py`에서 `ALPHA_ENT`/`W_SPARSE`를 참조하는 곳은
> `train_step()`, `run_training()`, `visualize_epoch_snapshots()` 세 곳뿐이며,
> `AI_design_v1_2.py`는 이 셋 중 **어느 것도 호출하지 않는다**(자체 `train_step_multi()` /
> `run_training_multi()` 사용). 전역 변이는 이득 없이 부작용만 만든다.
>
> 구현 후 검증: `grep -n "usv18.train_step\|usv18.run_training\|usv18.visualize_epoch_snapshots" AI_design_v1_3.py`
> 가 **빈 결과**여야 한다. 비어 있지 않다면 이 판단을 재검토할 것.

#### §2.3.1 호출부 수정 — 반드시 함께 고칠 것 (개정3 신규)

> **초판의 결함**: 위 TO-BE는 `train_step_multi()`의 **정의부**만 바꾼다. `run_training_multi()`
> 안에서 이 함수를 실제로 호출하는 지점을 함께 고치지 않으면, 정의부는 `alpha_ent`에 기본값
> `ALPHA_ENT_V13`이 항상 들어가 **명목상은 동작하지만 §5.6(Stage4에서 0으로 끄기)이 적용될
> 경로가 없다.** `/synod review` 최종 검토에서 Gemini·OpenAI 양측이 독립적으로 지적했다.

- **대상**: `run_training_multi()`의 본 학습 루프(`while epoch < loop_max:`) 안, `train_step_multi()` 호출

- **AS-IS**

```python
info = train_step_multi(model, data, optimizer, target_mps, target_area,
                         epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                         collision_spec, alda_state, pruning_state, segment_ids=seg_ids)
```

- **TO-BE**

```python
info = train_step_multi(model, data, optimizer, target_mps, target_area,
                         epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                         collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                         alpha_ent=ALPHA_ENT_V13, stage4_active=False)   # [v1.3] §2.3 인자화
```

**Stage4 진입 후의 호출부**(§5.6의 루프)는 이와 별개로 `alpha_ent=alpha_ent_active`(0.0),
`stage4_active=True`를 전달한다 — 두 호출부가 서로 다른 `alpha_ent` 값을 넘긴다는 점을
구현 시 혼동하지 말 것. 구현 후 검증:
`grep -n "train_step_multi(" AI_design_v1_3.py` 결과가 **정확히 2곳**(본 루프, Stage4 루프)이고
**둘 다** `alpha_ent=`, `stage4_active=`를 명시적으로 전달하는지 확인한다.

### §2.4 로그 보강 (필수)

`run_training_multi()`의 epoch 로그에 다음을 추가한다. 1단계 재학습의 관찰 대상이며,
**§5.1의 Stage 4 진입 판정에도 그대로 쓰인다.**

```python
# |log_alpha| > 2.4 이면 z_gate가 γ=-0.1, ζ=1.1 stretched sigmoid에서 정확히 0/1로 포화
la = model.log_alpha_candidates.detach()
sat_ratio = float((la.abs() > 2.4).float().mean())
print(f"  ... | area={float(area):,.1f} | log_alpha_sat={sat_ratio:.2%} "
      f"| |la|_mean={float(la.abs().mean()):.2f}")
```

**판정 기준**: `ALPHA_ENT_V13` 상향이 효과가 있다면 `log_alpha_sat`이 학습 후반에 **상승**해야 한다.
`area`가 감소 추세여야 §2.1이 의도대로 동작한 것이다.

### §2.4.1 ★ 그래디언트 데드존과 전역 사망 원장 (개정2 신규 — idea_v4 §8 체크리스트 2번의 답)

> **초판의 결함**: idea_v4 §8 체크리스트 2번은 "`l_sparse` 제거 후 `log_alpha`에 실제로 그래디언트가
> 흐르는지 (`z_gate`의 `clamp`가 포화 구간에서 그래디언트를 죽임)"를 확인하라고 요구했으나,
> 본 지시서 초판에는 **이에 대한 대응이 §2.4의 print 한 줄 말고 전혀 없었다.**

#### 왜 문제인가

```
z_gate = clamp(sigmoid(la)·(ζ−γ)+γ, 0, 1),  γ=−0.1, ζ=1.1
  sigmoid(la)·1.2 − 0.1 ≥ 1  ⟺  la ≥ +2.398
  sigmoid(la)·1.2 − 0.1 ≤ 0  ⟺  la ≤ −2.398
```

즉 **`|log_alpha| > 2.4`는 `clamp`의 평평한 구간**이며 `∂z_gate/∂log_alpha = 0`이다.
`z_gate`를 지나는 **모든** 손실 경로(`f`, `term_Mp`, `contrib_phys`, 신규 `contrib_mass`)의
그래디언트가 동시에 끊긴다. §2.4가 성공 지표로 쓰는 `2.4`와 그래디언트가 죽는 지점이 **같은 숫자**다.

- `+2.4` 초과 포화 → 파트 존재 확정. 두께·좌표는 계속 학습되므로 **무해**하다.
- `−2.4` 미만 포화 → `z_gate = 0`. 엔트로피 항도 같은 `clamp`를 지나므로 반대 방향으로도 못 민다.
  **양방향으로 잠긴 래칫**이며, EWMA 프루닝(`<0.08` → `DELETED`)과 §5.2 부활 금지가 겹쳐 **영구 삭제**다.

#### 정량 확인 — `ALPHA_ENT = 0.05`는 안전하나 `0.1`은 경계다

`∂(contrib_entropy)/∂z_k = ALPHA_ENT · gate_mult · (−logit(z_k)) / 34` 이므로:

| 게이트 값 | α=0.05 | α=0.1 | 물리 면적 `f` |
|---|---|---|---|
| z = 0.5 | 0 | 0 | 0.0083 |
| z = 0.3 | 0.00125 | 0.0025 | 0.0083 |
| z = 0.1 | 0.0032 | 0.0065 | 0.0083 |
| z = 0.05 | 0.0043 | **0.0087** | 0.0083 |

애매한 구간(z≈0.3)에서 엔트로피는 물리의 **1/7**에 불과하다. 예전 `l_sparse`(0.0147)가 물리를
1.8배 압도했던 것과 성격이 다르므로, **제안 A와 C가 서로를 무력화한다는 우려는 α=0.05에서 성립하지
않는다.** 다만 압력이 `z→0`에서 로그로 커져 **§7 1단계 실패 조치가 지시하는 `ALPHA_ENT = 0.1`에서는
`z≈0.05` 부근에서 물리를 역전**한다. 엔트로피는 잘못된 결정을 *만들지는* 않지만 **이미 죽어가는
게이트를 비가역으로 확정**시킨다. 따라서 α를 0.1로 올릴 때는 아래 원장을 반드시 함께 볼 것.

#### ★ 전역 사망 원장 — 특정 섹션 하드코딩 금지 (사용자 확정, 개정2)

> v1_2에서 sec 10·11·12가 걸린 것은 **우연한 표본**이다. `l_sparse` 병리는 전 섹션·전 후보에
> 균일하게 작용했고(Patch2는 17/17 전멸), 특정 섹션을 감시 조건에 박아 넣으면 다음 실행에서
> 다른 섹션이 죽을 때 감시망이 통째로 빈다. **판정은 전 섹션 × 전 후보에 대해 수행한다.**

0-포화 자체는 제안 C가 **의도한 동작**이므로 그것만으로 중단하면 정상 수렴까지 잡는다.
정상과 병리를 가르는 축은 v1_2 병리의 정의에서 그대로 나온다 —
**"물리가 아직 그 파트를 필요로 하는 상태에서 죽었는가."**

```python
# [v1.3 개정2] 게이트 사망 원장(ledger). 전 섹션 × 전 후보. 특정 섹션 하드코딩 금지.
la    = model.log_alpha_candidates.detach()
alive = (pruning_state['state'] != STATE_DELETED)
dead0 = (la < -2.4) & alive              # 0쪽 포화 = 그래디언트 소멸 = 비가역

newly_dead = dead0 & (~dead0_prev)
for k, c in newly_dead.nonzero().tolist():
    mp_k = float(mp_ratio_per_section[k])                    # 해당 섹션 hard/target
    verdict = "JUSTIFIED" if mp_k >= DEATH_MP_THRESHOLD else "SUSPECT"
    death_log.append({"epoch": epoch, "sec": k, "cand": c,
                      "mp_at_death": mp_k, "area": float(area), "verdict": verdict})
    if verdict == "SUSPECT":
        print(f"  !! SUSPECT DEATH sec{k} cand{c} @ep{epoch} "
              f"(Mp={mp_k:.1%} < {DEATH_MP_THRESHOLD:.0%}, area={float(area):,.0f})")
dead0_prev = dead0.clone()

g = model.log_alpha_candidates.grad
live = (la.abs() <= 2.4) & alive
n_susp = sum(d["verdict"] == "SUSPECT" for d in death_log)
print(f"  ... | dead0={int(dead0.sum())}/{int(alive.sum())} | suspect={n_susp} "
      f"| la_grad_live={float(g[live].abs().mean()) if live.any() else 0.0:.2e}")
```

- `JUSTIFIED`(Mp 충족 상태의 죽음)는 **원하는 동작**이므로 세기만 한다.
- `SUSPECT`(Mp 미달인데 죽음)만 경보 대상이며, **§7 롤백 규칙 R1의 입력**이다.
- `la_grad_live`가 0으로 수렴하면 학습할 게이트가 남아 있지 않다는 뜻이다(정상 종료 신호이거나,
  전량 조기 포화의 징후 — `dead0` 추이와 함께 읽을 것).
- `death_log`는 리포트에 표로 덤프한다(§7 판정 근거이자 다음 세션의 진단 자료).

### §2.5 ALDA 이중상승(dual ascent) 부재 — 인지 사항 (개정 신규)

`usv18.make_alda_state()`는 `mu_Mp`/`rho_Mp`를 갱신하는 dual-ascent 스케줄을 전제하지만,
**그 갱신 코드는 `usv18.run_training()` 안에만 있고 `run_training_multi()`는 이를 호출하지 않는다**
(§2.3에서 monkey-patch를 기각한 바로 그 사실의 이면이다). 따라서 실행 내내

```
mu_Mp = 1.0 (고정),  rho_Mp = 200.0 (고정)
term_Mp = 100 · relu(상대Mp오차 − 0.005)²
```

즉 ALDA는 **이름과 달리 고정계수 penalty**로 동작한다.

- **긍정적 귀결**: `mu`가 폭주해 면적 항 `f`를 압도하는 시나리오는 이 코드에서 **발생하지 않는다.**
  총 상대 Mp 오차가 0.5% 이내로 들어오면 `term_Mp`의 그래디언트가 실제로 소멸하고 `f`가 지배한다.
- **조치**: v1.3에서는 **현행 유지**(추가 변경 없음). 다만 1단계 재학습에서 면적이 목표 근처에
  정체하면, 최소 개입으로 `f`의 계수를 2~5배 올리는 것을 검토한다(§7 1단계 실패 시 조치).
  **10배 이상은 금지** — `asymmetric_huber_phys`가 과다 강성도 penalty하므로 균형이 깨진다.

### §2.6 질량 감량 목표 — 초기 단면적의 95% (사용자 확정 2026-08-08)

**목표: 최종 단면적 ≤ 22,854.1 mm² (= 24,056.9 × 0.95).**

> ### ★ `target_area`만 낮추는 것으로는 목표가 되지 않는다
>
> 유일한 질량 항 `f = area / target_area`는 **선형항**이다. `target_area`를 5% 낮추면 기울기가
> `1/0.95 = 1.053`배가 될 뿐, **"22,854까지 줄여라"는 신호가 아니다.** `f`는 이미 Mp가 허용하는
> 한 계속 아래로 밀고 있었고, 최종 면적은 `f`(하강)와 `contrib_phys` + `term_Mp`(상승)의
> **균형점**에서 결정된다. 정규화 상수를 바꿔도 균형점은 5%밖에 안 움직인다.
>
> 목표선을 실제 제약으로 만들려면 **초과 시 벌점을 받는 항**이 필요하다.

**수단: `l_mass`를 loss에 편입한다.** `usv18.compute_mass_loss()`가 이미 반환하고 있으나
현재 `loss`에 더해지지 않는(§2.1) 항으로, 형태가 정확히 예산 제약이다.

```python
l_mass = torch.relu(area / target_area - 1.0) ** 2      # 목표 초과분만 이차 벌점
```

#### §2.6.1 `target_area` 설정

- **대상**: `run_training_multi()`의 `target_area is None` 분기
- **AS-IS**

```python
if target_area is None:
    target_area, _ = usv18.compute_section_area(x[:, :2].cpu(), x[:, 6:7].cpu(), data.edge_index.cpu())
print(f"[mass] target_area = {target_area:.1f} mm^2 (17-section 합산 초기 단면적)")
```

- **TO-BE**

```python
area_init_ref, _ = usv18.compute_section_area(x[:, :2].cpu(), x[:, 6:7].cpu(), data.edge_index.cpu())
if target_area is None:
    target_area = AREA_TARGET_RATIO * area_init_ref     # [v1.3] §2.6 감량 목표 95%
print(f"[mass] area_init = {area_init_ref:.1f} mm^2 | target_area = {target_area:.1f} mm^2 "
      f"(감량 목표 {AREA_TARGET_RATIO:.0%})")
```

`area_init_ref`는 이후 로그·리포트에서 달성률 표기에 재사용한다.

#### §2.6.2 `l_mass`를 loss에 편입

- **대상**: `train_step_multi()`의 `loss = (...)` 조립부
- **TO-BE**

```python
contrib_mass = weights['w_mass'] * l_mass        # [v1.3] §2.6 감량 목표 예산 제약

loss = (contrib_phys + contrib_smooth + L_alda_effective
        + contrib_order + contrib_anchor + contrib_sat
        + contrib_sparse + contrib_entropy + contrib_continuity
        + contrib_mass)                          # ← 추가
```

`'w_mass': W_MASS_V13`는 **§2.1 TO-BE 블록에 이미 포함되어 있다**(개정2). 여기서 다시 추가하지 말 것.
`info` 반환 dict에 `"contrib_mass": contrib_mass.item()`을 추가해 로그로 추적한다.

##### §2.6.2-a `contrib_mass` 계산문의 정확한 삽입 지점 (개정3 신규)

> **초판의 결함**: 위 TO-BE는 `loss = (...)` 튜플에 `contrib_mass`를 더하라는 것만 보여주고,
> `contrib_mass = weights['w_mass'] * l_mass` **계산문 자체를 어디에 놓을지**는 서술하지 않았다.
> `train_step_multi()`에는 Stage1(좌표 zero-out), Stage2(연속 파트 두께 detach) 등 `epoch` 조건부
> 분기가 여럿 있어, 구현자가 잘못된 분기 안에 넣으면 `contrib_mass`가 특정 stage에서만 계산되고
> 다른 stage에서 `NameError`가 나거나 그래디언트가 새는 사고가 난다.

`l_mass`는 이미 `area, l_mass = usv18.compute_mass_loss(new_coords, t_final, edge_index, edge_attr,
target_area)` 호출로 **stage 조건과 무관하게 매 스텝 계산되고 있다**(v1_2 기존 코드, `l_collision`
계산 직전). `contrib_mass`도 동일하게 **stage-gated 되지 않는다** — Stage1/Stage2/Stage4 어느
epoch에서도 무조건 계산·적용한다(면적 목표는 항상 유효한 제약이므로).

**규칙: `contrib_mass`는 `contrib_phys`/`contrib_smooth`/`contrib_order`/`contrib_anchor`/
`contrib_sat` 등 다른 `contrib_*` 변수들과 정확히 같은 블록, 즉 `loss = (...)` 조립 **바로 앞,
`L_alda`/`L_alda_effective` 계산 다음 줄**에 둔다. 이 블록은 이미 어떤 `epoch` 분기 밖에 있으므로
(Stage1/Stage2 분기는 그보다 앞의 `new_coords`/`t_final` 산출 단계에서 끝난다), `contrib_mass`
계산문을 다른 분기 안에 넣을 필요도, 넣어서도 안 된다.

```python
# TO-BE — 다른 contrib_* 와 같은 위치, L_alda_effective 계산 다음
L_alda, g_Mp_val, g_col_val = usv18.compute_alda_loss(
    area, target_area, pred_mp_total, target_mp_total, l_collision, alda_state)
L_alda_effective = L_alda * max(gate, 0.05)

contrib_phys   = weights['w_phys']   * l_phys_total * s_phys
contrib_smooth = weights['w_smooth'] * l_smooth     * s_smooth
contrib_order  = weights['w_order']  * l_order
contrib_anchor = weights['w_anchor'] * l_anchor
contrib_sat    = weights['w_sat']    * l_sat
contrib_mass   = weights['w_mass']   * l_mass        # [v1.3] §2.6 — 다른 contrib_*와 동일 위치, stage 무관
```

**Stage4와의 관계**: §5.6은 `w_sparse`/`alpha_ent`를 Stage4에서 0으로 끄지만 **`w_mass`는 끄지
않는다** — 면적 감량 목표는 Stage4(좌표·두께 재수렴)에서도 계속 유효한 제약이기 때문이다.
§5.6 표에 `w_mass: 유지(W_MASS_V13 그대로)` 행을 추가한다(아래 §5.6 갱신 참조).

#### §2.6.3 가중치 `W_MASS_V13 = 10.0`의 근거

`d(l_mass)/d(area) = 2·(ratio − 1) / target_area`이므로, 목표를 5% 초과한 지점에서

```
w_mass · d(l_mass)/d(area) = 10 × 2 × 0.05 / 22,854 = 4.4e-5
       f 의 d(f)/d(area)   =              1 / 22,854 = 4.4e-5
```

**목표선 부근에서 하강 압력이 정확히 2배가 된다.** 목표 안쪽에서는 `relu`가 0이라 `f`만 남으므로
과도한 감량으로 폭주하지 않는다. 참고 수치:

| 면적 | ratio | `l_mass` | `contrib_mass` (w=10) |
|---|---|---|---|
| 26,497.8 (v1_2 결과) | 1.159 | 0.0254 | 0.254 |
| 24,056.9 (초기) | 1.053 | 0.00277 | 0.028 |
| 22,854.1 (**목표**) | 1.000 | 0 | **0** |

`W_MASS_V13`는 5~20 범위에서 조정 가능하다. 20을 넘기면 Mp 제약과 경쟁해 sec 11 미달을
악화시킬 수 있으므로 상향 시 §7 3단계 지표를 함께 볼 것.

#### §2.6.4 리포트 baseline은 바뀌지 않는다

`report_design_comparison()`의 `area_init`은 `t_init = x[:, 6]`에서 **독립적으로** 재계산되므로
(`:729`), `target_area`를 낮춰도 initial vs final 비교 기준선은 여전히 실제 설계값
**24,056.9 mm²** 이다. 리포트 면적 표에 다음 행을 추가한다.

```
| 감량 목표 | 22,854.1 mm² (초기 대비 95%) |
| 달성률    | {area_hard / 22854.1 * 100:.1f}% (100% 이하면 목표 달성) |
```

#### §2.6.5 CLI 인자 추가

```python
parser.add_argument("--area-target-ratio", type=float, default=AREA_TARGET_RATIO,
                    help="감량 목표 비율. 1.0 = 초기 단면적 유지, 0.95 = 5%% 감량 (기본)")
parser.add_argument("--w-mass", type=float, default=W_MASS_V13,
                    help="l_mass 가중치. 0.0이면 예산 제약 비활성")
```

#### §2.6.6 ⚠️ 실현 가능성 — 무리하면 되돌릴 것

**95% 감량과 전 섹션 Mp 달성이 동시에 가능한지는 확인되지 않았다.** v1_2 실행은 면적 **+10.1%**
상태에서도 sec 11이 95.0%로 미달이었다. 여기서 면적을 −5%까지 끌어내리면서 sec 11을 98% 위로
올리라는 요구는 **상당히 공격적**이다.

다만 v1_2의 +10.1%는 `l_sparse`가 만든 인공적 결과이므로(§2.1), §2·§3 적용 후의 진짜 하한은
아직 모른다. 1단계 재학습이 이 질문의 답이다.

**완화 순서 (§7 1단계에서 Mp와 면적이 동시에 안 잡힐 때)**

1. `W_MASS_V13` 10.0 → 5.0 (제약을 무르게)
2. `AREA_TARGET_RATIO` 0.95 → **0.97** 재시도
3. 그래도 안 되면 → **1.00**(초기 단면적 유지)으로 후퇴하고, 그 사실을 리포트에 명시
4. **Mp 목표를 낮추는 것은 금지** — 강성은 안전 요구사항이다

역순(먼저 Mp를 포기)으로 완화하지 말 것.

---

## §3. 두께 한계 T_MIN 0.7 / T_MAX 2.3 — **전 구간 도달성 확보**

> **요구사항 (2026-08-08, 사용자 지시)**: 최종 두께는 `[0.7, 2.3]` **전 구간에서 자유롭게
> 결정되어야 한다.** 현행 구조는 각 파트가 자기 초기 두께 주변의 좁은 창에서만 움직이므로
> `[T_MIN, T_MAX]`는 "넘지 못하는 한계"일 뿐 "도달 가능한 범위"가 아니다.
> **`T_MIN`/`T_MAX`만 바꾸는 것으로는 이 요구를 충족할 수 없다. `DELTA_SCALE`을 반드시 함께 바꾼다.**

### §3.1 클래스 속성 오버라이드

- **대상**: `class CGDN17(usv18.CGDN):` 선언 직후 (상수 블록은 §1에 따라 **이보다 앞**에 있어야 함)

```python
class CGDN17(usv18.CGDN):
    T_MIN       = T_MIN_V13        # 0.7   (usv18.CGDN.T_MIN = 0.1)
    T_MAX       = T_MAX_V13        # 2.3   (usv18.CGDN.T_MAX = 2.5)
    DELTA_SCALE = DELTA_SCALE_V13  # 10.0  (usv18.CGDN.DELTA_SCALE = 1.35) — §3.2 필수
```

`usv18.CGDN`은 수정하지 않는다. 서브클래스 속성이 `self.T_MIN`/`self.T_MAX`/`self.DELTA_SCALE`
참조를 모두 덮는다. `usv18.compute_saturation_loss(..., delta_scale=model.DELTA_SCALE)` 역시
`model.DELTA_SCALE`을 인자로 받으므로 자동으로 따라온다(§3.5).

### §3.2 ★ 왜 T_MIN/T_MAX만 바꾸면 안 되는가

두께 매핑식은 다음과 같다 (`CGDN17.forward()`).

```python
t_initial_frac  = (t_initial - t_min) / (t_max - t_min)
t_initial_frac  = torch.clamp(t_initial_frac, 1e-4, 1.0 - 1e-4)
t_initial_logit = torch.logit(t_initial_frac)
t_raw = t_min + (t_max - t_min) * torch.sigmoid(t_initial_logit + delta_t_part)
```

학습되는 변수는 두께가 아니라 **`delta_t_part` = `leaky_tanh(net_out) * DELTA_SCALE`, 즉 logit
공간의 오프셋**이며 `DELTA_SCALE=1.35`에 묶여 있다. 따라서 도달 가능 두께는
**초기 두께가 앉은 logit을 중심으로 ±DELTA_SCALE만큼의 창**뿐이고, `[T_MIN, T_MAX]` 전체는
`delta → ±∞`에서만 도달하는 점근 한계다.

두 가지가 겹쳐 문제가 된다.

1. **창이 너무 좁다** — `DELTA_SCALE=1.35`면 시그모이드 중앙부(Plate, frac=0.5625)에서조차
   도달 범위가 `[1.100, 2.032]`로, `[0.7, 2.3]`의 58%밖에 안 된다.
2. **Outer Hat은 아예 동결된다** — 초기 두께가 정확히 2.300 = 새 `T_MAX`라 `frac = 1.0`,
   `clamp(1-1e-4)`에 걸려 `logit ≈ 9.21`이라는 극단 포화 지점에 앉는다. 여기서는 logit을 ±1.35
   움직여도 sigmoid 출력이 0.99961 → 0.99997로 0.00036밖에 안 변한다.

**실측 도달 범위** (`t_raw = T_MIN + 1.6 × sigmoid(logit(clamp(frac)) + delta)`, `delta ∈ [-D, +D]`)

| 설정 | Outer (2.300) | Plate/Inner (1.600) | Patch1 (1.400) | 판정 |
|---|---|---|---|---|
| 현행 (0.1, 2.5), clamp 1e-4, D=1.35 | [1.877, 2.445] | — | — | 참고 |
| (0.7, 2.3), clamp 1e-4, **D=1.35** | **[2.2994, 2.3000]** | [1.100, 2.032] | [0.969, 1.900] | **동결 + 전 구간 미달** |
| (0.7, 2.3), clamp 0.01/0.99, D=1.35 | [2.240, 2.296] | [1.100, 2.032] | [0.969, 1.900] | 동결만 해소, 여전히 미달 |
| (0.7, 2.3), clamp 0.01/0.99, **D=5.0** | [1.340, 2.300] | [0.714, 2.292] | [0.708, 2.286] | Outer만 하방 부족 |
| **(0.7, 2.3), clamp 0.01/0.99, D=10.0** | **[0.707, 2.300]** | **[0.700, 2.300]** | **[0.700, 2.300]** | **★ 채택** |

`DELTA_SCALE = 10.0` + `FRAC_CLAMP = [0.01, 0.99]`에서 **모든 파트가 [0.700, 2.300] 전 구간을
0.001mm 오차 이내로 커버**한다. `FRAC_CLAMP`는 앵커 logit을 `±4.60`으로 제한해 Outer Hat이
포화점에 앉는 것을 막는 역할이고, 전 구간 도달성은 `DELTA_SCALE`이 만든다. **둘 다 필요하다.**

> ### ⚠️ 도달 범위 표의 유효 조건 (개정 추가)
>
> 위 수치는 `delta_t_part = ±DELTA_SCALE`을 가정한 **정적 상한**이다. 실제로는 두 가지가 더 곱해진다.
>
> 1. **`thickness_gate` 승수** — `forward()`에서 `delta_t_part = delta_t_part * thickness_gate`이고
>    `thickness_gate = usv18.thickness_gate_value(epoch) = sigmoid(0.6·(epoch − 128))`이다.
>    따라서 표의 범위는 **epoch ≳ 150 이후**에만 유효하다(그 시점 승수 ≈ 1.0).
> 2. **그룹 평균 pooling** — `delta_t_part`는 `group_mean[inverse]`, 즉 같은 두께 그룹에 속한
>    전 노드의 `leaky_tanh(net_out)` **평균**이다. 극단값에 도달하려면 그룹 내 노드가 **함께**
>    포화해야 한다. 개별 노드 기준의 도달성이 아니다.
>
> 이 두 조건은 §3.4 assert에 반영하지 않는다(학습 전 1회 정적 설계 검사이므로).
> 대신 **리포트에 실측 최종 두께를 찍어 표와 대조**한다(§3.6).

**epoch-0 두께 오차**: Outer Hat만 2.300 → **2.284** (−0.016mm, −0.7%). `frac`이 0.99로 클램프되기
때문이며, 다른 파트는 `frac`이 0.4375~0.5625라 **오차 0**이다.

> 리포트 baseline은 영향받지 않는다. `report_design_comparison()`의 `mp_init`, `area_init`은
> `t_init = x[:, 6]`(노드 피처 원본)을 직접 쓰며 `t_raw`를 거치지 않는다. 즉 initial vs final
> 비교의 기준선은 여전히 실제 설계값 2.300mm이고, 클램프가 바꾸는 것은 옵티마이저의 출발점뿐이다.
> 이 사실은 리포트에 명시한다(§3.6).

**기각된 대안**

- `T_MAX`를 2.35/2.4로 올려 Outer를 내부에 두기 → 요구 2 위반.
- 동결을 "의도된 하드 제약"으로 수용 → 전 구간 자유 결정이라는 요구와 정면으로 배치된다.
- 시그모이드 중심을 0.5로 재정렬(앵커 제거) → 전 파트 epoch-0 두께가 1.5mm가 되어 초기 설계
  형상이 무너진다. **기각.**
- `FRAC_CLAMP_HI`만 0.95로 낮추기 → 도달 범위가 `[2.030, 2.279]`에 그친다. **폐기.**

### §3.3 수정 — **반드시 두 곳 모두**

`frac` 계산식은 코드에 **두 번** 나타난다. 한쪽만 고치면 리포트의 `t_raw` 복원이 학습과 어긋나
soft/hard Mp 비교가 조용히 틀어진다.

**(1) `CGDN17.forward()`** — 두께 매핑 블록
**(2) `report_design_comparison()`** — `t_raw` 복원 블록

```python
# [v1.3] 두 곳이 갈라지는 사고를 구조적으로 막기 위해 헬퍼로 추출한다 (필수).
def _t_init_logit(t_init, t_min, t_max):
    frac = torch.clamp((t_init - t_min) / (t_max - t_min), FRAC_CLAMP_LO, FRAC_CLAMP_HI)
    return torch.logit(frac)
```

`forward()`와 `report_design_comparison()` 양쪽 모두 **이 헬퍼만 호출**한다.
`1e-4` 리터럴이 남아 있지 않은지 확인할 것: `grep -n "1e-4, 1.0 - 1e-4" AI_design_v1_3.py` → 빈 결과.

### §3.4 초기값 가드 — 범위 검사가 아니라 **전 구간 도달성 검사**

`build_bpillar_17section()` 호출 직후에 아래 가드를 넣는다.

```python
@torch.no_grad()
def assert_thickness_reachability(x, t_min=T_MIN_V13, t_max=T_MAX_V13,
                                  delta_scale=DELTA_SCALE_V13, tol=0.01):
    """[v1.3] 모든 노드가 [T_MIN+tol, T_MAX-tol] 전 구간에 도달 가능한지 검증한다.
    단순 범위 검사(t_init in [T_MIN,T_MAX])는 §3.2의 동결을 통과시키므로 사용하지 않는다.
    주의: thickness_gate 승수와 그룹 평균 pooling은 반영하지 않은 정적 상한이다(§3.2 유효 조건)."""
    t_init = x[:, 6]
    assert (t_init >= t_min - 1e-6).all() and (t_init <= t_max + 1e-6).all(), \
        f"초기 두께가 [{t_min}, {t_max}] 밖: min={t_init.min():.3f} max={t_init.max():.3f}"

    logit = _t_init_logit(t_init, t_min, t_max)
    lo = t_min + (t_max - t_min) * torch.sigmoid(logit - delta_scale)
    hi = t_min + (t_max - t_min) * torch.sigmoid(logit + delta_scale)

    bad_lo = lo > t_min + tol
    bad_hi = hi < t_max - tol
    assert not bad_lo.any(), (
        f"[Reachability] 하한 미달 노드 {int(bad_lo.sum())}개. "
        f"최악 lo={float(lo.max()):.4f}mm > {t_min + tol:.2f}mm "
        f"(t_init={float(t_init[lo.argmax()]):.3f}). "
        f"DELTA_SCALE({delta_scale})을 올릴 것 (command_v1_3.md §3.2)")
    assert not bad_hi.any(), (
        f"[Reachability] 상한 미달 노드 {int(bad_hi.sum())}개. "
        f"최악 hi={float(hi.min()):.4f}mm < {t_max - tol:.2f}mm "
        f"(t_init={float(t_init[hi.argmin()]):.3f}). "
        f"DELTA_SCALE({delta_scale})을 올릴 것 (command_v1_3.md §3.2)")
    return lo, hi
```

`DELTA_SCALE_V13 = 10.0`, `FRAC_CLAMP = [0.01, 0.99]`에서 `tol = 0.01`로 통과함을 실측 확인했다
(최악 노드 Outer Hat: lo=0.7072, hi=2.3000).

### §3.5 `DELTA_SCALE` 상향의 부작용 (반드시 인지할 것)

`DELTA_SCALE` 1.35 → 10.0은 7.4배 변화다.

1. **`l_sat`는 자동 추종하며, 충돌은 무시 가능하다.** *(개정 — 초안의 우려를 정량 검토 후 하향)*
   `compute_saturation_loss(delta_t_part, delta_scale=model.DELTA_SCALE, knee=0.9)`는
   `threshold = 0.9 × delta_scale`이므로 knee가 1.215 → 9.0으로 비례해 따라간다.
   도달 범위 끝(`|δ| ≈ 10`)은 knee를 넘지만, 기여는 `w_sat × relu(1)² = 0.01 × 1 = 0.01`에 불과하다.
   `contrib_phys`(w=10.0), `f`(≈1.0) 대비 **1/1000 이하**이므로 두께가 경계로 가는 것을 막지 못한다.
   **`w_sat` 재조정 불필요.** (숙의 라운드에서 Gemini·OpenAI 양측이 정량 계산 후 합의)

2. **두께 방향 그래디언트가 약 7배 강해진다.**
   `delta_t_part = leaky_tanh(net_out) × DELTA_SCALE`이고 `leaky_tanh'(0) ≈ 0.967`이므로
   `∂δ/∂net_out`이 1.31 → 9.67이 된다. 두께가 초반에 튀거나 진동하면
   **`DELTA_SCALE`을 되돌리지 말고** 다음 순서로 대응한다.
   - `grad_clip = 10.0` (기존값) 유지 여부 확인
   - `optimizer`의 `thickness_decoder` 그룹 lr을 `0.3 ~ 0.5배`로 분리 조정
   - 그래도 불안정하면 `DELTA_SCALE = 8.0` (도달 범위 [0.751, 2.300], Outer 하방 0.05mm 손실)

3. **두께가 전 구간을 쓰게 되므로 해의 성격이 바뀐다.** 이전에는 두께가 초기값 근처에 갇혀
   Mp 조정을 좌표가 떠맡았다. 이제 두께가 0.7까지 내려갈 수 있으므로, §2와 결합하면
   **연속 파트가 얇아지는 방향으로 해가 크게 이동할 수 있다.** 의도된 것이지만,
   `l_collision`·`l_smooth`·`l_order`가 새 영역에서도 유효한지 1단계 재학습에서 확인해야 한다.

4. **`T_MIN = 0.7`의 역효과 — 관측 필수** *(개정 신규)*
   살아남은 패치는 최소 0.7mm를 부과받는다. idea_v4 §5 D.2는 이를 "애매한 얇은 패치가 사라져
   존재 결정이 선명해진다"는 상승작용으로 봤으나, **연속 부재 두께 분할이 금지된 지금(§5.0)
   패치는 유일한 국소 보강 수단**이다. 최적화가 "0.7mm를 감수할 만큼 sec 11이 급한가"를
   저울질하다 패치를 포기할 위험이 있다.
   → **1단계·2단계 재학습에서 §2.4.1 사망 원장의 `SUSPECT` 사망을 최우선 관측**한다
     (전 섹션 × 전 후보. 특정 섹션을 지정하지 않는다 — 사용자 확정, 개정2).

#### §3.5-4-a 조치 사다리 (개정2 신규 — 초판은 "재검토"에서 문장이 끝났다)

> **`T_MIN`/`T_MAX`는 전 파트 공통으로 적용한다(생산성 고려, 사용자 확정 2026-08-08).**
> 파트 클래스별로 다른 하한을 두는 안은 **기각**되었다. 따라서 `T_MIN = 0.7`은 **불변**이며,
> "`T_MIN`을 의심하라"는 **진단 문구일 뿐 조치 대상이 아니다.**

```
SUSPECT 사망이 관측되면 (전 섹션 기준, §7 R1):

[원인 판별]  1단계는 T_MIN이 아직 0.1, 2단계에서 0.7이 되므로 교락 없이 갈린다.
  - 1단계 생존 / 2단계 사망 → T_MIN(0.7 진입 장벽)이 원인
  - 1단계부터 사망          → §2(ALPHA_ENT / §2.4.1 데드존)가 원인

[T_MIN이 원인인 경우]  T_MIN 자체는 제조 제약이므로 변경 금지.
  패치가 0.7mm를 감당할 여지를 질량 예산 쪽에서 만든다 (§2.6.6과 동일 사다리):
   1. W_MASS_V13        10.0 → 5.0
   2. AREA_TARGET_RATIO 0.95 → 0.97
   3. AREA_TARGET_RATIO      → 1.00 (초기 단면적 유지) + 리포트에 명시
   4. 그래도 패치가 죽고 Mp가 미달이면 → 재학습 중단, 사용자에게 보고.
      "T_MIN=0.7 / 전 섹션 Mp ≥ 98% / 감량"이 동시에 성립하지 않는다는 것은
      설계 제약 자체의 문제이며 구현 에이전트가 판단할 사항이 아니다.
  ※ 어느 단계에서도 Mp 목표를 낮추는 것은 금지.

[§2가 원인인 경우]  §7 롤백 규칙의 조치를 따른다(ALPHA_ENT 0.02로 하향).
```

### §3.6 리포트 보강 (필수)

`report_design_comparison()`의 두께 표에 다음 열/각주를 추가한다.

- 각 파트별 `t_init`(설계값) / `epoch-0 t_raw` / `final t_raw` / 도달 가능 범위 `[lo, hi]`
- 각주: "`FRAC_CLAMP=[0.01, 0.99]`, `DELTA_SCALE=10.0` 적용. 전 파트가 [0.700, 2.300] 전 구간
  도달 가능(정적 상한, §3.2 유효 조건 참조). Outer Hat의 옵티마이저 출발점은 2.284mm이나
  Mp/면적 baseline은 실제 설계값 2.300mm 기준이다 (command_v1_3.md §3.2)."

### §3.7 기존 가중치 폐기

`weights/bpillar_17sec_v1_2.pt`는 **재사용 금지**(Outer Hat 2.394 > 새 T_MAX 2.3). 반드시 재학습한다.
로드 경로가 있다면 v1.3에서는 새 파일명(`bpillar_17sec_v1_3.pt`)으로 분리한다.

---

## §4. 이진화 임계 정렬 — `BINARIZE_THRESHOLD = 0.3`

- **대상**: `report_design_comparison()`의 `exists_cand` 계산
- **AS-IS**

```python
exists_cand = (state.to(device) != STATE_DELETED) & (z_gate[:, cand_idx] > 0.5)
```

- **TO-BE**

```python
exists_cand = (state.to(device) != STATE_DELETED) & (z_gate[:, cand_idx] >= BINARIZE_THRESHOLD)
```

`0.5` 리터럴이 남아 있는 곳이 더 없는지 확인할 것(시각화 함수 포함):
`grep -n "z_gate.*> *0\.5\|> *0\.5.*z_gate" AI_design_v1_3.py`

> **비교 연산자 통일**: §5.2의 강제 이진화와 동일하게 **`>=`** 를 쓴다. 경계값 0.3에서 두 곳의
> 판정이 갈리면 Stage 4 종료 assert(§5.7)가 설명 불가능하게 실패한다.

### §4.1 기대치 조정 — 반드시 문서/리포트에 남길 것

`exists = (state != DELETED) AND (z_gate >= threshold)`이므로 **이미 `DELETED` 확정된 게이트는
임계값을 낮춰도 되살아나지 않는다.** 현행 실행 기준 효과 범위:

| 대상 | 상태 | 임계 0.3의 효과 |
|---|---|---|
| Patch1 sec 0~9 | DELETED 확정 | **없음** |
| Patch2 전 섹션 | DELETED 확정 | **없음** |
| **Patch1 sec 10·11·12** | ALIVE, z_gate < 0.5 | **여기에만 효과** |

즉 §4 단독으로는 섹션 11 정도만 구제된다. 근본 해결은 §2(애초에 잘못 죽지 않게 하는 것)이며,
**§2 없이 §4만 적용하면 정당화되지 않은 질량이 추가될 뿐이다. A → B 순서를 지킬 것.**

> **금지**: `STATE_DELETED` 부활 / 프루닝 임계(0.08) 완화. `DELETED` 비가역성은
> `compute_segment_ids()`의 1D CCL이 "그룹은 단조 세분화만 된다(재병합 없음)"고 가정하는 근거다.
> 부활시키면 run 그룹이 재병합되어 두께 pooling 키가 학습 도중 뒤바뀐다.
> **이 금지는 §5.2의 강제 이진화에도 그대로 적용된다** (사용자 확정, 2026-08-08).

### §4.2 진단 출력 추가

리포트에 "죽음의 구간 잔존 게이트" 표를 추가한다. 이것이 §4의 효과 측정 지표이자
**§5.1 Stage 4 진입 판정의 보조 지표**다.

```python
zone = (state.to(device) != STATE_DELETED) & (z_gate[:, cand_idx] > 0.08) \
                                           & (z_gate[:, cand_idx] < 0.5)
lines.append(f"\n- 죽음의 구간(0.08 < z_gate < 0.5) 잔존 게이트: **{int(zone.sum())}개** "
             f"(임계 {BINARIZE_THRESHOLD} 적용 시 구제: "
             f"{int(((z_gate[:, cand_idx] >= BINARIZE_THRESHOLD) & zone).sum())}개)")
```

---

## §5. Stage 4 — 존재 확정 후 형상·두께 재수렴 (전면 개정)

> **개정 사유**: 초안 §5는 (a) 연속 부재 run pooling을 제안했으나 절단 규칙 미정의로 **no-op**임을
> 자인했고, (b) 옵티마이저 재빌드가 **존재하지 않는 변수명**을 참조했으며, (c) 기본 비활성이라
> idea_v4 §6.1의 3개 모델 합의(“좌표만으로는 불충분”)에 대한 답이 사실상 비어 있었다.
> 사용자 결정(2026-08-08)에 따라 아래와 같이 재정의한다.

### §5.0 성격 및 제조 제약

Stage 4는 **파트 존재 여부 학습을 종료**하고, 확정된 구성 위에서 **좌표와 두께만** 재수렴시키는
독립 구간이다. 게이트가 상수가 되므로 **학습이 보는 구조 = 제조되는 구조**이며, §0의 세 번째
병리("조용한 죽음의 구간")가 확률적 개선이 아니라 **구조적으로 소멸**한다.

> ### ★ 연속 부재 두께 그룹 분할 금지 (사용자 확정)
>
> Outer Hat / Inner / Plate는 전 섹션에 존재하는 **단일 판재**이므로 **부재당 두께는 하나**다.
> 두께 그룹 분할(run pooling)은 **Patch1 / Patch2에만** 적용한다.
>
> 현행 코드가 이미 이 규칙이며 **변경하지 않는다**:
> ```python
> composite_key = torch.where(is_continuous, part_ids_local, run_key)
> #                             연속 부재는 part_id 단일 키 유지 ↑
> ```
>
> **초안 §5.5(연속 파트 run pooling)는 삭제한다.** 절단선을 넣으려면 TWB(테일러드 블랭크)를
> 전제해야 하고 용접선 위치가 새 설계 변수가 되므로, 본 프로젝트 범위 밖이다.
>
> **귀결**: 섹션별 국소 보강 수단은 **패치 존재 여부 + 좌표(형상)** 둘뿐이다. 따라서
> Mp 미달 섹션 구제의 정상 경로는 "해당 섹션의 Patch를 살리는 것"이며(v1_2에서는 sec 10·11·12가
> 여기 해당했다), 그것은 §2·§4가 하는 일이다.
> **Stage 4는 존재를 확정할 뿐 없던 패치를 만들지 못한다**(§5.8 한계 참조).

### §5.1 진입 조건 — 자동 판정

본 학습 루프(`while epoch < loop_max`) **종료 직후** 판정한다.
`curriculum_max_epochs`는 절대 건드리지 않는다(스케줄 불연속 방지, v1.1 결정 사항).

> ### ★ `stage4_start_epoch` 정의 (개정3 — 개정2의 `final_epoch` 재사용에서 개명)
>
> 루프가 `epoch`를 언제 증가시키는지, adaptive extension이 `loop_max`를 갱신했는지에 따라
> 탈출 시점 `epoch` 값이 달라진다. **루프 바디 안에서 마지막 실행 epoch를 명시적으로 기록**해
> off-by-one을 구조적으로 제거한다.
>
> ```python
> # 본 학습 루프
> last_epoch = -1
> while epoch < loop_max:
>     ...
>     last_epoch = epoch          # [v1.3] 실제로 실행된 마지막 epoch
>     epoch += 1
> ```
>
> `curriculum_max_epochs`를 골랐다가 extension으로 그보다 더 돌았던 경우
> `range(stage4_start, stage4_end)`가 **이미 지나온 epoch를 되감아** 스케줄 함수가 과거 값을
> 반환한다(§5.6이 경고한 바로 그 사고). 아래 assert가 이를 잡는다.
>
> **⚠️ 개명 사유 (개정3 — `/synod review` 최종 검토에서 적발)**: 개정2는 이 값을 `final_epoch`로
> 명명했으나, `AI_design_v1_2.py`의 `run_training_multi()`에는 **이미 동명의 `final_epoch`가
> 존재한다**(루프 안 `if epoch == loop_max - 1: final_epoch = epoch`, 리포트의
> `usv18.compute_gate_temperature(final_epoch)` 호출에 쓰인다). 두 변수는 **의미도 값도 다르다**
> (기존: "마지막으로 학습이 실제로 수행된 epoch 번호" = `last_epoch`와 동일 개념·"다음 epoch"이
> 아님 / 신규: "Stage4가 시작하는 다음 epoch 번호" = `last_epoch + 1`). 같은 이름을 재사용하면
> 구현자가 리포트의 temperature 계산에 Stage4 시작점 값을 잘못 흘려 넣어 `z_gate` 하드 샘플이
> 학습 종료 시점과 어긋나는 사고가 난다(v1.2 개발 당시 실제로 겪었던 것과 같은 종류의 버그,
> 헤더 주석 §4.5 참조). **본 개정에서 신규 변수는 `stage4_start_epoch`로 개명**하고,
> 기존 `final_epoch`(리포트용)는 이름과 값 모두 손대지 않는다.

```python
# [v1.3] Stage 4는 adaptive extension이 모두 끝난 뒤 시작하는 독립 구간
la = model.log_alpha_candidates.detach().cpu()
alive = (pruning_state['state'] != STATE_DELETED)              # (17, n_cand)
ambiguous = int((alive & (la.abs() <= 2.4)).sum())             # 0/1로 포화하지 않은 생존 게이트

if ENABLE_STAGE4 is None:
    enter_stage4 = (ambiguous > 0)      # 자동: 애매한 게이트가 하나라도 있으면 진입
else:
    enter_stage4 = bool(ENABLE_STAGE4)  # 강제 on/off

stage4_start_epoch = last_epoch + 1  # [v1.3 개정3] 루프 탈출 시점 기준. 다음에 쓸 첫 epoch 번호.
                                      # 기존 리포트용 `final_epoch`와는 별개 변수 — 절대 혼용 금지.
stage4_start  = stage4_start_epoch
stage4_end    = stage4_start_epoch + STAGE4_EPOCHS
stage4_active = False

assert last_epoch >= 0, "본 학습 루프가 한 번도 실행되지 않았다"
assert stage4_start_epoch >= curriculum_max_epochs, \
    f"Stage4 시작점({stage4_start_epoch})이 커리큘럼 종료({curriculum_max_epochs})보다 앞선다 — 스케줄 되감김"
```

`|log_alpha| > 2.4`면 `z_gate`가 γ=−0.1, ζ=1.1 stretched sigmoid에서 정확히 0/1로 포화한다(§2.4).

### §5.2 강제 이진화 — DELETED는 부활시키지 않는다

```python
if enter_stage4:
    print(f">>> Stage 4: 존재 확정 + 형상/두께 재수렴 ({stage4_start} → {stage4_end}), "
          f"애매 게이트 {ambiguous}개")

    with torch.no_grad():
        model.eval()                                   # ★ HardConcrete 난수 차단 (필수)
        z_gate_final, _ = model.compute_gates_multi(
            training=False, temperature=usv18.compute_gate_temperature(stage4_start))
        model.train()

        cand_idx = torch.tensor(CANDIDATE_PARTS, device=device)
        zc    = z_gate_final[:, cand_idx].detach().cpu()          # (17, n_cand)
        alive = (pruning_state['state'] != STATE_DELETED)         # (17, n_cand)

        # ★ 결정 ①(사용자 확정): DELETED 부활 없음. state와 임계값을 AND로 결합.
        hard_exists = alive & (zc >= BINARIZE_THRESHOLD)          # 0.3

        for k in range(NUM_SECTIONS):
            for c in range(len(CANDIDATE_PARTS)):
                model.log_alpha_candidates[k, c] = 5.0 if hard_exists[k, c] else -5.0

    model.log_alpha_candidates.requires_grad_(False)
```

`|log_alpha| = 5 > 2.4`이므로 `z_gate`가 **정확히 0.0 / 1.0으로 포화**한다. 근사가 아니다.

> **부활 금지의 귀결 — 리포트에 명시할 것**: `hard_exists`는 기존 존재맵의 **부분집합**이므로
> run은 **더 쪼개지기만 하고 병합되지 않는다.** 실질적으로 "분할"만 발생한다.
> 이는 §4.1의 "그룹은 단조 세분화만 된다"는 CCL 가정과 **일치**하므로 충돌하지 않는다.

### §5.3 파트 재분할 — `compute_segment_ids()` 시그니처 확장

현행 함수는 `pruning_state`만 받고 `STATE_DELETED`만 절단점으로 삼는다. 0.3 미만으로 죽이기로 한
섹션이 반영되지 않으면 그룹이 그대로여서 재분할이 no-op이 된다.

- **대상**: `AI_design_v1_2.py`의 `def compute_segment_ids(pruning_state)`

```python
def compute_segment_ids(pruning_state, hard_exists=None):
    """[v1.3] hard_exists가 주어지면 그것을 존재맵으로 삼아 run을 자른다(Stage 4).
    None이면 기존 동작(STATE_DELETED 기준) — 본 학습 루프 호출부는 수정 불필요."""
    if hard_exists is None:
        cut = (pruning_state['state'] == STATE_DELETED)
    else:
        cut = ~hard_exists
    ...  # 이하 1D CCL 로직 동일
```

```python
    seg_ids = compute_segment_ids(pruning_state, hard_exists=hard_exists)
```

`hard_exists=None` 기본값을 유지해 **본 학습 루프의 호출부는 수정하지 않는다.**

### §5.4 학습 대상 — 좌표 + 전 파트 두께 (사용자 확정)

게이트만 얼리고, 나머지는 전부 연다.

| 대상 | Stage 4 | 근거 |
|---|---|---|
| `log_alpha_candidates` | **동결** (`requires_grad_(False)`) | 파트 구성 확정 |
| 좌표 (`coord_decoder` 포함 `main`) | 학습 | 단면 형상 |
| **연속 부재 두께** | **학습 (detach 해제)** | 부재당 스칼라 1개 → 섹션 간 진동 원천 불가, 제조성 위험 없음 |
| 패치 두께 (run 단위) | 학습 | 확정된 run 구조 위에서 국소 조절 |

두께 범위 `[0.7, 2.3]`은 **별도 clip이 필요 없다.** `t_raw = T_MIN + (T_MAX−T_MIN)·sigmoid(·)`
구조상 하드 제약이며, §3의 `DELTA_SCALE=10.0` + `FRAC_CLAMP`로 전 구간 도달성이 확보되어 있다.

- **대상**: `train_step_multi()`의 Stage2 detach 블록
- **AS-IS**

```python
if epoch >= ASWC_STAGE2_END:
    continuous_mask = torch.isin(part_ids.long(), torch.tensor(CONTINUOUS_PARTS, device=x.device))
    t_final = torch.where(continuous_mask.unsqueeze(1), t_final.detach(), t_final)
```

- **TO-BE**

```python
if epoch >= ASWC_STAGE2_END and not stage4_active:      # [v1.3] Stage 4는 두께 재학습 허용
    continuous_mask = torch.isin(part_ids.long(), torch.tensor(CONTINUOUS_PARTS, device=x.device))
    t_final = torch.where(continuous_mask.unsqueeze(1), t_final.detach(), t_final)
```

`stage4_active`는 `train_step_multi()`의 **keyword 인자로 전달**한다(전역 변수 참조 금지).

### §5.5 옵티마이저 재빌드 — 실제 변수명 사용

> ### ⚠️ 초안의 치명적 오류 (수정)
>
> 초안은 `coord_params` / `thickness_params`를 참조했으나 **두 이름 모두 코드에 존재하지 않는다.**
> 실제 변수는 `main_params` / `thick_params` / `gate_params`이며(그룹명 `'main'`,
> `'thickness_decoder'`, `'gate_params'`), **좌표 디코더는 `main_params` 안에 GNN 트렁크·
> node_encoder·FiLM과 섞여 있어 분리되지 않는다.** 분리를 시도하지 말고 기존 그룹을 그대로 쓰되
> `gate_params`만 제외한다.

```python
    optimizer = optim.AdamW(
        [{'params': main_params,  'name': 'main',              'lr': lr * STAGE4_LR_SCALE},
         {'params': thick_params, 'name': 'thickness_decoder', 'lr': lr * STAGE4_LR_SCALE}],
        lr=lr * STAGE4_LR_SCALE, weight_decay=1e-4)
```

**옵티마이저 객체를 새로 만들어야 한다.** `param_groups`의 lr만 바꾸는 것으로는 부족하다 —
Stage 4는 (a) 게이트를 상수화하고 (b) sparsity/entropy 항을 제거하며 (c) 연속 부재 detach를 푸는,
손실 지형의 **불연속 변화**이므로 AdamW의 1·2차 모멘텀이 남아 있으면 과거 그래디언트 방향의
관성으로 파라미터가 튄다.

`gate_params` 그룹이 사라지지만, `GATE_ACTIVE_EPOCH` 처리(`group.get('name') == 'gate_params'`)는
`gate_stage_done`이 이미 `True`라 재실행되지 않는다. 구현 후 이 플래그를 확인할 것.

### §5.6 스케줄 처리 — "학습 초기화"의 정확한 범위 (결정 ②)

**결정 ② = B안(옵티마이저 + 스케줄 재시작, 모델 가중치 유지).**
단, **커리큘럼 가중치는 재시작하지 않고 최종값으로 고정한다.**

`get_curriculum_weights_v10(epoch, max_epochs, ratio)`를 재시작하면 `s_phys`가 초기 단계 값으로
되돌아가 **물리(Mp) 항이 약해진다.** Stage 4의 목적이 Mp 정합인데 정반대로 간다.

| 스케줄 | Stage 4 처리 |
|---|---|
| LR / AdamW 모멘텀 | **재시작** (`lr × STAGE4_LR_SCALE`부터, §5.5) |
| 모델 가중치 | **유지** (1~3단계 형상을 이어받는다. 랜덤 재초기화 금지) |
| `get_curriculum_weights_v10` (`s_phys`, `s_smooth`) | **최종값 고정** |
| `thickness_gate_value(epoch)` | **1.0 고정** |
| `continuity_weight_schedule` | **최종값 고정** |
| `compute_gate_temperature` | 무의미(게이트 동결). 아무 값 |
| `weights['w_sparse']` | **0.0** — 게이트가 상수이므로 항이 무의미 |
| `alpha_ent` | **0.0** — 동일 이유 |
| `weights['w_mass']` | **유지**(`W_MASS_V13` 그대로, 끄지 않는다) — 면적 감량 목표(§2.6)는 게이트 존재 확정과 무관하게 Stage 4의 좌표·두께 재수렴에도 계속 적용되는 제약이다(§2.6.2-a) |

구현은 `stage4_active=True` 플래그 하나로 위 전부를 분기시킨다.
**전역 `epoch`가 계속 증가하는 상태에서 스케줄 함수를 그대로 호출하면 의도치 않은 값이 들어온다.**

> ### ★ 커리큘럼 고정의 구현 방식 (개정2)
>
> Stage 4의 `ep`는 `curriculum_max_epochs`를 **넘어간다**. 따라서 `get_curriculum_weights_v10(ep, ...)`을
> 그대로 호출하면 범위 밖 외삽이 된다. **루프 탈출 시점의 가중치를 1회 계산해 그 튜플을 재사용**한다.
>
> ```python
> frozen_w = get_curriculum_weights_v10(last_epoch, curriculum_max_epochs, ratio)   # 1회 계산 후 고정
> # Stage 4 전 구간에서 스케줄 함수를 다시 호출하지 않고 frozen_w를 그대로 넘긴다
> ```

```python
    weights['w_sparse'] = 0.0
    alpha_ent_active    = 0.0
    stage4_active       = True

    for ep in range(stage4_start, stage4_end):
        info = train_step_multi(model, data, optimizer, target_mps, target_area,
                                ..., epoch=ep, segment_ids=seg_ids,
                                alpha_ent=alpha_ent_active, stage4_active=True)
```

### §5.7 종료 assert

```python
# Stage 4 종료 시: 게이트가 0/1로 고정되었으므로 정의상 성립해야 한다
assert torch.allclose(t_soft, t_hard, atol=1e-6), \
    f"Stage 4 종료 후에도 soft≠hard (max diff={float((t_soft-t_hard).abs().max()):.3e})"
```

`atol`을 1e-2/1e-3으로 느슨하게 잡자는 제안은 **기각**한다. `log_alpha = ±5`면 `z_gate`는
stretched sigmoid에서 정확히 0.0/1.0으로 포화하므로 `t_soft`와 `t_hard`는 부동소수점 오차 수준으로
동일해야 한다. 1e-6에서 실패한다면 이진화 고정 자체가 새는 것이므로 tolerance를 올려 덮지 말고
원인을 찾을 것(가장 흔한 원인: §4의 `>` / §5.2의 `>=` 비교 연산자 불일치).

### §5.8 Stage 4의 한계 — 문서·리포트에 명시할 것

1. **없던 패치를 만들지 못한다.** `hard_exists`는 기존 존재맵의 부분집합이므로, 3단계 종료 시점에
   **어느 섹션이든** 그 Patch가 이미 `DELETED`로 확정돼 있으면 **Stage 4로도 구제 불가**다
   (v1_2에서는 sec 10·11·12가 여기 해당했으나, 이는 표본일 뿐 조건은 전 섹션 공통이다).
   → Mp 미달 섹션의 실질적 해결은 §2·§3 재학습에서 결정된다(§2.4.1, §3.5-4, §7).
2. **연속 부재의 섹션별 보정은 불가능하다**(§5.0). Stage 4가 여는 연속 부재 두께는 부재당
   스칼라 1개이므로 **전체 질량 조절**에는 쓰이지만 특정 섹션만 보강하지 못한다.
3. **좌표 이동 여력을 확인할 것.** `l_anchor`는 초기 형상(`base_coords`) 기준으로 당기고
   `max_displacement = 50.0mm` clamp도 `base_coords` 기준 누적이다. Stage 4 진입 시점의
   누적 변위 최대값을 로그로 찍어 여유를 확인한다(idea_v4 §9-2 미검증 항목).

### §5.9 반환 시그니처 및 `__main__` 갱신 (필수, 개정3 신규)

> **초판의 결함**: §5.0~§5.8은 Stage 4를 `run_training_multi()` **내부**에 통합하는 방법만
> 서술했다. 그러나 v1_2의 `run_training_multi()`는 이미 9-튜플
> `model, history, base_coords_snapshot, final_coords, final_z_gate, seg_ids, split_epochs,
> pruning_state, final_epoch`를 반환하고, `__main__`이 이를 9개 변수로 언패킹한다
> (`AI_design_v1_2.py:1240-1241`, `:1274-1275`). Stage 4가 §2.4.1의 `death_log`(사망 원장,
> 리포트에 표로 덤프해야 함)와 Stage 4 진입 여부(`enter_stage4`)라는 **새 상태**를 만들어내는데,
> 이를 반환값에 포함시킬지, `__main__`의 언패킹 리스트를 몇 개로 바꿀지가 문서에 없었다.
> 그대로 구현하면 `ValueError: not enough values to unpack` 또는 리포트가 `death_log`에
> 접근할 방법이 없어 §2.4.1 요구("death_log는 리포트에 표로 덤프한다")를 충족하지 못한다.

**대상**: `run_training_multi()`의 함수 말미 `return` 문, `AI_design_v1_2.py` 최하단 `__main__` 블록.

- **AS-IS** (`run_training_multi()` 말미)

```python
return model, history, base_coords_snapshot, final_coords, final_z_gate, seg_ids, split_epochs, \
    pruning_state, final_epoch
```

- **TO-BE**

```python
return model, history, base_coords_snapshot, final_coords, final_z_gate, seg_ids, split_epochs, \
    pruning_state, final_epoch, death_log, enter_stage4    # [v1.3] §5.9 — death_log·Stage4 진입 여부 추가
```

`death_log`는 §2.4.1에서 학습 내내 append되는 리스트이며, `enter_stage4`는 §5.1에서 계산된
bool이다. 둘 다 루프 종료 시점에 이미 지역 변수로 존재하므로 새로 계산할 필요는 없다.
**`stage4_start_epoch`, `stage4_start`, `stage4_end`, `stage4_active`는 반환하지 않는다** —
이들은 Stage 4 진행 중에만 의미가 있는 루프 내부 상태이고, 호출부가 필요로 하는 것은 결과
(`death_log`)와 진입 여부(`enter_stage4`)뿐이다.

- **`__main__` AS-IS**

```python
model, history, base_coords, final_coords, final_z_gate, final_seg_ids, split_epochs, \
    pruning_state, final_epoch = run_training_multi(data, target_mps, max_epochs=args.epochs)
```

- **`__main__` TO-BE**

```python
model, history, base_coords, final_coords, final_z_gate, final_seg_ids, split_epochs, \
    pruning_state, final_epoch, death_log, enter_stage4 = \
    run_training_multi(data, target_mps, max_epochs=args.epochs)    # [v1.3] §5.9 — 11개로 확장
```

`report_design_comparison()` 호출부에 `death_log`를 인자로 추가 전달해 §2.4.1이 요구하는
사망 원장 표를 리포트에 렌더링한다(리포트 함수 시그니처 확장은 §2.4.1의 연장이며, 여기서는
호출부만 다룬다). 구현 후 검증:
`grep -n "return model, history" AI_design_v1_3.py`와
`grep -n "= run_training_multi(" AI_design_v1_3.py` 양쪽의 언패킹 변수 개수가
**정확히 11개로 일치**하는지 확인한다.

---

## §6. 단위 테스트 (학습 불필요)

### §6.1 `tests/test_thickness_range_v1_3.py`

```python
def test_full_range_reachable():
    """§3.2 — 모든 노드가 [0.71, 2.29] 전 구간에 도달 가능해야 한다. 핵심 요구사항."""
    data = build_bpillar_17section()
    lo, hi = assert_thickness_reachability(data.x)
    assert float(lo.max()) <= T_MIN_V13 + 0.01
    assert float(hi.min()) >= T_MAX_V13 - 0.01

def test_outer_hat_not_frozen():
    """§3.2 — t_init == T_MAX 인 Outer Hat이 특히 위험하다. 하한까지 내려갈 수 있어야 한다."""
    data = build_bpillar_17section()
    lo, hi = assert_thickness_reachability(data.x)
    outer = (data.x[:, 6] > 2.29)
    assert float(lo[outer].max()) <= 0.75

def test_reachability_fails_on_old_delta_scale():
    """가드가 실제로 동작하는지 — DELTA_SCALE=1.35 이면 반드시 실패해야 한다(회귀 방지)."""
    data = build_bpillar_17section()
    with pytest.raises(AssertionError):
        assert_thickness_reachability(data.x, delta_scale=1.35)

def test_frac_helper_is_single_source():
    """§3.3 — forward와 report가 동일 헬퍼(_t_init_logit)를 쓰는지. 1e-4 리터럴 잔존 금지."""
    src = open("AI_design_v1_3.py", encoding="utf-8").read()
    assert "1e-4, 1.0 - 1e-4" not in src
    assert src.count("_t_init_logit(") >= 3      # 정의 1 + 호출 2

def test_thickness_grad_is_alive():
    """1-epoch forward/backward. Outer Hat 두께 그래디언트가 NaN이 아니고 유의미해야 한다."""
    ...  # |grad| > 1e-6, isfinite
```

### §6.2 `tests/test_sparse_entropy_v1_3.py`

```python
def test_sparse_contribution_is_zero():
    """§2.1 — W_SPARSE_V13=0 이면 contrib_sparse == 0."""

def test_entropy_pushes_to_binary():
    """§2.3 — z=0.5에서 엔트로피 손실이 최대, z=0/1에서 최소여야 한다(대칭 이진화 압력)."""

def test_alpha_ent_is_injectable():
    """§2.3/§5.6 — train_step_multi가 alpha_ent를 인자로 받아 0으로 끌 수 있어야 한다."""
    import inspect
    assert 'alpha_ent' in inspect.signature(train_step_multi).parameters

def test_no_usv18_train_loop_used():
    """§2.3 — usv18.train_step/run_training/visualize_epoch_snapshots 미호출 확인."""
    src = open("AI_design_v1_3.py", encoding="utf-8").read()
    for name in ("usv18.train_step", "usv18.run_training", "usv18.visualize_epoch_snapshots"):
        assert name not in src

def test_target_area_is_95_percent():
    """§2.6.1 — target_area가 초기 단면적의 95%로 설정되어야 한다."""
    data = build_bpillar_17section()
    area_init, _ = usv18.compute_section_area(data.x[:, :2], data.x[:, 6:7], data.edge_index)
    assert abs(AREA_TARGET_RATIO * float(area_init) - 22854.1) < 1.0

def test_l_mass_is_in_loss():
    """§2.6.2 — l_mass가 loss에 편입되어야 예산 제약이 성립한다(v1_2에서는 누락되어 있었다)."""
    src = open("AI_design_v1_3.py", encoding="utf-8").read()
    loss_expr = src.split("loss = (")[1].split("\n\n")[0]
    assert "contrib_mass" in loss_expr

def test_l_mass_zero_below_target():
    """§2.6.3 — 목표 이하에서는 relu가 0이라 예산 항이 사라져야 한다(과도 감량 방지)."""
    area = torch.tensor(22000.0); target = torch.tensor(22854.1)
    assert float(torch.relu(area / target - 1.0) ** 2) == 0.0

def test_report_baseline_is_initial_area():
    """§2.6.4 — target_area를 낮춰도 리포트 baseline은 24,056.9(t_init 기준)이어야 한다."""
    ...  # area_init이 t_init에서 독립 재계산되는지 확인

# ── [개정2 신규] ────────────────────────────────────────────────
def test_area_weighted_sparse_requires_nonzero_weight():
    """§2.2 — 스위치 True + 가중치 0.0 조합은 침묵 no-op이므로 반드시 실패해야 한다.
    주의: 위의 test_sparse_contribution_is_zero는 이 고장난 설정도 통과시켜 은폐한다."""
    with pytest.raises(AssertionError):
        exec_guard(use_area_weighted=True, w_sparse=0.0)

def test_per_candidate_area_ignores_gate():
    """§2.2.1 — 가중치가 z_gate에 의존하면 자기강화 루프가 생긴다. t_raw만 써야 한다."""
    a1 = _per_candidate_area(coords, t_raw, ...)
    a2 = _per_candidate_area(coords, t_raw, ...)   # z_gate만 다르게 준 경우
    assert torch.allclose(a1, a2)

def test_dead_zone_boundary_is_2p4():
    """§2.4.1 — |log_alpha| > 2.4에서 z_gate가 clamp에 걸려 그래디언트가 0이어야 한다."""
    la = torch.tensor([2.5], requires_grad=True)
    z = torch.clamp(torch.sigmoid(la) * 1.2 - 0.1, 0.0, 1.0)
    z.backward()
    assert float(la.grad.abs()) < 1e-12

def test_death_ledger_classifies_suspect():
    """§2.4.1 — Mp 미달 상태의 사망은 SUSPECT, 충족 상태의 사망은 JUSTIFIED."""
    assert classify_death(mp=0.95) == "SUSPECT"
    assert classify_death(mp=0.99) == "JUSTIFIED"

def test_rollback_rules_are_section_agnostic():
    """§7.0 — 롤백 규칙 구현에 특정 섹션 번호가 하드코딩되어서는 안 된다(사용자 확정)."""
    src = open("AI_design_v1_3.py", encoding="utf-8").read()
    rb = src.split("# --- rollback rules ---")[1].split("# --- end rollback ---")[0]
    for tok in ("sec 10", "sec 11", "sec 12", "[10,", "[10, 11, 12]", "(10, 11, 12)"):
        assert tok not in rb
```

### §6.3 `tests/test_stage4_binarize_v1_3.py` — **[개정 신규]**

```python
def test_hard_exists_never_revives_deleted():
    """§5.2 결정① — DELETED 파트는 z_gate가 아무리 높아도 부활하지 않아야 한다."""
    ps = make_pruning_state_multi()
    ps['state'][5, 0] = STATE_DELETED
    zc = torch.full((NUM_SECTIONS, len(CANDIDATE_PARTS)), 0.9)   # 전부 임계 초과
    alive = (ps['state'] != STATE_DELETED)
    hard_exists = alive & (zc >= BINARIZE_THRESHOLD)
    assert not bool(hard_exists[5, 0])

def test_hard_exists_is_subset_no_merge():
    """§5.2 — hard_exists는 기존 존재맵의 부분집합 → run은 분할만, 병합 없음."""
    ...  # alive.sum() >= hard_exists.sum() 및 (hard_exists & ~alive).sum() == 0

def test_segment_ids_backward_compatible():
    """§5.3 — hard_exists=None 이면 기존 동작과 완전히 동일해야 한다."""
    ps = make_pruning_state_multi()
    ps['state'][5:8, 0] = STATE_DELETED
    assert torch.equal(compute_segment_ids(ps), compute_segment_ids(ps, hard_exists=None))

def test_segment_ids_cuts_on_hard_exists():
    """§5.3 — 0.3 미만으로 죽인 섹션이 실제 절단점이 되어야 한다(그룹 수 증가)."""
    ...  # hard_exists로 중간 섹션을 False 처리 → 고유 seg id 개수 증가 확인

def test_log_alpha_5_saturates_gate():
    """§5.2/§5.7 — |log_alpha|=5 이면 z_gate가 정확히 0.0/1.0 이어야 한다."""
    ...  # allclose(z_gate, 0.0 또는 1.0, atol=1e-6)

def test_stage4_start_epoch_is_after_curriculum():
    """§5.1 개정3 — stage4_start_epoch = last_epoch + 1 이며 커리큘럼 종료보다 앞설 수 없다.
    기존 리포트용 final_epoch(run_training_multi 루프의 `final_epoch = epoch`)와는 별개 변수이므로
    이 이름이 재사용되지 않았는지도 함께 확인한다(개명 사유는 §5.1 참조)."""
    src = open("AI_design_v1_3.py", encoding="utf-8").read()
    assert "stage4_start_epoch = last_epoch + 1" in src
    assert "stage4_start_epoch >= curriculum_max_epochs" in src
    assert "final_epoch  = last_epoch + 1" not in src and "final_epoch = last_epoch + 1" not in src

def test_binarize_comparator_consistency():
    """§4/§5.2 — 두 곳 모두 '>=' 여야 한다. 경계값 0.3에서 판정이 갈리면 §5.7 assert가 깨진다."""
    src = open("AI_design_v1_3.py", encoding="utf-8").read()
    assert "> BINARIZE_THRESHOLD" not in src
    assert src.count(">= BINARIZE_THRESHOLD") >= 2
```

### §6.4 dry-run 규칙

기존 v1.1/v1.2와 동일하게 dry-run은 **`model.eval()`** 로 수행한다. HardConcrete의 train 모드
난수가 균일 두께 assert를 깨뜨린다.

---

## §7. 단계별 판정 기준 (Go / No-Go)

| 단계 | 통과 조건 | 실패 시 조치 |
|---|---|---|
| **1 (§2)** | 최종 단면적이 **22,854.1 mm² 이하**(감량 목표 95%, §2.6). `log_alpha_sat` 학습 후반 상승. **전 섹션 `SUSPECT` 사망 0건**이고 R1~R3 미발동(§7.0) | 게이트 중간값 정체 → §2.2(B안) + `ALPHA_ENT_V13` 0.1로 상향. 면적이 목표 근처 정체 → §2.5에 따라 `f` 계수 2~5배 검토(10배 이상 금지). **면적·Mp 동시 불가 → §2.6.6 완화 순서** |
| **2 (§3)** | `assert_thickness_reachability` 통과. 최종 두께가 초기값 근처에 갇히지 않고 **파트별로 서로 다른 방향으로 움직임**. **전 섹션 `SUSPECT` 사망 0건**(§2.4.1) | 도달성 실패 → `DELTA_SCALE` 상향. 두께 진동/발산 → §3.5-2 순서로 대응하되 `DELTA_SCALE`을 1.35로 되돌리지 말 것. **`SUSPECT` 사망 발생 → §3.5-4-a 조치 사다리** |
| **3 (§4)** | 죽음의 구간 잔존 게이트 감소. 전 섹션 hard/target ≥ 98% | 게이트가 0/1로 수렴하지 않았으면 → **§5 자동 진입**. **Mp 미달 섹션의 Patch가 `DELETED`로 확정되었으면**(어느 섹션이든) → **§5로 해결 불가**, 1단계로 회귀(§5.8-1) |
| **4 (§5)** | §5.7 assert 통과. 전 섹션 hard/target ≥ 98%. **단면적 ≤ 22,854.1 mm²**. 두께 전 파트 [0.7, 2.3]. **죽음의 구간 잔존 게이트 0개**(정의상) | assert 실패 → tolerance 완화 금지. §4/§5.2 비교 연산자 불일치부터 확인 |

### §7.0 롤백 규칙 (개정2 — 전면 교체)

> **초판 규칙**: "1단계에서 50 epoch 이내에 Patch1 **평균** 게이트가 0.2 이하로 떨어지고 연속 파트
> 두께가 상한에 붙으면 즉시 중단."
>
> **폐기 사유**: (a) 시간 창이 50 epoch로 너무 짧다 — v1_2의 게이트 하강은 epoch 680→1499에 걸쳐
> 일어났다. (b) **평균**이라 국소 붕괴를 못 본다. (c) 어느 섹션인지는 애초에 알 필요가 없다
> (사용자 확정: 특정 섹션 하드코딩 금지).

아래 셋으로 교체한다. **전부 섹션 무관이며, 전 섹션 × 전 후보에 적용한다.**

| # | 트리거 | 근거 | 조치 |
|:-:|---|---|---|
| **R1** | `epoch ≥ ASWC_STAGE2_END` 이후 `SUSPECT` 사망(§2.4.1)이 **1건이라도** 발생 | 물리가 요구하는 파트를 비가역으로 잃음. 커리큘럼이 충분히 램프된 뒤이므로 "학습 초반이라 Mp가 나쁜 것"과 구분됨 | 즉시 중단 → `ALPHA_ENT_V13` **0.02**로 낮춰 재시도 |
| **R2** | 임의의 `DEATH_BURST_WINDOW`(100) epoch 창에서 신규 0-포화가 **당시 ALIVE의 20% 이상** | 연쇄 붕괴(Patch2 전멸 패턴). Mp 판정과 무관하게 **속도만으로** 잡는다 | 즉시 중단 → `W_MASS_V13` **5.0**으로 낮춰 재시도 |
| **R3** | 신규 0-포화가 누적되는 동안 `area`가 **`AREA_REGRESS_WINDOW`(200) epoch 전보다 증가** | **질량 역설(Req 4)의 정의 그 자체.** v1_2에서는 1500 epoch를 다 돌고 리포트를 봐야 알 수 있었다 | 즉시 중단 → `ALPHA_ENT_V13` **0.02**로 낮춰 재시도 |

**어느 규칙에서도 `W_SPARSE_V13`를 다시 올리는 것은 금지한다** — 그것이 원래 병인이다.
`T_MIN`을 되돌리는 것도 금지한다(전 파트 공통 제조 제약, §3.5-4-a).

```python
# R2 / R3 구현 (death_log와 area 이력만 있으면 된다)
if epoch >= ASWC_STAGE2_END and any(d["verdict"] == "SUSPECT" for d in death_log):
    raise RuntimeError(f"[R1] SUSPECT death detected — {death_log[-1]}")

recent = [d for d in death_log if d["epoch"] > epoch - DEATH_BURST_WINDOW]
if len(recent) >= DEATH_BURST_RATIO * int(alive.sum()):
    raise RuntimeError(f"[R2] 연쇄 붕괴: {DEATH_BURST_WINDOW}ep 내 {len(recent)}개 사망")

if death_log and area_hist.get(epoch - AREA_REGRESS_WINDOW) is not None \
   and float(area) > area_hist[epoch - AREA_REGRESS_WINDOW] \
   and any(d["epoch"] > epoch - AREA_REGRESS_WINDOW for d in death_log):
    raise RuntimeError(f"[R3] 질량 역설: 게이트가 죽는 중 area 증가 "
                       f"({area_hist[epoch - AREA_REGRESS_WINDOW]:,.0f} → {float(area):,.0f})")
```

### §7.2 v1_2 회귀 참조선 (판정 조건 아님)

sec 10·11·12는 v1_2에서 조용히 삭제되어 sec 11 Mp 95.0%를 유발한 **표본**이다. 통과/중단을 가르는
조건으로 쓰지 않되, 진단 편의를 위해 리포트에 다음 한 줄을 남긴다.

```
| v1_2 회귀 참조 | sec 10·11·12 Patch1: {생존/사망} — 판정 조건 아님, 진단 참고용 |
```

### §7.1 감량 목표 — 확정 (2026-08-08 사용자 결정)

**`AREA_TARGET_RATIO = 0.95`. 최종 단면적 목표 22,854.1 mm² (초기 24,056.9 대비 −5%).**

- idea_v4.md §8 체크리스트에 있었으나 본 지시서 초안에서 누락되었던 항목이다.
- 구현은 §2.6. **`target_area`를 낮추는 것만으로는 목표가 되지 않으며**(선형항 `f`의 정규화
  상수일 뿐), `l_mass`를 `loss`에 편입해야 실제 예산 제약이 된다 — §2.6의 핵심이다.
- 실현 가능성은 미확인이다. 완화가 필요하면 **§2.6.6의 순서**(w_mass ↓ → ratio 0.97 → 1.00)를
  따르고, **Mp 목표를 먼저 낮추는 것은 금지**한다.

---

## §8. 미검증 가정 (그대로 믿지 말 것)

idea_v4.md §9를 계승하며, 본 문서에서 추가·정정된 항목을 표시한다.

1. **"섹션 11의 게이트는 0.15였다"** — 근거 없음. 실측은 생존 섹션 평균 0.584뿐이다.
2. **"좌표가 `max_displacement`(50.0mm)에 걸려 그래디언트가 0이다"** — 측정해 봐야 안다(§5.8-3).
3. **§2만으로 게이트가 자연 수렴한다** — 논리적으로 타당하나 실행으로 확인되지 않았다. 1단계 관찰 대상.
4. **`DELTA_SCALE = 10.0`에서 학습이 안정적이라는 것** — §3.2 표의 도달 범위는 **정적 상한 실측**이며
   확정 사실이나, `thickness_gate` 승수와 그룹 평균 pooling을 반영하지 않았다(§3.2 유효 조건).
   스케일 7.4배 상향이 옵티마이저 동역학에 미치는 영향(§3.5-2)은 **재학습으로만 알 수 있다.**
5. **[정정] `l_mass`가 질량 목적함수라는 전제** — **거짓으로 판명.** `l_mass`는 `loss`에 더해지지
   않는다. 실제 질량 항은 `compute_alda_loss`의 `f = area/target_area`이다(§2.1).
   제안 A의 결론은 유효하나 근거 문장이 교체되었다. **§2.6에서 `l_mass`를 감량 목표의
   예산 제약으로 loss에 편입한다** — v1.3의 의도적 신규 변경이며, v1_2 대비 손실 항이 하나 늘어난다.
6. **[신규] 95% 감량과 전 섹션 Mp 달성이 양립 가능한지** — **미확인.** v1_2는 면적 +10.1%
   상태에서도 sec 11이 95.0% 미달이었다. 다만 그 +10.1%는 `l_sparse`가 만든 인공적 결과이므로
   §2·§3 적용 후의 진짜 면적 하한은 아직 모른다. **1단계 재학습이 이 질문의 답이며, 목표
   미달 시 §2.6.6의 완화 순서를 따른다.**
7. **[신규] ALDA dual ascent가 동작한다는 전제** — **거짓.** `mu_Mp`/`rho_Mp` 갱신 코드는
   `usv18.run_training()` 안에만 있고 호출되지 않는다. 고정계수 penalty로 동작한다(§2.5).
   `f` 계수와 Mp 항의 균형이 적절한지는 1단계 재학습의 관찰 대상이다.
8. **[신규] Stage 4의 좌표 자유도가 Mp 미달 섹션을 구제하기에 충분한지** — 연속 부재 두께 분할이
   금지되어(§5.0) 국소 보강 수단이 "패치 + 좌표"뿐이다. 패치가 살아 있다는 전제 하에서만
   §7 4단계 통과가 가능하다. **실행으로만 확인된다.**
9. **[삭제] 연속 파트 run pooling의 절단 규칙** — 제조 제약상 **금지**로 확정되어 미결 항목에서 제외.
10. **[개정2 신규] `ALPHA_ENT = 0.05`에서 데드존이 문제가 되지 않는다는 것** — §2.4.1의 정량 비교는
    엔트로피 압력이 `z≈0.3`에서 물리 신호의 1/7임을 보이지만, 이는 **정적 그래디언트 비교**이며
    `gate_mult`·옵티마이저 모멘텀·`term_Mp`와의 상호작용은 반영하지 않았다. §7 1단계 실패 시
    `ALPHA_ENT = 0.1`로 올리면 `z≈0.05`에서 물리를 역전하므로, 상향 시 §2.4.1 원장을 함께 볼 것.
11. **[개정2 신규] 롤백 규칙 R1의 임계(`DEATH_MP_THRESHOLD = 0.98`)가 적절한지** — `ASWC_STAGE2_END`
    직후에는 Mp가 아직 수렴 중이라 `SUSPECT` 오탐이 날 수 있다. 1단계에서 오탐이 잦으면
    **임계를 낮추지 말고 판정 시작 epoch를 뒤로 미룰 것**(오탐을 줄이려 0.95로 낮추면 v1_2의
    sec 11(95.0%)이 정상으로 분류되어 감시망이 무의미해진다).

---

<details>
<summary>숙의 과정</summary>

### 초판: Synod 세션 `synod-20260808-idea-cgnn13`

**모델 기여**
- **Gemini (Architect→Critic→Defense)**: `T_MAX=2.3` + `t_init=2.300` 조합의 **logit 포화 동결**을
  최초 지적(§3.2) — idea_v4.md에 없던 신규 발견. Stage 4 옵티마이저 재빌드 필요성(모멘텀 쇼크)도 제시(§5.5).
- **OpenAI (Explorer→Critic→Prosecution)**: 변경마다 단위 테스트를 요구(§6), `z_gate`가 temperature를
  쓰지 않는다는 점을 재확인. "백오프가 initial-vs-final 비교의 기준선을 오염시킨다"는 공격 덕분에
  `mp_init`/`area_init`이 `t_init`을 직접 쓴다는 사실을 확인해 §3.2 각주로 확정.
- **Claude (Validator/Judge)**: 도달 가능 두께 범위를 수치로 실측, `frac` 계산식이 **두 곳**에
  중복 존재한다는 함정 발견(§3.3), `ALPHA_ENT` monkey-patch 불필요를 코드로 반증(§2.3).

**사용자 지시에 의한 정정 (2026-08-08)**: 숙의 결과는 `FRAC_CLAMP_HI = 0.95`로 Outer Hat 동결만
해소하는 안이었으나, 사용자가 "최종 두께가 `[0.7, 2.3]` 범위에서 결정되기를 원한다"고 요구를
명확히 하여 진단을 확장했다. 동결은 증상의 일부일 뿐이고 **`DELTA_SCALE = 1.35`가 모든 파트의
도달 범위를 초기값 주변 ±0.9mm로 묶고 있는 것**이 본질이었다. §3은 이에 따라 전면 개정되었다.

### 개정: Synod 세션 `synod-20260808-review-cmdv13` (본 문서 리뷰)

**적발된 결함**
1. **[치명] §1 상수 배치 → import 시 `NameError`.** `CGDN17`(L135)이 L907의 상수를 클래스 속성으로
   참조. Gemini·OpenAI 양측이 독립적으로 최우선 지적. → §1에서 배치 규칙 수정.
2. **[치명] §5 옵티마이저 재빌드가 존재하지 않는 변수 참조.** `coord_params`/`thickness_params`는
   코드에 없으며 실제는 `main_params`/`thick_params`/`gate_params`. → §5.5에서 수정.
3. **[중대] §2.1의 근거가 코드와 불일치.** `l_mass`가 `loss`에 더해지지 않음을 Claude가 소스에서
   확인. 결론은 유효하나 근거를 `L_alda`의 `f`로 교체. → §2.1, §8-5.
4. **[신규] ALDA dual ascent 미동작.** `mu_Mp`/`rho_Mp` 갱신이 `usv18.run_training()`에만 존재.
   OpenAI가 Prosecution에서 "μ가 10⁵까지 폭주해 면적이 무시된다"고 공격했으나, 이 코드 경로에는
   갱신 자체가 없어 **기각**. 동시에 Gemini의 "제약 만족 시 relu 소멸" 방어가 성립함이 확인됨. → §2.5.
5. **[정정] `l_sat` vs `DELTA_SCALE` 충돌은 무해.** 양 모델이 정량 계산 후 합의(기여 0.01,
   `contrib_phys` 대비 1/1000). 초안 §3.5-1의 우려를 하향 조정.

**해결된 주요 쟁점**
1. "W_SPARSE=0이면 질량 감소 압력이 소멸한다" → **철회**(양 모델). `L_alda`의 `f`가 상시 선형
   하강 압력을 공급하며 `∂f/∂z_k = A_k/target_area`.
2. "μ_Mp 폭주로 면적이 무시된다" → **기각**. dual ascent 코드가 실행 경로에 없음.
3. 연속 부재 두께 pooling 완화(idea_v4 §6.1의 3개 모델 합의) → **제조 제약상 채택 불가**(사용자 확정).
   초안 §5.5 삭제, 금지 조항으로 전환. sec 11 구제 책임이 §2·§3으로 이관됨(§5.8-1).

**사용자 결정 (2026-08-08, 리뷰 이후)**
- ① 동적 분할은 Patch1/Patch2에만. 연속 부재는 절단 금지.
- ② 강제 이진화 시 `DELETED`는 부활시키지 않음 → `(state != DELETED) AND (z_gate >= 0.3)`.
- ③ "학습 초기화"는 옵티마이저 + 스케줄 재시작(모델 가중치 유지). 단 커리큘럼은 최종값 고정(§5.6).
- ④ Stage 4에서 파트 존재는 동결, **연속 부재 + 패치의 좌표 및 두께**를 [0.7, 2.3] 내에서 조정.

**신뢰 점수 (T = min(C×R×I/S, 2.0))**
- Claude: **2.00** (소스 직접 확인 기반. `l_mass`·ALDA·변수명 결함을 코드로 확정)
- Gemini: **1.41** (Solver에서 확신도 98로 오류 주장했으나 Critic에서 모범적으로 철회.
  Defense의 dual-ascent 주기 서술에 오기)
- OpenAI: **1.03** (쟁점 포착은 유효. confidence evidence에 존재하지 않는 "A/B 로그" 날조 — 감점)

**폴백 발생**: 없음. `gemini-3 flash --thinking high` / `openai-cli o3` 전 라운드 정상 응답.
세션 기록: `.omc/synod/synod-20260808-review-cmdv13/`

### 개정2: Synod 세션 `synod-20260808-review-ideav4-vs-cmdv13` (idea_v4 ↔ 본 문서 반영도 리뷰)

**적발된 결함**
1. **[치명] §5.1 `final_epoch` 미정의.** `NameError`뿐 아니라, 구현자가 `curriculum_max_epochs`로
   추측하면 extension 구간을 되감아 스케줄이 조용히 과거 값을 반환한다(§5.6이 경고한 사고).
   OpenAI 단독 포착. → §5.1에 `last_epoch` 기록 방식 + assert 2건.
2. **[중대] idea_v4 §8 체크리스트 2번(포화 구간 그래디언트 소멸)이 통째로 누락.** §2.4는 포화
   *비율*만 print하고, §2.3은 `ALPHA_ENT`를 올려 오히려 포화를 가속한다. Gemini·OpenAI가 Critic
   라운드에서 독립적으로 지적. → §2.4.1 신설.
3. **[정정] 데드존의 심각도는 하향.** Prosecution은 "제안 A와 C가 상호 파괴한다"며 치명으로 몰았으나
   **근거로 제시한 실험 로그가 날조**여서 기각. Judge가 직접 계산한 결과 α=0.05의 엔트로피 압력은
   `z≈0.3`에서 물리 신호의 1/7이라 설계는 안전하며, 문제는 **관측 수단의 부재**로 재분류.
   단 α=0.1에서는 `z≈0.05` 부근에서 역전하므로 §8-10에 미검증 가정으로 등록.
4. **[중대] §2.2 B안의 침묵 no-op.** 스위치 True + `W_SPARSE_V13=0.0` 조합에 가드가 없고,
   §6.2의 `test_sparse_contribution_is_zero`가 그 고장난 설정을 **통과시켜 은폐**한다.
   → §1 가드 assert + §6.2 대칭 테스트.
5. **[기각] `thick_params` / `'thickness_decoder'` 혼용이 오류라는 주장** — 후자는 param group의
   **라벨 문자열**이며 변수명이 아니다. Gemini의 Solver 주장을 OpenAI Critic이 반증.
6. **[기각] `alpha_ent` 인자 추가가 기존 호출부를 깬다는 주장** — 기본값이 있어 `TypeError` 없음.

**사용자 결정 (2026-08-08, 리뷰 이후)**
- ① 게이트 사망 감시는 **전 섹션 × 전 후보**에 적용한다. sec 10·11·12는 v1_2의 우연한 표본이므로
  판정 조건에 하드코딩하지 않는다 → §2.4.1 사망 원장, §7.0 R1~R3, §7.2 회귀 참조선.
- ② `final_epoch`은 **루프 탈출 시점의 epoch** 기준 → §5.1.
- ③ `T_MIN`/`T_MAX`는 **전 파트 공통 적용**(생산성 고려). 파트별 분리 하한 기각 →
  `T_MIN` 재검토는 진단 전용이 되고 조치는 질량 예산 사다리로 이관 → §3.5-4-a.

**신뢰 점수 (T = min(C×R×I/S, 2.0))**
- Claude: **2.00** (두 문서 전문 대조로 반영 매트릭스 작성, 엔트로피 vs 물리 그래디언트 정량 비교로
  자신의 ERROR 판정을 WARNING으로 하향)
- Gemini: **1.30** (Critic의 `T_MIN`→패치 포기 연쇄 분석이 최고 기여. 단 Solver 오독 2건,
  Defense에서 문서에 없는 "자동 체크포인트 롤백"을 방어 근거로 날조)
- OpenAI: **0.95** (`final_epoch`·scope-creep 단독 포착. 그러나 Prosecution에서 존재하지 않는
  실험 로그 날조 — 이전 세션과 동일 실패 패턴이므로 감점 유지)

**폴백 발생**: 없음. 3라운드 전부 정상 응답.
세션 기록: `.omc/synod/synod-20260808-review-ideav4-vs-cmdv13/`

</details>
