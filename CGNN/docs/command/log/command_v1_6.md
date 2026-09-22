# command_v1_6.md — AI_design_v1_5.py → AI_design_v1_6.py: 구조적 붕괴(자기교차) 수정 지시서

작성일: 2026-08-09 (`/synod design` 세션, Gemini flash conf 98 + OpenAI o3 conf 77)
참조: `docs/idea/idea_v1_6.md`(설계 근거, 신뢰도 84%), `docs/review/review_v1_5.md`(실행 결과 리뷰),
`AI_design_v1_5.py`(기반 코드), `uni_section_v18.py`(**불변 — import만**)

> **앵커 규칙**: 본 문서의 위치 지정은 **함수/변수 이름 기준**이다. 라인 번호는 참고용이며
> 코드 변경으로 어긋날 수 있다.

> **불변 규칙**: `uni_section_v18.py`는 한 줄도 수정하지 않는다. §4(중립 초기화+지연
> 엔트로피)와 §7(R1~R3 롤백)의 상수·로직도 절대 수정하지 않는다 — idea_v1_6.md와 이번 design
> 세션 양쪽에서 재확인된 제약이다. `continuity_weight_schedule()`과 Stage4 진입 로직은 이
> 불변 목록에 속하지 않는다(review_v1_5.md 4번 권고 참고).

> **정정 이력**: design 세션에서 OpenAI가 §3(파트 간 충돌 배리어)에 대해 "기존
> `collision_spec` 순회 구조는 인접 그리드 셀 전제라 임의 파트 간 교차를 탐지 못하며 O(N²)
> 완전 재설계가 필요하다"고 주장했으나, Claude가 `uni_section_v18.build_collision_spec()`
> 코드를 직접 확인한 결과 **이미 섹션 내 모든 unordered 파트 쌍(Outer-Inner 포함)에 대해
> Huber 상한이 적용된 관통 페널티를 계산 중**임이 확인됐다(OpenAI 주장은 사실과 다름 —
> 근거 없이 기각). 실제 문제는 `_signed_projections()`의 `valid_mask`가 좌표 투영이 세그먼트
> 범위 [0,1]을 벗어나면 해당 쌍의 페널티를 **조용히 무효화**한다는 데 있을 가능성이 높다 —
> §3은 새 메커니즘이 아니라 기존 메커니즘의 이 공백을 메우는 확장으로 재정의됐다.

---

## §0. 목표 및 산출물

`AI_design_v1_5.py`를 복사해 **`AI_design_v1_6.py`** 를 만들고, `review_v1_5.md`에서 확정된
구조적 붕괴 원인(연속성 가중치 무하한 감쇠 + Stage4 급격한 이진화 충격)을 고친다.

| 결함 | 근본 원인 | 대응 |
|---|---|---|
| **Outer/Inner 파트 자기교차** | `continuity_weight_schedule()`가 epoch 400 무렵 0.05까지 무하한 감쇠 → 형상 정합성 규제 사실상 소멸 | §1(폐루프 적응형 하한) |
| **Stage4 진입 시 좌표 급변** | candidate 게이트가 z≈0.5에서 300+ epoch 방치되다 epoch 500에 ±5.0으로 한 번에 강제 이진화 | §2(사전 게이트 첨예화) |
| (2차 방어선) 파트 간 관통이 좌표 이동 후 방치될 위험 | `_signed_projections`의 valid_mask가 투영 범위 이탈 시 페널티를 무효화 | §3(폴백 페널티, P1 — §1·§2로 해결 안 될 시 적용) |

**적용 순서**: §1(연속성 하한)과 §2(게이트 첨예화)를 함께 적용해 재실행하고 결과를 먼저
확인한다. §3은 P1(후순위)이며, §1·§2 적용 후에도 자기교차가 재발할 때만 추가한다.

---

## §1. `w_continuity` 폐루프 적응형 하한

### §1.1 상태 저장소 추가

**대상**: `run_training_multi()` 내부, epoch 루프 시작 직전(기존 `alpha_ent_start` 계산 근처).

```python
# [v1.6 §1] 연속성 가중치 폐루프 제어 상태 — 매 epoch 갱신되므로 클로저가 아닌 mutable dict로
# run_training_multi 스코프에 존재해야 한다(Critic 세션: OpenAI가 지적한 nonlocal 누락 위험을
# dict로 원천 차단 — 클로저 변수 재할당 대신 dict 키 갱신이므로 nonlocal 선언 불필요).
continuity_state = {"w_adaptive": 1.0, "last_l_continuity": 0.0}
```

### §1.2 신규 상수 블록

기존 v1.5 상수 블록 뒤에 추가한다.

```python
# ══════════════════════════════════════════════════════════════════
# [v1.6] 로컬 오버라이드 상수 — 연속성 폐루프 제어 + 게이트 사전 첨예화
# 근거: docs/idea/idea_v1_6.md, docs/review/review_v1_5.md, docs/command/command_v1_6.md
# ══════════════════════════════════════════════════════════════════

# §1. 연속성 가중치 폐루프 제어 — 기존 continuity_weight_schedule()의 w_min=0.05 무하한 감쇠가
# 구조적 붕괴의 1차 원인(review_v1_5.md)이었다. 완전 대체가 아니라 기존 sigmoid를 하한선으로
# 두고, 실측 위반이 클 때만 폐루프가 이를 웃도는 값을 강제하는 방식으로 설계한다(design 세션
# Judge 판정: "물리적 유효성은 협상 불가능한 하드 제약"이라는 idea_v1_6.md의 결론을 실행
# 메커니즘 층위에서 구현).
CONTINUITY_W_FLOOR = 0.3    # 절대 하한 — 폐루프가 실패해도 이 값 밑으로는 못 내려간다
CONTINUITY_TAU      = 0.1   # 허용 위반 임계(review_v1_5.md 관측치: l_continuity가 0.8까지
                             # 커졌던 사례 기준, 이보다 훨씬 낮은 값에서 미리 개입)
CONTINUITY_ETA       = 2.0  # 폐루프 이득(design 세션: OpenAI 원안 eta=10.0은 1-epoch 지연과
                             # 결합 시 상한/하한 사이를 진동(limit-cycle)할 위험이 있어 하향
                             # 조정 — 진동 시 재조정 필요, §1.5 참고)
CONTINUITY_EMA_BETA  = 0.9  # last_l_continuity에 적용할 EMA 평활 계수(1-epoch 지연으로 인한
                             # 진동을 완화하기 위해 원시값 대신 지수이동평균을 사용 — design
                             # 세션에서 OpenAI가 지적한 "PID lag" 위험에 대한 대응)
```

### §1.3 스케줄 함수 교체

**대상**: `AI_design_v1_5.py`의 `continuity_weight_schedule()`(L803 부근).

```python
# 변경 전
def continuity_weight_schedule(epoch, stage2_end, w_max=1.0, w_min=0.05, beta=0.1):
    """학습 초반 크게(형태 붕괴 방지) -> 후반 sigmoid decay(로컬 Mp 만족 우선)."""
    return w_min + (w_max - w_min) / (1.0 + math.exp(beta * (epoch - stage2_end)))

# 변경 후 — 기존 함수는 이름을 유지한 채 "기저값(baseline)" 계산 전용으로 남기고, 신규 함수를
# 별도로 추가한다(호출부 100% 하위 호환 — 기존 시그니처를 쓰는 코드가 있다면 그대로 동작).
def continuity_weight_schedule(epoch, stage2_end, w_max=1.0, w_min=0.05, beta=0.1):
    """[변경 없음] sigmoid 기저값 — get_adaptive_continuity_weight()의 폐루프 하한 아래 깔리는
    베이스라인으로만 쓰인다. 이 함수 자체는 v1.5와 동일하다."""
    return w_min + (w_max - w_min) / (1.0 + math.exp(beta * (epoch - stage2_end)))


def get_adaptive_continuity_weight(epoch, stage2_end, continuity_state,
                                    w_max=1.0, w_min=0.05, beta=0.1,
                                    w_floor=CONTINUITY_W_FLOOR, tau=CONTINUITY_TAU,
                                    eta=CONTINUITY_ETA):
    """[v1.6 §1] 폐루프 적응형 연속성 가중치.
    w_base: 기존 v1.5 sigmoid(변경 없음, 하한선 역할).
    w_adaptive: 직전 epoch의 (EMA 평활된) l_continuity가 tau를 넘으면 가중치를 끌어올리고,
    tau 밑이면 서서히 낮춘다 — 단 w_floor 밑으로는 절대 못 내려간다(design 세션 Judge 판정:
    "물리적 유효성은 협상 불가능"이므로 하드 플로어는 controller 실패와 무관하게 항상 유효).
    최종값은 max(w_base, w_adaptive) — 둘 중 더 보수적인(높은) 쪽을 취한다(§1.5 D1 참고:
    OpenAI 지적 — 이 max 없이 단순 대체하면 초반 구간에서 오히려 가중치가 낮아지는 역효과 발생)."""
    w_base = continuity_weight_schedule(epoch, stage2_end, w_max, w_min, beta)
    if epoch > 0:
        error = continuity_state["last_l_continuity"] - tau
        continuity_state["w_adaptive"] = max(
            w_floor, min(w_max, continuity_state["w_adaptive"] + eta * error))
    else:
        continuity_state["w_adaptive"] = w_max
    return max(w_base, continuity_state["w_adaptive"])
```

### §1.4 호출부 수정

**대상**: `train_step_multi()` 내부 `w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)`
(L1307 부근)와 `run_training_multi()`의 호출 지점.

`train_step_multi()`는 이미 `w_col_aux_val` 등 스케줄 값을 인자로 받는 패턴이 있으므로 동일하게
`w_continuity_val` 인자를 추가한다(design 세션: OpenAI가 우려한 "시그니처 비대화"는, v1.5에서
이미 확립된 패턴을 그대로 따르는 것이 새 전역 상태보다 낫다는 것이 Judge 판정 — 기존 관례 준수).

```python
# train_step_multi() 시그니처에 추가
def train_step_multi(model, data, optimizer, target_mps, target_area,
                      epoch, max_epochs, weights, curriculum,
                      curriculum_ratio, collision_spec, alda_state,
                      pruning_state=None, segment_ids=None,
                      alpha_ent=ALPHA_ENT_V13, stage4_active=False, frozen_curriculum=None,
                      w_col_aux_val=None, w_continuity_val=None):        # [v1.6 §1]
    ...
    # 기존: w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)
    # 변경 후: 호출부에서 전달받은 값 사용, 미전달 시(예: Stage4) 기존 sigmoid로 폴백
    if w_continuity_val is None:
        w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)
    else:
        w_continuity = w_continuity_val
```

`run_training_multi()` 메인 루프(§3.3 위치, L1405 부근)에서:

```python
curr_w_continuity = get_adaptive_continuity_weight(epoch, ASWC_STAGE2_END, continuity_state)
info = train_step_multi(model, data, optimizer, target_mps, target_area,
                         epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                         collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                         alpha_ent=alpha_ent_schedule(epoch), stage4_active=False,
                         w_col_aux_val=curr_w_col_aux,
                         w_continuity_val=curr_w_continuity)             # [v1.6 §1]
# epoch 종료 후 상태 갱신(EMA 평활 — §1.5 A2 참고)
continuity_state["last_l_continuity"] = (
    CONTINUITY_EMA_BETA * continuity_state["last_l_continuity"]
    + (1 - CONTINUITY_EMA_BETA) * float(info["l_continuity"]))
```

Stage4 호출부(L1586 부근)는 **수정하지 않는다** — `w_continuity_val` 미전달 시 기존 sigmoid로
폴백하며, Stage4는 이미 continuity가 안정된 시점(epoch 500 이후)에 시작하므로 무해하다
(v1.5의 `w_col_aux_val` sentinel 패턴과 동일한 논리, review_v1_5.md에서 이미 검증된 설계).

### §1.5 design 세션에서 검토·기각된 대안

- **OpenAI 원안 eta=10.0**: 1-epoch 피드백 지연과 결합 시 가중치가 floor와 상한 사이를
  진동(limit-cycle)할 위험이 있다는 자체 지적을 받아들여 eta=2.0으로 하향, EMA 평활
  (β=0.9)을 추가로 도입해 지연 문제를 완화했다.
- **OpenAI가 제기한 DDP(분산 학습)/체크포인트 재개 시나리오**: 이 코드베이스에는 분산 학습
  또는 체크포인트 재개 메커니즘이 존재하지 않는다(design 세션에서 코드 확인 — 단일 프로세스
  스크립트, `torch.save`는 학습 종료 후 최종 가중치 저장 1회뿐). 해당 우려는 이번 세션의
  스코프 밖으로 판정해 기각한다.
- **D1(OpenAI 지적, 채택)**: `w_floor`만 단순 적용하면 초반 구간(sigmoid가 이미 1.0에 가까운
  구간)에서 오히려 값을 낮출 위험이 있어, 최종식을 `max(w_base, w_adaptive)`로 확정했다.

---

## §2. Stage4 사전 게이트 첨예화

### §2.1 신규 상수 블록 (§1.2에 이어 추가)

```python
# §2. 게이트 사전 첨예화 — candidate 게이트(z_gate)가 z≈0.5에서 300+ epoch 방치되다 Stage4
# (epoch 500)에서 한 번에 ±5.0으로 강제 이진화되는 충격을 완화한다. design 세션에서 OpenAI가
# 발견한 핵심 제약: ALPHA_ENT_START_EPOCH(§4, 불변)가 ASWC_STAGE2_END=400과 동일하므로,
# 첨예화가 epoch 400까지 진행되면 "게이트를 거의 이진화 vs 엔트로피가 저-엔트로피(이진화된
# 상태)를 즉시 벌점"이라는 정반대 압력이 같은 epoch에 충돌해 그래디언트 부호가 뒤집히는
# ping-pong 불안정을 유발할 수 있다(design 세션 Critic 판정, 근거: 코드 확인). 따라서 첨예화는
# 반드시 entropy 활성화(epoch 400) **이전에 완료**하고 그 이후로는 beta_steep을 고정한다.
GATE_STEEP_START = 200      # 첨예화 시작 epoch(idea_v1_6.md 원안 유지)
GATE_STEEP_END   = 350      # 종료 시점 — 원안(400)에서 앞당김. entropy 시작(400)과 최소
                             # 50epoch 여유를 두어 두 메커니즘의 직접 충돌을 피한다
GATE_STEEP_MAX   = 3.0      # 최대 배율(idea_v1_6.md 원안 5.0에서 하향 — OpenAI가 지적한 fp
                             # 정밀도 손실 위험 구간(log_alpha*beta가 ±10 근접)을 피하기 위해
                             # 보수적으로 설정. 재보정 시 |log_alpha_candidates|*GATE_STEEP_MAX
                             # 의 최대값이 ±8을 넘지 않는지 확인할 것
```

### §2.2 `CGDN17.compute_gates_multi()` 수정

**대상**: `AI_design_v1_5.py`의 `CGDN17.compute_gates_multi()`(L327 부근).

```python
# 변경 전
def compute_gates_multi(self, training, temperature):
    """원본 compute_gates()(line 188)는 1D 텐서만 받으므로, (17,len(CANDIDATE_PARTS))를
    flatten(-1)해서 그대로 통과시킨 뒤 동일 shape으로 unflatten한다..."""
    flat_log_alpha = self.log_alpha_candidates.reshape(-1)
    z_gate_flat, z_open_flat = usv18.compute_gates(
        flat_log_alpha, training=training, temperature=temperature, s_protect=None
    )
    ...

# 변경 후 — epoch을 모델 속성으로 받아 첨예화 배율을 log_alpha에 곱한 뒤 기존 immutable
# usv18.compute_gates()를 그대로 호출한다(usv18.py는 한 글자도 수정하지 않는다).
def compute_gates_multi(self, training, temperature):
    """[v1.6 §2] epoch 200~350 구간에 log_alpha_candidates를 GATE_STEEP_MAX까지 선형
    증폭해 시그모이드를 미리 첨예화한다 — 실제 파라미터 값은 건드리지 않고(forward 시점의
    스케일링만), Stage4 진입 시점(epoch 500)에는 이미 게이트가 0/1 근처에 있어 강제 이진화가
    작은 보정에 그치도록 만든다. self.current_epoch은 run_training_multi가 매 epoch 설정한다
    (§2.3)."""
    epoch = getattr(self, 'current_epoch', 0)
    if epoch < GATE_STEEP_START:
        beta_steep = 1.0
    elif epoch < GATE_STEEP_END:
        progress = (epoch - GATE_STEEP_START) / (GATE_STEEP_END - GATE_STEEP_START)
        beta_steep = 1.0 + progress * (GATE_STEEP_MAX - 1.0)
    else:
        beta_steep = GATE_STEEP_MAX   # epoch 350 이후 고정 — §4 entropy와의 충돌 방지(§2.1)

    flat_log_alpha = (self.log_alpha_candidates * beta_steep).reshape(-1)   # [v1.6 §2]
    z_gate_flat, z_open_flat = usv18.compute_gates(
        flat_log_alpha, training=training, temperature=temperature, s_protect=None
    )
    z_gate_cand = z_gate_flat.view_as(self.log_alpha_candidates)
    z_open_cand = z_open_flat.view_as(self.log_alpha_candidates)
    # (이하 원본과 동일)
```

**주의**: `flat_log_alpha`는 이제 forward 전달용 스케일된 값이다. R4 진단(`model.log_alpha_candidates
< 0.0`, L1492 부근)과 death ledger(`la = model.log_alpha_candidates.detach().cpu()`, L1501
부근)는 **원본 파라미터**(`self.log_alpha_candidates`, 스케일 전)를 직접 참조하므로 영향받지
않는다 — 부호는 배율에 무관하게 보존된다(`beta_steep > 0`이므로 sign(x*beta) == sign(x)).

### §2.3 `run_training_multi()`에 epoch 주입

**대상**: 메인 epoch 루프 최상단(§1.1 `continuity_state` 초기화 근처, 루프 진입 직후).

```python
while epoch < loop_max:
    model.current_epoch = epoch    # [v1.6 §2] compute_gates_multi()의 첨예화 스케줄용
    ...
```

Stage4 진입 시점(`model(...)` 호출, L1633 부근, `training=False`)에서도 `model.current_epoch`은
이미 GATE_STEEP_END(350)를 넘긴 값으로 설정돼 있으므로 `beta_steep=GATE_STEEP_MAX`로
자동 고정된다 — 별도 처리 불필요.

### §2.4 design 세션에서 검토·조정된 사항

- **beta_max 원안 5.0 → 3.0으로 하향**: OpenAI가 `log_alpha * beta`가 ±10 근접 시 fp16/fp32
  정밀도 손실 및 시그모이드 그래디언트 소실 위험을 지적했다. `INIT_LOG_ALPHA_V14=0.0`(§4 불변,
  중립 초기화)에서 시작해 학습이 진행되며 `log_alpha_candidates`가 통상 ±2.4(R4 sat 기준선)
  근방까지 이동하는 것으로 관측되므로(기존 로그 `log_alpha_sat` 지표 기준), beta=3.0이면
  최대 스케일된 값이 약 ±7.2로 안전 마진을 확보한다. 실행 후 `log_alpha_sat` 로그를 참고해
  beta_max 재조정 가능.
- **첨예화 종료 시점 400 → 350으로 단축**: §2.1에서 설명한 entropy 스케줄 충돌 회피가 근거.
- **Straight-Through Estimator(STE) 적용 여부**: idea_v1_6.md와 design 세션 모두 그래디언트
  소실 위험을 지적했으나, beta_max를 3.0으로 낮춘 시점에서 시그모이드가 완전히 포화하지
  않으므로(σ(±7.2)≈0.9993/0.0007, 그래디언트가 0은 아님) 이번 버전에서는 STE를 도입하지
  않는다 — **§7 검증 계획에서 그래디언트 소실 여부를 실측으로 확인하고, 문제가 확인되면
  후속 버전에서 STE를 추가**하는 것으로 범위를 좁혔다(과설계 방지).

---

## §3. (P1, 후순위) 파트 간 관통 페널티의 valid_mask 공백 보강

**적용 조건**: §1·§2를 먼저 적용해 재실행한 뒤에도 Outer/Inner 자기교차가 재발할 때만 착수한다.

### §3.1 배경 — 정정된 문제 정의

design 세션에서 `uni_section_v18.build_collision_spec()`(L562)과
`_signed_projections()`(L538)를 직접 확인한 결과, **파트 간(Outer-Inner 포함) 관통 페널티는
이미 존재하며 v1.5의 Huber 상한도 이미 적용되어 있다**(`compute_collision_penalty_unclamped()`,
`AI_design_v1_5.py` L820). 따라서 완전히 새로운 메커니즘은 불필요하다.

실제 공백은 `_signed_projections()`가 반환하는 `valid_mask`(`t_proj`가 세그먼트 매개변수
[0,1] 범위 밖이면 `False`)에 있다 — 좌표가 학습 중 크게 이동해 투영점이 세그먼트 범위를
벗어나면, 해당 점-세그먼트 쌍은 **관통 여부와 무관하게 페널티 계산에서 조용히 제외**된다.
자기교차가 진행될수록(정점들이 원래 배치에서 멀어질수록) 이 조건에 해당하는 쌍이 늘어나
"교차가 심해질수록 억제력이 약해지는" 역설적 상황이 발생할 수 있다.

### §3.2 제안 보강 (구현 시 상세 설계 필요 — 본 문서는 방향만 제시)

`compute_collision_penalty_unclamped()`(`AI_design_v1_5.py` L820) 내부에서, 방향(`d`)별로
`valid.sum() == 0`이면 `continue`하는 현재 로직(L791 부근)을 다음과 같이 보강한다:

```python
# 방향: valid_mask가 전부 False인 쌍에 대해, 세그먼트 끝점으로 클램프한 최근접 거리 기반
# 폴백 페널티를 추가한다(투영 범위 이탈 = 관통 여부 불명이 아니라 "다른 방식으로 확인 필요"
# 로 처리). 구체적 거리 계산식·가중치는 §1·§2 적용 후 실측 재현 여부를 보고 별도 설계 세션에서
# 확정한다 — 현재는 필요성만 design 세션에서 확인됨.
```

**§7 검증 계획**에서 §1·§2만 적용한 재실행 결과 자기교차가 사라지면 §3은 아예 불필요할 수
있다 — review_v1_5.md의 1·2차 원인(연속성 감쇠, Stage4 충격) 분석이 맞다면 §1·§2만으로
충분할 가능성이 높다(design 세션 Gemini conf98의 평가와 일치).

---

## §4. 절대 변경 금지 항목 (재확인)

- §4(중립 초기화+지연 엔트로피): `INIT_LOG_ALPHA_V14`, `ALPHA_ENT_START_EPOCH`,
  `ALPHA_ENT_ANNEAL_EPOCHS`, `alpha_ent_schedule()` 로직 — §2의 게이트 첨예화는 이 스케줄과
  **별도의 forward-시점 스케일링**이며 §4의 파라미터·로직 자체를 건드리지 않는다.
- §7(R1~R3 롤백): `DEATH_MP_THRESHOLD`, `DEATH_BURST_RATIO`, R1/R2/R3 raise 조건문.
- `uni_section_v18.py` 전체 — `compute_gates()`, `compute_gate_temperature()`,
  `build_collision_spec()`, `_signed_projections()` 모두 원본 그대로 호출만 한다.

---

## §5. 파일 헤더 갱신

`AI_design_v1_6.py` 파일 최상단 docstring(v1.5 항목 바로 위)에 v1.6 변경 이력을 추가한다.

```
AI_design_v1_6.py — Structural Collapse Fix (v1.6)
─────────────────────────────────────
[v1.6 변경, docs/command/command_v1_6.md 실행]
  §1 연속성 가중치 폐루프 하한: continuity_weight_schedule()의 무하한 감쇠(w_min=0.05) 위에
     실측 l_continuity 기반 폐루프 제어를 추가, 절대 하한 0.3을 보장.
  §2 Stage4 사전 게이트 첨예화: CGDN17.compute_gates_multi()에서 epoch 200~350 구간에
     log_alpha_candidates를 forward 시점에만 최대 3배 증폭해 게이트가 Stage4(epoch 500)
     진입 전 이미 0/1 근처에 도달하도록 유도. §4 엔트로피 스케줄(epoch 400 시작)과의 충돌을
     피하기 위해 원안(종료 400)보다 앞당겨 350에서 종료·고정.
  §3(P1, 미적용 가능): 파트 간 관통 페널티의 valid_mask 공백 — §1·§2로 해결 시 불필요.
  구현: /synod design 세션(Gemini flash conf98, OpenAI o3 conf77)에서 OpenAI가 제기한
  "collision_spec이 인접 셀 전제라 파트 간 교차를 못 잡는다"는 주장을 Claude가
  uni_section_v18.build_collision_spec() 코드 확인으로 반증(이미 전 파트쌍 커버) —
  §3을 새 메커니즘이 아닌 기존 메커니즘의 valid_mask 공백 보강으로 재정의했다. 또한
  OpenAI가 발견한 게이트 첨예화-엔트로피 스케줄 충돌(둘 다 epoch 400 근방에서 활성)을
  반영해 §2 종료 시점을 400→350으로 앞당겼다.
그 외 로직은 AI_design_v1_5.py와 동일.
─────────────────────────────────────
```

---

## §6. 검증 계획

1. **단위 테스트**: `get_adaptive_continuity_weight()` — epoch 0/100/400/500 입력 시 반환값이
   `max(sigmoid_baseline, w_adaptive)` 공식과 일치하는지, `w_floor=0.3` 밑으로 내려가지
   않는지 확인. `compute_gates_multi()`의 `beta_steep` — epoch 199/200/350/400/500에서
   1.0/1.0/3.0/3.0/3.0을 반환하는지 확인.
2. **스모크 테스트(20 epoch)**: 콘솔 로그에 `w_continuity`(폐루프 반영값)와 `log_alpha_sat`
   비율이 정상 출력되는지 확인.
3. **회귀 테스트(전체 700 epoch)**: v1.5가 보였던 패턴 재확인 —
   - `l_continuity`가 epoch 250~500 구간에서 0.8까지 커지지 않는지(목표: 0.3 근방 이하 유지)
   - epoch 400 근방(§2 종료 직후 + §4 entropy 시작)에서 `contrib_order`/gate 값에 ping-pong
     진동이 발생하지 않는지 — 발생 시 `GATE_STEEP_END`를 더 앞당기거나 `CONTINUITY_ETA`
     재조정
   - Stage4 진입 시점(epoch 500) `z_gate`가 0.05 이하 또는 0.95 이상에 대부분 도달했는지
     (목표: v1.5의 0.49~0.52 정체 대비 개선)
   - 최종 형상(3D 시각화)에서 Outer/Inner 자기교차가 사라졌는지 육안 확인
4. **§3 적용 여부 판정**: 3번 통과 시 §3은 적용하지 않고 문서화만 유지. 미통과 시 §3.2를
   구체 설계해 별도 command 문서(command_v1_7.md 등)로 분리 착수.

---

## 숙의 과정

<details>
<summary>Synod 세션 상세 (design 모드, Solver 1라운드 + Claude 코드 검증)</summary>

**Round 1 (Solver)**: Gemini(Architect, conf98)가 §1(PI형 폐루프, eta=10.0)·§2(beta_max=5.0,
epoch 200~400)·§3(Outer-Inner 유클리드 거리 기반 신규 배리어, 완전히 새로운 함수로 제안)에
대한 구체적 수식과 코드를 제시했다. OpenAI(Explorer, conf77)는 세 가지 실질적 위험을
지적했다 — (a) 폐루프 상태를 어디에 둘지의 재진입성/1-epoch 지연 문제, (b) **§2 첨예화
종료(원안 400)가 §4 엔트로피 시작(400)과 정확히 겹쳐 그래디언트 부호가 상충할 수 있다는
점**(코드로 검증됨 — 채택), (c) 기존 `collision_spec`이 인접 셀 전제라 파트 간 임의 교차를
못 잡는다는 주장(코드로 반증됨 — 기각).

**Claude 코드 검증(2차 비평 역할)**: `uni_section_v18.build_collision_spec()`을 직접 읽어
OpenAI의 (c) 주장이 사실과 다름을 확인 — 이미 섹션 내 모든 파트쌍을 커버한다. 실제 공백은
`_signed_projections()`의 `valid_mask`임을 코드 근거로 재정의했다. OpenAI의 DDP/체크포인트
우려는 이 코드베이스에 해당 메커니즘 자체가 없어 스코프 밖으로 기각했다. (b)는 검증 가능한
상수(`ALPHA_ENT_START_EPOCH` 기본값 = `ASWC_STAGE2_END` = 400)로 확인해 채택, §2 종료를
350으로 앞당겼다.

**Judge(Claude) 최종 판정**: §1 eta는 10.0→2.0 + EMA 평활 추가(진동 위험 완화), §2
beta_max는 5.0→3.0(정밀도 여유) + 종료 400→350(엔트로피 충돌 회피), §3은 신규 메커니즘이
아닌 기존 `valid_mask` 공백에 대한 후순위(P1) 보강으로 재정의, §1·§2만으로 재실행 후
필요성을 재판정하도록 §6에 조건부 게이트를 추가했다.

**신뢰 점수**: Gemini 98(T2.0), OpenAI 77(T1.2, DDP/체크포인트 등 스코프 밖 주장 일부로
감점, 단 (b) 발견의 기여는 최종 판정에 그대로 반영). **최종 신뢰도 91%**(§1·§2는 코드로
직접 재검증됐고, §3은 후순위·조건부로 명시해 불확실성을 문서 구조 자체에 반영).

</details>
