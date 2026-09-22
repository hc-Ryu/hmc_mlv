# idea_v1_6.md — v1.5 구조적 붕괴(자기교차) 해결 설계 아이디어

작성일: 2026-08-09 (`/synod idea` 세션, Gemini flash conf 95→90 + OpenAI gpt4o conf 85→75)
근거: `docs/review/review_v1_5.md`(구조적 붕괴 원인 분석 — 1차 원인: `w_continuity` 무하한
감쇠, 2차 증폭: Stage4 게이트 강제 이진화 충격, v1.5 Huber 상한은 원인에서 제외)
대상: `reports/v1_5/AI_design_v1_5.md`(실행 로그), `reports/v1_5/AI_design_v1_5_result.png`
(Outer/Inner 파트 윤곽선이 바깥으로 튀어나오고 자기교차하는 최종 형상)

---

## 아이디어 평가

### 1. `w_continuity`에 폐루프(closed-loop) 적응형 하한 도입 (최우선, P0)

**설계**: 현재 `continuity_weight_schedule(epoch, ASWC_STAGE2_END=400)`은 1.0→0.05로
무하한 감쇠한다. 이를 고정 스케줄이 아니라, 실측 `l_continuity`가 허용 오차 τ를 넘으면
가중치를 다시 끌어올리는 **PID형 폐루프 제어**로 교체한다.

```
w_continuity(t+1) = max(w_floor, w_continuity(t) + alpha * (l_continuity(t) - tau))
w_floor = 0.3   # 하한 — review_v1_5.md 권고치와 동일
tau     = 0.1   # 허용 위반 임계(연속성 손실이 이 이하면 가중치를 더 낮춰도 안전)
alpha   = 하이퍼파라미터, 초기값 0.5 권장(재보정 필요)
```

- **왜 고정 floor(예: 상수 0.3)가 아니라 폐루프인가**: 1라운드에서 OpenAI가 제기한 반론 —
  17섹션의 `target_mp`가 26M~47M N·mm로 폭넓게 분포하므로, 고정된 높은 continuity 가중치는
  섹션별로 정당하게 필요한 형상 차이까지 과도하게 억제할 위험(over-constraining)이 있다.
  2라운드 Critic에서 Gemini가 이 우려에 대해 "물리적 유효성(비자기교차)은 협상 불가능한 하드
  제약이며, Mp 목표를 맞추기 위해 형상 정합성을 희생하는 것 자체가 실패 모드"라고 반박했고,
  Claude(Judge)도 이 반박에 동의한다 — 다만 **그 실행 수단으로는 고정 상수보다 폐루프 제어가
  기술적으로 우월**하다(위반이 작을 때는 자동으로 완화되어 OpenAI가 우려한 경직성 문제를
  구조적으로 해소하면서도, 위반이 커지면 즉시 재개입한다). 두 모델의 상충 지점은 "제약을 걸어야
  하는가"가 아니라 "고정 상수 vs 적응형"이었다 — 후자로 판정.
- **주의(Missing Consideration, Critic 라운드)**: epoch 200~400 구간(아래 아이디어 2의 게이트
  어닐링과 겹침)에서는 게이트 변화로 인해 좌표의 물리적 의미 자체가 바뀌는 "가상(transient)"
  연속성 위반이 발생할 수 있다 — 이 구간에서는 PID 컨트롤러의 반응 속도를 감쇠(damping)시켜
  일시적 위반에 과잉 반응하지 않도록 해야 한다.
- **실현 난이도**: 하 — 매 epoch 종료 시 스칼라 가중치 1개를 업데이트하는 5줄 내외 코드.
- **크래시 위험**: 없음 — 그래디언트 계산 방식 자체는 건드리지 않고, 기존 손실 항의 가중치
  스케줄만 바꾼다(v1.5의 Huber 상한과 완전히 독립).

### 2. Stage4 하드 이진화를 온도 어닐링으로 대체 (최우선, P0)

**설계**: 현재 epoch 500에 candidate 게이트(z_gate)를 한 번에 ±5.0으로 강제 이진화하는
Stage4 진입을, epoch 200~400 구간에 걸친 **Gumbel-Softmax/시그모이드 온도 어닐링**으로
대체한다.

```
z_gate(t) = sigmoid(beta(t) * z_gate_logit)
beta(t): 1.0(epoch 200) -> 10.0(epoch 400)로 점진 증가
```

- **근거**: 두 모델이 1라운드에서 **독립적으로** 동일한 아이디어에 수렴했다(Gemini는
  "Homotopy-Based Gate Annealing", OpenAI는 "Gate Stability Pre-Stage4"라는 이름으로 각각
  제안). 2라운드에서 이 수렴 자체를 "낮은 리스크의 검증된 합의"로 재확인했다. candidate
  게이트가 z≈0.5 부근에서 300+ epoch 방치되는 근본 문제(§4의 원래 목적이 이 두 게이트에서
  작동하지 않음, review_v1_5.md 3번 권고사항)를 훈련 후반부의 급격한 topology 충격 없이
  해소한다.
- **주의(Missing Consideration, Critic 라운드— Gemini)**: 온도 β(t)가 커질수록(gate가
  binary에 가까워질수록) 그래디언트가 소실되는 saturation 문제가 발생한다. **Straight-Through
  Estimator(STE)** 적용 또는 최소 온도 하한(β_max에서도 완전히 0/1로 굳지 않도록 T_min 설정)이
  필요 — 이 보완 없이 순수 온도 스케일링만 적용하면 좌표 최적화가 조기에 멈출 위험이 있다.
- **실현 난이도**: 중 — candidate 파트의 forward pass에서 게이팅 변수 계산부를 온도 스케일로
  감싸야 한다(기존 `compute_gates`/HardConcrete 로직과의 상호작용 확인 필요).
- **크래시 위험**: 낮음 — 급격한 이진화가 제거되므로 오히려 Stage4 진입 시점의 손실 스파이크
  위험이 줄어든다. STE 미적용 시 그래디언트 소실로 인한 학습 정체 위험은 있음(크래시는 아님).

### 3. 파트 간(inter-contour) 자기교차 직접 방지 페널티 신설 (우선, P1)

**설계**: 1라운드에서 두 모델이 서로 다른 메커니즘을 제안했다 — Gemini는 `contrib_order`를
clipped log-barrier로 강화(파트 **내부** 정점 순서 위반을 무한대에 가깝게 억제), OpenAI는
슈레이스(signed-area) 공식 기반 직접 페널티를 제안했다. **2라운드 Critic에서 Gemini가 두
제안 모두를 기각**했다 — 관찰된 붕괴는 Outer와 Inner처럼 **서로 다른 파트 간의 교차**인데,
1D 정점 순서(vertex ordering)도 슈레이스 공식(단일 폐다각형 전제)도 이 문제를 근본적으로
다루지 못한다는 것이 기술적으로 타당한 지적이다. 따라서:

- **채택안**: 파트 간 **선분 교차 판정(line-segment intersection, 외적 기반) 페널티** 또는
  **부호付 거리장(Signed Distance Field, SDF) 배리어**를 Outer-Inner, Outer-Plate 등 실제로
  겹치면 안 되는 파트 쌍에 대해 추가한다. 이는 기존 `compute_collision_penalty_unclamped()`
  (v1.5에서 Huber 상한을 적용한 바로 그 함수, `collision_spec`으로 파트 쌍을 순회하는 구조가
  이미 존재)와 **동일한 구조를 재사용**할 수 있다는 것이 이번 세션에서 확인된 핵심 통찰 —
  즉 완전히 새로운 항이 아니라, 기존 충돌 페널티 파이프라인에 "관통 거리"뿐 아니라 "교차 여부"
  판정을 추가하는 확장으로 구현 가능하다.
- **v1.5와의 관계**: 이 페널티도 Huber 상한을 그대로 적용해야 한다 — 그렇지 않으면 v1.4가
  겪었던 것과 동일한 무클램프 폭주 위험을 새 항에서 재현할 수 있다(review_v1_5.md가 검증한
  기존 Huber 인프라를 재사용).
- **실현 난이도**: 중상 — 선분 교차 판정은 파트 쌍마다 O(n²) 비교가 필요할 수 있어, 기존
  `collision_spec`의 153쌍(섹션×파트쌍) 구조에 얹되 연산 비용을 프로파일링해야 한다.
- **크래시 위험**: 낮음(Huber 상한 재사용 전제 시) — 상한 없이 구현할 경우 High.

### 4. 사후 형상 유효성 검증 게이트 (보류, P3 — 채택하지 않음)

1라운드에서 OpenAI가 "훈련 중 방지 대신 Stage4 출력물에 대한 사후 convexity/유효성 검사"를
대안으로 제시했으나, 스스로 "계산 비용이 크고 정상적인 설계를 과도하게 기각할 위험"을
지적하며 낮은 우선순위로 분류했다. Claude(Judge) 판정: 이는 **탐지(detection)일 뿐
방지(prevention)가 아니므로** 위 1~3번이 실패했을 때의 최종 안전망으로만 가치가 있다 — 이번
버전(v1.6)의 1차 대응으로는 채택하지 않는다.

---

## 권장 구현 순서

1. **[P0] 아이디어 1(연속성 폐루프 하한)** — 가장 낮은 구현 난이도로 review_v1_5.md의 1차
   원인을 직접 해소. 단독 적용만으로 재실행 후 `l_continuity`/`contrib_order` 추이 재확인 권장.
2. **[P0] 아이디어 2(게이트 온도 어닐링)** — 2차 증폭 원인 해소. STE 또는 온도 하한 보완 필수.
3. **[P1] 아이디어 3(파트 간 교차 직접 페널티)** — 1·2번 적용 후에도 자기교차가 재발하면 추가.
   기존 `compute_collision_penalty_unclamped()` 인프라(v1.5 Huber 상한 포함)를 재사용해
   구현 리스크를 낮춘다.
4. **[보류] 아이디어 4** — 1~3번이 모두 실패할 경우에만 안전망으로 재검토.

**§6 불변 원칙 준수 확인**: 위 아이디어 모두 `uni_section_v18.py`, §4(중립 초기화+지연
엔트로피 — 아이디어 2는 이와 별개로 candidate 게이트의 *출력 스케일링*만 다루며 초기화·엔트로피
스케줄 자체는 건드리지 않음), §7(R1~R3 롤백 규칙)을 변경하지 않는다. `w_continuity` 스케줄과
Stage4 진입 로직은 애초에 이 불변 목록에 속하지 않았다(review_v1_5.md 4번 권고 참고).

---

<details>
<summary>Synod 세션 상세 (idea 모드, Solver→Critic 2라운드, Defense 라운드는 생략)</summary>

**Round 1 (Solver)**: Gemini(Architect, conf95)는 (1)연속성 가중치 PID 폐루프, (2)Stage4
게이트 온도 어닐링, (3)vertex ordering 로그-배리어 3가지를 제안하고 우선순위표로 정리했다.
OpenAI(Explorer, conf85)는 연속성 floor의 과도한 제약 위험을 지적하며, 직접적인 자기교차
페널티(슈레이스 공식)와 독립적으로 동일한 게이트 어닐링 아이디어를 제시했다. 사후 유효성
검사는 스스로 낮은 우선순위로 강등했다.

**Round 2 (Critic)**: Gemini가 자기 자신과 OpenAI의 자기교차 방지안(vertex ordering,
슈레이스 공식) **둘 다를 기술적으로 반박**했다 — 관찰된 붕괴가 파트 **내부**가 아닌 파트
**간** 교차이므로 1D 순서·단일 다각형 전제의 슈레이스 공식은 부적합하며, 선분 교차/SDF 기반
파트-쌍 배리어가 필요하다는 대안을 제시했다. 또한 OpenAI의 "over-constraining" 우려를
"물리적 유효성은 협상 불가능한 하드 제약"이라는 논리로 기각하되, 그 실행 수단은 고정 상수가
아닌 폐루프 제어가 맞다고 정리했다. Gumbel-Softmax 온도 어닐링의 그래디언트 소실(saturation)
위험도 새로 지적해 STE 보완을 권고했다. OpenAI critic(conf75)은 PID 미세조정 부재, 두 페널티
메커니즘의 동등성 가정에 대한 신중론을 유지했으나 새로운 반박 근거를 제시하지는 못했다.

**Judge(Claude) 최종 판정**: 아이디어 1(폐루프 연속성 하한)·2(게이트 온도 어닐링)를 P0로
채택, 아이디어 3(파트 간 직접 교차 페널티, Gemini의 2라운드 수정안 기준)을 P1 후속으로 채택,
사후 검증 게이트는 보류. "over-constraining이냐 vs 하드 제약이냐"는 논쟁은 실행 메커니즘
선택(폐루프 vs 고정 상수) 문제로 재구성해 해소했다.

**신뢰 점수**: Gemini 95(solver)→90(critic, 자기 제안 일부 반박 포함), OpenAI 85(solver)→
75(critic, 새 근거 부족으로 소폭 하향). **최종 신뢰도 84%**.

</details>
