# uni_section v8 → v10 → v11 → v12 진화 분석 및 v13 방향 제안

- 작성: /synod review (Claude Judge + Gemini flash/high Architect + OpenAI o3/medium Explorer, 실 병렬 교차검증, conf 92%/92%, 조기 합의)
- 대상 코드: `CGNN/uni-section/code/uni_section_v8.py`, `v10.py`, `v11.py`, `v12.py`
- 대상 결과: `CGNN/uni-section/results/results_v8`, `results_v10`, `results_v11`, `results_v12`
- 프로젝트 목표(우선순위순): ① 전소성 모멘트(Mp) 목표 만족 ② 질량 감소(두께 축소/부재 제거) ③ Reasonable한 형상

---

## 1. 버전별 요약 (실측 로그 기반)

| 버전 | 핵심 설계 | 실측 결과 |
|---|---|---|
| **v8** | Ghost Gate α로 collision 가중 | **붕괴.** α가 조기에 1.0 포화 → collision 사실상 무력화. 두께 발산(+3.93mm), l_collision=53~557, feasibility 미달성 |
| **v10** | α 제거, 두께-연동 gap collision v5, mesh order loss, T_MAX 2.5mm, 2단계 두께학습 | **Mp 성공, 경량화 실패.** Mp err 0.01~0.03%, collision 0.016. 그러나 area +48% (1157→1713mm²) |
| **v11** | HardConcrete stochastic gate, one-sided mass loss, λ0 희소화 | **불안정.** epoch109 일시적 경량화(area 1447, 부재#2 제거) 성공하나 stochastic variance로 z가 진동, feasibility on/off 반복, 최종 area 여전히 +36% |
| **v12** | Deterministic gate, Augmented-Lagrangian 질량 페널티, 3단계 스케줄, Cascade 그래프 수술 | **안정적 수렴, 프루닝 미발동.** Mp/collision 깨끗이 수렴하나 z가 0.56까지만 하락, τ=0.1 트리거 도달 못함. 실제 부재 제거 0건, area 여전히 +30% |

### 버전별 상세

**v8 (기준선)**
- custom autograd Function 제거 → native autograd로 dMp/dt gradient 복원 (Envelope Theorem)
- collision: 파트쌍별 margin 자동산정(인접쌍만 검사), alpha(Ghost Gate)로 collision 가중
- mass loss: target_area 기반 + sigmoid gate (Mp err<5%에서만 활성)
- T_MAX=4.0mm, leaky tanh 두께헤드
- 결과: 300 epoch 끝까지 feasibility(Mp err<2% AND collision<0.05) 미달성. 최종 Mp err 44.76%, l_collision 53.2. Best-Mp epoch114에서 err 0.25%이나 그 순간 l_collision=557(완전 붕괴)

**v10 (collision v5 재설계 + mesh order + 2단계 두께학습)**
- collision v5: surface-gap = σ×projection − (t_a+t_b)/2 − clearance, 두께 t를 detach하지 않음(두께가 직접 밀어내기 기여), Ghost Gate alpha 완전 삭제, 전체 unordered 쌍(10쌍) 검사, 부호 앵커(σ) 고정
- mesh order loss 신규 추가(상시 활성) — 파트 내부 엣지 방향 반전(교차) 방지, v8 붕괴의 근본 대책
- 2단계 두께 학습: epoch 128까지 두께 사실상 동결(gate), 이후 sigmoid ramp + optimizer state 국소 리셋
- 비대칭 Huber phys loss (undershoot 2x 가중), T_MAX 2.5mm로 축소
- 결과: epoch 111에서 Mp err 0.01%, l_collision 0.0109 (feasible). 최종(epoch299) Mp err 0.03%, l_collision 0.0159, area 1713mm²(초기 1157mm² 대비 +48% 증가 — mass loss가 대칭적으로 area 보존을 강제해 오히려 증가). 부재 제거 없음

**v11 (v10 + 부재 프루닝/위상 경량화 추가)**
- HardConcreteGate(L0 relaxation)로 파트별 존재게이트 z∈[0,1], t_final = z·t_raw
- mass loss 개조: 대칭 → one-sided relu(area−target)² (초과만 처벌, 감소는 자유) + λ0·mean(E[z]) 희소화 보상
- collision violation에 z_i·z_j 곱해 사라지는 부재 무력화(clearance엔 곱하지 않음 — 부호반전 버그 회피)
- 보호파트 S_protect={0,1}=Outer Hat, Inner Plate → z=1 고정, λ0 어닐링: epoch128 이후 선형 warmup
- 결과: 최경량 feasible epoch109: Mp err 0.61%, area 1447mm², l_collision 0.0077, 부재#2(Inner Hat) 제거(z<0.5). 하지만 최종(epoch299)에는 z[2]=0.32로 다시 살짝 살아나는 등 z가 진동(stochastic 게이트로 인한 variance). area가 초기 대비 +36%(여전히 증가)

**v12 (v11 프루닝 정상화 + Cascade Stop-Snap-Resume 그래프 수술 파이프라인)**
- §1 정상화: 물리(Mp/면적/충돌)엔 결정론적 게이트 z_gate 사용(stochastic 샘플 제거, variance trap 해소), 희소화만 z_open(확률)에 부과
- feas_gate 데드락 제거 → Augmented-Lagrangian 질량 페널티(μ_m·relu(area/target−1), μ_m을 epoch마다 점증) + w_area·(area/area_init) 상시 압력
- 3단계 스케줄: Stage1(0-90) 형상, Stage2(90-150) 두께 언락, Stage3(150+) 프루닝 활성
- 프루닝 트리거: 후보 파트 z<0.1이 20 epoch 연속 → Cascade 그래프 수술(Y-Snap으로 이웃 파트에 노드 밀착 후 삭제, part_id 재색인, collision spec 재구축, warm-start로 다음 cycle 재학습)
- 보호파트 확장: {0,1,2}=Outer Hat, Inner Plate, Inner Hat
- 결과: Cycle 0에서 수렴, 그러나 프루닝 트리거 미발동. 최종 epoch299: MpErr(eval) 낮으나 Patch2(#08, part4)의 z가 0.56까지 하락(감소 추세지만 τ=0.1 트리거에 도달 못함). area(eval)=1499mm²(여전히 초기 1157 대비 +30%). "최경량 Feasible epoch89, 잔여파트 게이트 제거=없음" — v12는 실제 부재를 하나도 제거하지 못한 채 종료. Cascade 메커니즘 자체는 미검증(트리거링된 적 없음)

---

## 2. 발견된 문제

- **[ERROR]** v8: Ghost Gate α가 조기 포화되어 collision 항이 사실상 죽는 구조적 결함 — v10에서 근본 수정됨(해결됨, 회귀 방지 확인 필요)
- **[ERROR]** 전 버전 공통: Mp 목표(목표1)와 질량 감소(목표2)가 단일 가중합 손실 안에서 구조적으로 상충 — Mp 그라디언트가 항상 질량 항을 압도해 area가 초기값 대비 30~48% **증가**하는 상태로 수렴. "질량 감소"라는 프로젝트 2순위 목표가 사실상 4개 버전 모두에서 달성되지 못함(v11의 일시적 순간 제외)
- **[ERROR]** v12: 프루닝 트리거(z<0.1, 20epoch 연속)가 300epoch 전체에서 한 번도 발동하지 않음 — Cascade 그래프 수술 메커니즘 자체가 미검증 상태로 남음. 원인: (a) 희소화 압력(λ_s, μ_m)이 두께 헤드 그라디언트 대비 약함, (b) Stage3 시작이 epoch150으로 늦어 남은 150epoch 안에 z가 0.1까지 못 내려감, (c) 보호 파트 확장({0,1}→{0,1,2})으로 최적화 공간이 사실상 Patch1/Patch2 두 개로 좁아짐
- **[WARNING]** collision v5의 두께 t를 detach하지 않는 설계 — 두께가 gap 부호를 바꿀 수 있어 미분 불연속(sign flip) 발생 시 진동 유발 가능
- **[WARNING]** v12 Cascade 수술 시 옵티마이저 state가 part_id 재색인 후에도 그대로 재사용됨 — 삭제된 파트의 모멘텀이 다음 사이클 새 파트에 잘못 주입될 위험 (미검증, 실제 트리거된 적이 없어 잠재 버그)
- **[INFO]** mesh order loss가 상시 활성화되어 v8의 criss-crossing 붕괴를 성공적으로 예방(v10 이후 l_order≈0 유지) — 유지할 것

---

## 3. 권장 사항 (v13 방향, 우선순위순)

1. **Mp/질량을 단일 가중합에서 분리 — 제약 기반(Constrained/Dual-Ascent) 최적화로 전환.**
   Mp err와 collision을 하드 제약(ε-tolerance)으로 취급하고, 만족 시에만 area 최소화가 주 하강 방향이 되도록 Lagrange 승수(λ_Mp, λ_mass)를 별도로 점증. 현재는 μ_m(질량)만 Augmented-Lagrangian이고 Mp 쪽엔 동등한 승수가 없어 비대칭적으로 Mp가 항상 이김.
   - 대안(OpenAI 제안): Lexicographic 최적화 — Step-A에서 Mp err<ε 달성 후 constraint로 고정, Step-B에서 min(area) s.t. Mp err≤ε, collision≤δ

2. **프루닝 트리거 재설계.**
   - Stage3 시작을 epoch100~120으로 앞당기거나 λ_s를 지수 램프(epoch당 ×1.1)로 강화
   - 트리거 조건을 "20epoch 연속 z<0.1" 대신 EWMA(z)<τ (window≈10) 방식으로 완화해 노이즈에 강건하면서 빠르게 판정
   - S_protect를 {0,1}로 되돌려 Inner Hat(#06)도 프루닝 후보에 포함시켜 탐색 공간 확대

3. **z 딜레마(0.1~0.5 사이 half-dead 상태) 해소.**
   z(1−z) 형태의 entropy 정규화 항을 추가해 게이트가 0 또는 1 중 하나로 수렴하도록 유도.

4. **collision gap의 미분 불연속 완화.**
   현재 hard relu(−gap) 대신 softplus 기반 soft-min proxy(β anneal)로 대체해 sign flip 시 진동 방지.

5. **Cascade 수술 시 옵티마이저 state 위생 관리.**
   part_id 재사용 대신 UUID 키 기반 파라미터 관리, 삭제 파트는 옵티마이저 state에서 완전 제거.

6. **경량화-매끄러움 균형.**
   프루닝 활성 단계에서만 Laplacian smoothing / 두께 TV(total variation) penalty를 추가해 "국소 급격한 두께 집중"이 아닌 "고르게 얇아지는" 형상을 유도.

### 추가 엣지케이스 (OpenAI Explorer 발굴)
- mesh-order loss가 signed-area 기반이면 특정 원형 구간에서 blind spot(면적=0) 발생 가능 → winding-number 기반 topological loss로 교체 검토
- 두께가 0에 근접하면 PNA bisection의 평면 모멘트 계산이 non-smooth → t_clamp≥0.15mm 하한 또는 root-find tolerance 적응 필요
- Y-Snap 시 노드가 다른 파트와 exact coincident되면 Jacobian singular 가능 → ε jitter(±1e-4mm) 추가 검토

---

## 4. 신뢰도 및 숙의 과정

**신뢰도: 92%** — 두 모델(Gemini Architect, OpenAI Explorer)이 근본 원인(단일 가중합 손실에서 Mp가 구조적으로 질량 감소를 압도)과 처방(제약 기반 최적화로 전환)에 대해 실측 로그를 근거로 독립적으로 강하게 수렴. 세부 구현 방식(Gemini: Dual-Ascent 수식·entropy 정규화, OpenAI: lexicographic/ε-constraint·softplus 완화·엣지케이스 안전장치)은 상호 보완적.

<details>
<summary>모델 기여 및 신뢰 점수</summary>

- **Gemini (Architect, flash/high thinking):** v13 파이프라인 구조 제시(Dual-Ascent 공식화, entropy 정규화, S_protect 축소), 아키텍처 다이어그램. Confidence 92
- **OpenAI (Explorer, o3/medium reasoning):** 프루닝 실패의 그라디언트 스케일 분석, collision sign-flip 불연속성, Cascade 옵티마이저 state 오염 등 숨은 엣지케이스 다수 발굴. Confidence 92
- **Claude (Judge):** 코드/로그 직접 대조로 양측 주장 검증, 최종 통합

양측 모두 can_exit=true(조기 합의 조건 충족)로 Critic/Defense 라운드는 생략됨.

</details>
