# uni_section_v10 재출발(Reboot) 가이드 — v11~v14 필수 이식 항목 및 v10 자체 결함

- 작성: /synod review (Claude Judge + Gemini flash/high Architect + OpenAI o3/medium Explorer, 실 병렬 교차검증, conf 98/93, 조기 합의)
- 대상: 사용자가 `uni_section_v10.py`를 신규 베이스라인으로 선택하여 재개발을 시작하는 상황
- 근거 문서: `docs/review/review_v12.md`(v8~v12 리뷰), `docs/command/command_v12.md`, `docs/command/command_v13.md`, 각 버전 `.py` 헤더 docstring
- 프로젝트 목표(우선순위순): ① Mp 목표 만족 ② 질량 감소(두께 축소/부재 제거) ③ 합리적 형상

> **핵심 결론(양 모델 만장일치, conf 98/93):** v10은 ①(Mp)과 형상 안정성은 달성했으나, **대칭형 mass loss와
> collision v5의 두께 non-detach 설계 때문에 ②(질량 감소)가 구조적으로 불가능**하다(area +48%). v11~v13에서
> 여러 번의 시행착오 끝에 v13에서 안정화된 ALDA(Augmented Lagrangian Dual-Ascent) + EWMA/히스테리시스
> 프루닝 + softplus collision(β 상한) + 옵티마이저 전체 재구성 조합을 v10 위에 이식해야만 프로젝트 목표
> ②를 달성할 수 있다. v11/v12에서 시도했다가 v13에서 폐기된 메커니즘(stochastic gate 직접 적용, 단일임계
> EWMA, 부분 옵티마이저 리셋, 무제한 β anneal)은 **재도입 금지**.

---

## (a) v11~v14에서 확정되어 v10에 반드시 이식해야 하는 항목

| ID | 우선순위 | 항목 | 완료 기준 | 비고 |
|----|------|------|-----------|------|
| A-1 | **P0** | ALDA 전환: area를 objective로, Mp·collision을 하드 제약(적응형 μ, ρ)으로 재정식화 (v13 §1) | Mp err<1~3%, l_collision<0.02 유지하며 μ가 1e5 미만에서 수렴 | μ 상한 1e5, grad-clip 10, 10epoch마다 승수 갱신 |
| A-2 | **P0** | Mass loss 비대칭화: `relu(area/target − 1)²` (초과만 처벌) 또는 A-1의 area-as-objective로 대체 | area≤target일 때 gradient=0 확인 | v10의 "area +48%" 회귀의 직접 원인 제거 |
| A-3 | **P0** | 게이트 이원화: 물리(Mp/면적/충돌)엔 **결정론적** z_gate, 희소화엔 확률적 z_open (v12 §1.1) | forward pass가 analytic gate와 diff<1e-6 | variance를 제약 조건에서 분리 |
| A-4 | **P0** | EWMA + 히스테리시스 프루닝 트리거 (α=0.15, ALIVE↔PENDING_DELETE 임계 0.08/0.15, 20epoch 확정) (v13 §2) | 삭제→복구 진동 시뮬레이션 테스트 통과 | 단일임계 EWMA 금지(§(c) C-2 참조) |
| A-5 | **P0** | Cascade 그래프 수술 시 옵티마이저 **전체 재구성**(부분 리셋 금지) (v13 §5) | 수술 직후 전 파라미터 Adam step=0부터 재시작 확인 | v12 잠재버그(모멘텀 오염) + v10 자체 버그(B-4) 동시 해결 |
| A-6 | **P0** | Collision v6: softplus(β anneal) 프록시 + **β≤60 하드캡** + **두께 t detach** (v13 §4) | d<-5mm에서 gradient 소실 없음(수치 확인), sign-flip 스파이크 없음 | v10 collision v5의 non-detach 설계(B-2) 대체 |
| A-7 | P1 | z(1-z) entropy 정규화, Mp err<3%일 때만 활성(alpha_ent coupling) (v13 §3) | 초기 epoch entropy loss<1e-3 | 단독 사용 시 조기 하중경로 붕괴 위험 — 반드시 Mp err와 결합 |
| A-8 | P1 | Laplacian/TV 상시 저가중 + 프루닝 구간 스케일업(on/off 이분법 폐기) (v13 §6) | 수술 전후 두께 표준편차 감소 확인 | v10 mesh order loss(상시 1.0 고정)와의 상호작용 주의(B-6) |
| A-9 | P1 | t_clamp=0.15mm(PNA 직전) + z_gate<0.02시 PNA 호출 skip(해석적 Mp=0) (v13 §7) | 두께 0 근접 부재에서 solver crash/NaN 없음 | |
| A-10 | P1 | Y-Snap 좌표중첩 방지 jitter(±1e-4mm) (v13 §7) | 스냅 후 모든 노드쌍 거리>1e-5 | Jacobian singular 방지 |
| A-11 | P2 | 로깅/시각화 패널 복원(v14): 파트별 두께 추이, 보조손실항 로그축 비교, g_Mp/g_col 진단 패널 | 20epoch마다 그림 저장 확인 | v10에 있었으나 v11~v13에서 유실됐던 패널 포함 |

---

## (b) v10 코드 자체의 결함 (v11~v14와 무관하게 v10만 봐도 존재)

| ID | 우선순위 | 결함 | 근거 (v10 코드) | 수정 |
|----|------|------|------------------|------|
| B-1 | **P0** | 대칭형 mass loss `((area-target)/target)²` — 면적 감소도 증가와 동일하게 처벌 | `compute_mass_loss()` | A-2 채택 |
| B-2 | **P0** | Collision v5가 두께 `t_final`을 **detach하지 않음** → gap 부호(σ·proj − t_sum − clearance)가 sign-flip될 때 미분 불연속·진동 가능 | `compute_collision_loss_v5()` docstring: "두께 t는 detach 금지... 밀어내기 gradient 경로가 핵심 메커니즘" — 설계 의도였으나 v13 Critic 검증으로 반증됨 | A-6 채택(softplus + detach) |
| B-3 | P1 | `build_collision_spec()`이 **초기 형상에서 σ(부호 앵커)를 한 번 고정**하여 이후 학습 내내 재계산하지 않음 — 학습이 크게 진행되어 파트 상대 위치·접촉 방향이 초기와 달라지는 경우(특히 A-1/A-4로 부재가 실제로 삭제·이동하는 재출발 시나리오) false-negative 판정 위험 | `build_collision_spec()` §2/D2 설계 근거 | A-6(softplus proxy)으로 전환 시 명시적 σ 앵커 메커니즘 자체가 대체되므로 함께 해소되는지 확인 필요. 유지한다면 주기적 재계산 또는 재검증 로직 추가 |
| B-4 | **P0** | 2단계 두께 학습에서 epoch 128 진입 시 `thickness_decoder` 파라미터 그룹만 **부분적으로** AdamW state 리셋 — Adam step counter 불일치로 살아남은 파라미터의 bias-correction이 일시적으로 어긋나 과대 스텝 발생 가능(v13 Critic이 동일 메커니즘을 수치 검증: 최대 6배) | `run_training()`: `optimizer.state.pop(p, None)` on `thickness_decoder` group only | A-5(전체 재구성)를 Stage 전환 시점에도 동일 원칙으로 적용 |
| B-5 | P1 | `T_MAX=2.5mm` 하드 클램프 + 대칭형 mass loss 조합이 두께를 상한 근처에 안착시키는 방향으로 압력을 가함(B-1 해소 전까지는 T_MAX 자체보다 mass loss 비대칭화가 우선) | `CGDN.T_MAX`, `compute_mass_loss()` | A-2/A-1 적용 후 T_MAX가 여전히 병목인지 재평가. 필요시 stage-wise 하한 completion |
| B-6 | P2 | Mesh order loss가 **상시 가중치 1.0 고정**으로 설계됨 — A-8(Laplacian/TV 프루닝 구간 스케일업)이나 Cascade 그래프 수술(부재 삭제·재색인) 도입 시, 수술 직후 위상이 급변하는 구간에서 과잉 제약으로 작용할 가능성 (v10 단독으로는 문제 없음 — A-4/A-5 이식 후 상호작용 검증 필요) | `compute_mesh_order_loss()`, `get_curriculum_weights_v10()`: `s_order` 커리큘럼 없이 상시 1.0 | 그래프 수술 후 N epoch 동안 order loss 완화 또는 재초기화 검토, 회귀 테스트(criss-cross=0 유지) |

---

## (c) v11~v13에서 시도했다가 v13/v14 이후 폐기·수정된 것 — 재출발 시 "재시도 금지"

| ID | 우선순위 | 재도입 금지 항목 | 폐기 근거 |
|----|------|-------------------|-----------|
| C-1 | **P0** | 물리 손실(Mp/면적/충돌)에 **stochastic 게이트 z를 직접 적용** (v11 HardConcreteGate 원안) | variance trap → 수렴 발산(v11 loss 1→5), epoch109 일시 성공 후 재붕괴 |
| C-2 | **P0** | **단일 임계치** EWMA 프루닝 트리거(순수 `EWMA(z)<0.08`, 히스테리시스 없음) | 삭제→복원 진동(오탐지 다발)을 v13 Critic 라운드가 검증 |
| C-3 | **P0** | Cascade/Stage 전환 시 **부분** 옵티마이저 state 리셋(v10/v12 방식) | Adam step counter 불일치로 순간 최대 6배 과대 스텝(v13 Critic 검증) — v10 자체에도 동일 결함 존재(B-4) |
| C-4 | P1 | Softplus collision β를 **무제한 anneal**(원안 스케줄대로면 epoch300에 β≈760) | 접촉 근방 gradient 소멸(d<-5mm에서 미분값 1e-4 이하), 침투 방치 |
| C-5 | P1 | Ghost Gate α 부활 (v8 방식) | v10에서 이미 근본 폐기됨 — softplus proxy(A-6)가 이를 완전히 대체하므로 재도입 불필요 |
| C-6 | P2 | Laplacian/TV 정규화의 On/Off 이분법 스케줄 | 게이트 히스테리시스로 ALIVE 복귀 시 두께 불연속 유발 |

---

## Pre-Flight 검증 순서 (재출발 착수 전 권장)

1. A-1~A-6(P0) 패치를 v10 위에 병합 → 단위 테스트 통과 확인
2. B-1/B-2/B-4(P0) v10 고유 결함을 A-2/A-6/A-5로 동시 해소 → collision·area smoke-test 재실행
3. A-7~A-10(P1) 통합, 토이 모델에서 최소 1개 부재가 실제로 프루닝되는지 확인 (v11/v12가 실패했던 지점)
4. C-시리즈 금지 항목이 코드에 재도입되지 않았는지 정적 검토
5. A-11 로깅 활성화 후 5epoch dry-run으로 그림 저장 확인
6. B-3/B-6은 A-4/A-5/A-8 이식 완료 후 상호작용 관점에서 별도 재검증

---

## 신뢰도 및 숙의 과정

**신뢰도: 95%** — Gemini(Architect, flash/high thinking, conf 98)와 OpenAI(Explorer, o3/medium reasoning, conf 93) 모두 `can_exit=true`로 조기 합의. 두 모델 모두 독립적으로 (1) v10의 대칭 mass loss와 collision non-detach가 근본 결함이라는 점, (2) v13의 ALDA+EWMA 히스테리시스+softplus+옵티마이저 전체재구성 조합이 필수 이식 대상이라는 점, (3) v11/v12에서 폐기된 메커니즘의 재도입 금지에 수렴했다.

<details>
<summary>모델 기여 및 신뢰 점수</summary>

- **Gemini (Architect, flash/high thinking, conf 98):** ALDA 전환을 포함한 구조적 로드맵(Phase 1~3) 제시, v10 결함을 mass loss·optimizer 국소리셋 2건으로 압축 정리.
- **OpenAI (Explorer, o3/medium reasoning, conf 93):** 실행 가능한 ID 기반 체크리스트(A/B/C 시리즈) 및 Pre-Flight Gate 절차 제시, Gemini가 놓친 v10 고유 결함 3건(B-3 σ 고정 앵커의 false-negative 위험, B-5 T_MAX-mass loss 상호작용, B-6 mesh order loss와 그래프 수술 상호작용)을 추가 발굴.
- **Claude (Judge):** 두 응답을 대조해 중복 항목 통합, OpenAI가 발굴한 v10 고유 결함(B-3/B-5/B-6)을 v10 소스 코드(`build_collision_spec`, `T_MAX`, `compute_mesh_order_loss`)와 직접 대조하여 채택, 최종 문서화.

양측 모두 `can_exit=true`(조기 합의 조건 충족)로 Critic/Defense 라운드는 생략됨.

</details>
