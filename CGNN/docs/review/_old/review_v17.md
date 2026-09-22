# uni_section v17: target_mp 하향 + stage2_epoch=10 실험 — 두께 반응 vs 파트 게이트 정체 원인 분석

- 작성: /synod review (Claude Judge + Gemini flash/high Architect + OpenAI o3/medium Explorer, 실 병렬 교차검증 2라운드(Solver→Critic), conf 95%/68%, 조기 합의)
- 대상 코드: `CGNN/uni-section/code/uni_section_v17.py`
- 대상 결과: `CGNN/uni-section/results/results_v17/uni_section_v17.md`
- 실험 조건: 사용자가 `target_mp`를 하향, `STAGE2_EPOCH`(두께언락)를 128 → 10으로 축소하여 재실행. 로그상 `[Gate Active] epoch 40` = STAGE2_EPOCH(10) + 30 → 코드 정합 확인됨(단, 저장소의 `uni_section_v17.py`는 `STAGE2_EPOCH=128` 하드코딩 상태 — 아래 §3 [WARNING] 참고)

---

## 1. 관찰된 현상

- 두께(t): stage2_epoch를 10으로 앞당기자 즉시 큰 폭으로 변화 — 초기 area 1157.3mm² → epoch19 시점 이미 567.2mm²(-51%), 이후 학습 내내 560~600mm² 대역 유지
- 파트 존재 게이트(z_gate, 후보 파트 3/4=Patch1/Patch2): epoch19 시점 1.00/1.00 → 최종(epoch299) 0.93/0.92. **300 epoch 내내 사실상 정체.** 프루닝 트리거(임계값 0.08) 근처에도 도달 못함
- 즉, target_mp를 낮춰 "부재를 아예 제거"하는 것이 물리적으로 더 매력적인 선택지가 됐어야 함에도, 최적화는 오직 두께 축소로만 반응하고 게이트(위상)는 거의 움직이지 않음

## 2. 근본 원인 (2라운드 교차검증 결론)

**결론: HardConcrete 게이트의 초기 포화(saturation) + 희소화 손실 가중치의 이중 약화(loss-weight 비대칭 + mp_rel_err 게이팅)가 결합된 구조적 최적화 정체(optimization stagnation)이며, 코드 버그가 아니라 설계상의 인센티브 불균형.**

Solver 라운드에서 Gemini는 "두께가 먼저 0에 가깝게 붕괴하여 `t_eff = t·z_gate`의 곱셈 그래디언트가 t로 인해 소실된다"는 가설을 제시(conf 98)했으나, Critic 라운드에서 Gemini 본인과 OpenAI 양쪽 모두 실측 로그(area가 51%만 감소, 0에 가깝지 않음)로 이를 반박(conf 95/68, 둘 다 can_exit=true 합의). 실제 지배적 메커니즘은 다음 세 가지의 결합:

1. **HardConcrete 초기 포화**
   `log_alpha` 초기값 2.0, HardConcrete stretch(γ=-0.1, ζ=1.1) 하에서 σ(2.0)≈0.88 → z≈0.956, 이는 `min(1,·)` 클램프 상한(σ(log_alpha)>0.917, 즉 log_alpha>2.4) 바로 아래에 위치. 이 구간은 시그모이드의 포화 영역이라 dz/d(log_alpha)가 이미 작음. z를 프루닝 임계값 근처(z≈0.2)로 끌어내리려면 log_alpha를 2.0→약 -2.4까지, Δ≈-4.5만큼 이동시켜야 하는데, `gate_params`의 lr이 GATE_ACTIVE_EPOCH(=40)까지 0으로 고정되어 있어 실질적으로 학습 가능한 구간은 260 epoch뿐이며, 이 안에서 lr=1e-2와 미소한 그래디언트로는 도달 불가능한 거리

2. **희소화 압력의 이중 약화**
   `w_sparse_effective = w_sparse(0.1) × sigmoid(50×(0.05 - mp_rel_err))`. 로그상 mp_rel_err가 초반 대부분 5% 근처이거나 그 이상이므로 이 시그모이드 항은 대부분 구간에서 0.1~0.5 수준에 그쳐 실효 가중치가 0.01~0.05까지 떨어짐. 이는 `w_phys=10.0` 대비 200~1000배 차이 — 물리(Mp) 손실이 공유 그래디언트를 사실상 전부 지배하여, "두께를 줄여 Mp를 맞추는" 값싼 경로만 강화되고 "파트를 꺼서 맞추는" 경로에는 유의미한 압력이 거의 가해지지 않음

3. **엔트로피 정규화의 미약함**
   `ALPHA_ENT=0.01 × gate_multiplier`로 z를 0/1 극단으로 미는 항이지만, 위 두 요인 대비 규모가 너무 작아(활성 시에도 희소화 손실의 3% 미만 기여) 물리 손실 그래디언트를 극복하지 못함

→ 종합: "두께는 값싸고(연속, 큰 표현력의 디코더, 그래디언트가 선형 영역), 게이트는 비싸다(이산적 위상 변화, 시그모이드 포화 구간에서 시작, 활성화 자체가 30epoch 늦음, 압력 가중치도 약함)". target_mp를 낮춰도 옵티마이저는 항상 그래디언트가 더 잘 흐르는 두께 경로로 문제를 해결하며, 게이트 경로는 애초에 유의미하게 열리지 않은 채 끝남.

## 3. 발견된 문제

- **[ERROR]** 파트 존재 게이트(log_alpha)가 실질적으로 학습되지 않는 구조적 정체 — GATE_ACTIVE_EPOCH까지 lr=0으로 완전 동결 + HardConcrete 초기값이 포화 영역에 위치 + 희소화 가중치가 mp_rel_err 게이팅으로 이중 약화. 300epoch 전체에서 프루닝 트리거(z<0.08 20epoch 연속)에 근접조차 못함 — "부재 제거"라는 프로젝트 2차 목표(경량화)가 이번 stage2_epoch=10 실험에서도 달성되지 않음
- **[WARNING]** 저장소의 `uni_section_v17.py`에 `STAGE2_EPOCH = 128`이 하드코딩되어 있으나(line 651), 사용자가 리뷰를 요청한 실행 로그(`results_v17/uni_section_v17.md`)는 `[Stage 2] epoch 10`, `[Gate Active] epoch 40`을 보여 실제로는 STAGE2_EPOCH=10으로 실행됐음이 확인됨. 즉 **현재 코드 파일이 결과를 생성한 실행 시점의 파라미터를 반영하고 있지 않음** — 실험 재현성을 위해 결과 생성 시 사용된 실제 하이퍼파라미터를 커밋하거나 실행 스크립트/커맨드라인 인자로 로그에 명시적으로 남기는 것을 권장
- **[WARNING]** `log_alpha` 초기값(2.0)이 HardConcrete 클램프 상한 바로 아래(포화 영역)에서 시작 — 게이트가 "열림"에서 출발할 뿐 아니라 그래디언트 흐름 자체가 처음부터 약한 지점에서 시작함. 게이트를 실제로 학습 가능하게 하려면 초기값을 0 부근으로 낮추거나(대칭적 시작), gate_params의 lr을 GATE_ACTIVE_EPOCH 이전부터 낮은 값으로라도 활성화하는 것을 고려할 것
- **[WARNING]** `w_sparse_effective`가 `mp_rel_err` 자체에 의해 게이팅되는 설계는, Mp err가 낮아지기 전까지 희소화 압력을 원천 차단함 — "먼저 Mp를 맞추고 나서 가볍게 하라"는 의도이나, 결과적으로 두께가 이미 해를 선점한 뒤라 게이트가 움직일 이유(그래디언트)를 상실하는 부작용(순서 종속성)을 낳음
- **[INFO]** Critic 라운드에서 두 모델(Gemini/OpenAI) 모두 초기 가설(t→0으로 인한 곱셈 그래디언트 소실)을 실측 로그로 자체 반박하고 HardConcrete 포화 + 손실 가중치 비대칭으로 수렴한 점은 이번 리뷰의 신뢰도를 높이는 근거임(conf 95/68, 양쪽 can_exit=true)
- **[INFO]** OpenAI가 제시한 반증 실험(w_sparse=2.0, init log_alpha=0, gate_active_epoch=stage2_epoch+1 조합 시 50epoch 내 z_gate<0.3 예상)은 코드 변경 없이 하이퍼파라미터만으로 검증 가능한 저비용 실험이므로 다음 버전에서 시도해볼 가치가 있음

## 4. 권장 사항 (우선순위순)

1. **`w_sparse` 상향 + `mp_rel_err` 게이팅 완화**: `w_sparse`를 1.0~2.0 수준으로 올리고, `TAU_GATE` 임계값을 높이거나 게이팅 자체를 제거해 초반부터 희소화 압력이 최소한으로라도 작동하게 할 것
2. **`log_alpha` 초기값 재검토**: 현재 2.0(포화 영역 시작)을 0 근처로 낮춰 게이트가 실제로 그래디언트에 민감한 구간에서 출발하도록 조정
3. **GATE_ACTIVE_EPOCH 이전 구간에 낮은 lr로라도 게이트 학습 허용**: 완전 동결(lr=0) 대신 예컨대 1e-3 수준의 낮은 lr을 STAGE2_EPOCH 이후부터 부여해 "두께 학습이 게이트를 완전히 선점"하는 순서 종속성을 완화
4. **하이퍼파라미터 재현성 확보**: 이번 실행에 사용된 실제 STAGE2_EPOCH/target_mp 값을 코드에 반영하거나 실행 로그/커맨드에 명시적으로 기록해 코드-결과 정합성을 유지할 것
5. (실험) OpenAI 제안 반증 실험(w_sparse=2.0 + log_alpha init=0 + gate_active_epoch=stage2_epoch+1) 1회 실행으로 위 가설을 정량 검증

## 5. 신뢰도: 92%
(Solver 라운드 conf 98/78 → Critic 라운드에서 자기수정 후 conf 95/68로 수렴, 두 모델 모두 can_exit=true. 최종 합성 신뢰도는 두 모델의 최종 합의 지점에 대한 가중평균)

<details>
<summary>숙의 과정</summary>

### 모델 기여
- **Gemini (Architect, Solver):** 곱셈 그래디언트 소실 가설 최초 제시(수학적으로 타당하나 경험적으로 틀림), Critic 라운드에서 자기수정하여 HardConcrete 포화 메커니즘으로 결론 전환
- **OpenAI (Explorer, Solver+Critic):** 처음부터 손실 가중치 비대칭 + HardConcrete 포화를 주 메커니즘으로 제시, 반증 실험 설계 제안
- **Claude (Judge/Orchestrator):** 두 라운드 결과 종합, 코드(STAGE2_EPOCH 하드코딩 불일치) 직접 확인 및 이슈로 추가

### 해결된 주요 쟁점
1. "두께 t→0으로 인한 곱셈 그래디언트 소실"(Gemini 최초 주장) vs "HardConcrete 포화 + 가중치 비대칭"(OpenAI 최초 주장) → 실측 로그(area -51%, 0 아님) 근거로 후자로 수렴, Gemini도 동의

### 신뢰 점수
- Gemini: Solver 98 → Critic 95 (자기수정 후 상향, 높은 신뢰)
- OpenAI: Solver 78 → Critic 68 (세부 수치 재계산 과정에서 다소 보수적으로 재평가, 그러나 can_exit=true)

</details>
