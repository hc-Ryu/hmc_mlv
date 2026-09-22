# review_v17_paratune.md

설계 근거: `/synod review` 세션(Claude Validator + Gemini Architect[flash] + OpenAI Explorer[o3, reasoning=medium])
리뷰 대상: `uni-section/code/uni_section_v17_paratune.py`, `uni-section/results/results_v17_paratune/`(재실행 후 실측 로그, `TARGET_MP=27,421,470 N·mm`, 구버전 v17과 동일 타겟)

이 리뷰는 이전 시도에서 노트북이 구버전 코드 그대로였던 문제(사용자가 직접 수정하여 해소)를 확인한 뒤, **실제로 새 코드가 실행된 로그**를 대상으로 재작성되었다. 배너에 `Adaptive TAU_GATE:`, `Temperature annealing:`, `init_log_alpha=2.0`이 실제로 찍혀 있어 이번 로그는 신뢰할 수 있는 근거다.

---

## 실측 비교: 구버전 v17 vs 신버전 v17_paratune (동일 TARGET_MP)

| 항목 | 구버전 v17 (정적 TAU_GATE=0.05, SPARSE_K=50.0, init_log_alpha 버그로 실제 0.0) | 신버전 v17_paratune (적응형 TAU_GATE[0.05~0.10], SPARSE_K=15.0, temperature 1.0→0.3 어닐링, init_log_alpha=2.0 수정) |
|---|---|---|
| Part 4 DELETED epoch | 326 | 681 |
| Part 3 DELETED epoch | 408 | 1039 |
| 최종(epoch 1500) Mp err | 0.10% | 0.57% |
| 최종 l_collision | 0.0055 | 0.0039 |
| 최종 면적 변화 | +5.3% | +3.5% |
| Best-feasible epoch | 1238 (Mp err 0.00%, l_collision 0.0066) | 950 (Mp err 0.00%, l_collision 0.0034) |
| 최종 프루닝 상태 | [3,4] 모두 DELETED | [3,4] 모두 DELETED |

두 런 모두 프루닝 성공. 다만 신버전에서 프루닝 트리거가 2~2.5배 느려졌고(326→681, 408→1039), 초반 epoch(19~239)의 `l_sparse`/총 loss가 눈에 띄게 더 노이지함(`l_sparse`가 39/59/119/139/179/199/219/459 epoch에서 반복적으로 정확히 1.0000 — 완전 개방 상태 — 을 찍음).

---

## 핵심 판정: 프루닝 지연의 원인은 적응형 TAU_GATE가 아니라 init_log_alpha 수정이다

Solver 라운드에서 Gemini와 OpenAI가 상반된 해석을 내놓았고, 실제 수치 계산으로 검증한 결과 **OpenAI의 정량적 분석이 맞다**고 판정한다.

**근거(직접 계산)**:
- 구버전: `init_log_alpha=0.0`(버그) → `z_gate = sigmoid(0.0) = 0.50`에서 시작. `ewma_z < 0.08`(삭제 임계값)까지 떨어져야 하는 거리 = 약 0.42.
- 신버전: `init_log_alpha=2.0`(수정됨) → `z_gate = sigmoid(2.0) ≈ 0.88`에서 시작. 같은 임계값까지 떨어져야 하는 거리 = 약 0.80 — **구버전의 거의 2배**.
- 게다가 `current_tau_gate`는 신버전에서도 epoch 59만에 `TAU_MIN=0.05`(구버전과 완전히 동일한 값)로 수렴해 **학습의 96% 이상(epoch 59~1500) 동안 두 런이 사실상 동일한 TAU_GATE 값으로 돌아갔다.** 즉 이 특정 타겟에서는 적응형 메커니즘이 "작동은 했지만 값 자체는 정적 버전과 같아졌다."

따라서 **프루닝이 2~2.5배 느려진 것은 적응형 TAU_GATE 로직 때문이 아니라, 게이트가 훨씬 더 열린 상태(0.88)에서 출발해 떨어질 거리가 거의 2배로 늘어난, `init_log_alpha` 버그 수정의 직접적이고 거의 전적인 결과**로 보는 것이 가장 설득력 있는 설명이다. Gemini는 이를 "SPARSE_K 완화 + init_log_alpha가 함께 만든 건강한 트레이드오프"로 해석했으나, 두 변수를 분리하지 않은 상태에서의 서술적 해석이며, 실제 시작 위치 산술 계산과 TAU_GATE가 곧바로 하한에 고정된다는 사실을 고려하면 근거가 약하다.

**중요한 함의**: 이는 결코 나쁜 소식이 아니다 — `init_log_alpha` 수정은 애초에 "배너와 실제 동작을 일치시키는" 별개의 버그 수정이었고, 신버전의 프루닝 지연은 적응형 TAU_GATE 설계의 결함이 아니라 이 버그 수정 자체의 부수 효과일 뿐이다. 다만 **"적응형 TAU_GATE가 프루닝을 더 신뢰성 있게 만들었다"는 주장은 이번 실측으로 증명되지 않았다** — 이번 런에서 적응형 로직은 임계값이 즉시 하한에 고정되어 사실상 정적 로직과 동일하게 작동했기 때문이다.

## 초반 loss/l_sparse 노이즈의 원인: SPARSE_K보다 temperature 어닐링일 가능성이 더 크다

- 신버전은 `temperature`를 `1.0`에서 시작해 30epoch에 걸쳐 `0.3`으로 어닐링한다. 구버전은 `temperature=0.5` 고정이었다.
- HardConcrete(Binary Concrete) 분포에서 `temperature`가 높을수록 시그모이드 완화가 평평해져 `z_open` 샘플링(`torch.rand_like` 기반 stochastic 항)의 분산이 커진다 — 즉 초반 온도가 구버전보다 2배 높은 것이 초반 `l_sparse`/`loss`의 노이즈를 설명하는 더 직접적인 메커니즘이다.
- `SPARSE_K=15.0`(완화)은 페널티의 "기울기"에는 영향을 주지만 확률적 분산 자체를 늘리지는 않는다 — `l_sparse`가 정확히 `1.0000`을 반복적으로 찍는 패턴은 완화된 시그모이드보다는 고온 샘플링에서 흔히 보이는 클램프 현상에 더 부합한다.
- Gemini가 이 노이즈를 "구조적으로 더 우수한 최적해를 찾기 위한 건강한 탐색"이라고 해석하고 이를 최종 `l_collision`(0.0039 < 0.0055) 개선의 근거로 든 것은 **단일 시드, 단일 런에서 나온 상관관계를 인과관계로 해석한 것**이라 근거가 약하다 — 이 판정에서는 채택하지 않는다.

## 이 비교 자체의 한계 (반드시 인지할 것)

1. **최종epoch 비교보다 best-feasible 체크포인트 비교가 맞다**는 Gemini의 지적은 타당하고 이견 없이 채택한다 — 두 런 모두 `best_feasible`/`best_mp` 체크포인트 로직을 쓰므로, 마지막 epoch(1500)의 값은 "우연히 그 시점에 어떤 상태였는가"일 뿐 최적 성능의 대푯값이 아니다. best-feasible 기준으로는 신버전(epoch 950, l_collision 0.0034)이 구버전(epoch 1238, l_collision 0.0066)보다 collision 지표가 낫다 — 그러나 이 역시 단일 런 비교이므로 "신버전이 더 낫다"고 단정할 근거로 쓰기엔 이르다.
2. **교란변수(confounding) 문제**: 이번 비교는 `init_log_alpha`, `SPARSE_K`, 정적→적응형 `TAU_GATE`, `temperature` 스케줄까지 네 가지를 동시에 바꾼 상태에서 관찰한 것이라, 관찰된 차이를 특정 변경 하나의 효과로 귀속시킬 수 없다. `TAU_GATE`/`SPARSE_K`/`temperature` 각각의 개별 효과를 분리하려면 하나씩만 바꾼 ablation 런이 필요하다(예: `init_log_alpha`는 0.0으로 고정한 채 적응형 TAU_GATE만 켜본 런, 또는 반대로).
3. **여전히 "쉬운" 타겟만 테스트됨**: `idea_v17_paratune.md`의 원래 문제 제기(`mp_rel_err`가 5% 위에서 정체될 때 프루닝 압력이 꺼진다)는 이번 타겟에서 재현되지 않는다(구버전·신버전 모두 epoch 100 이내 오차가 5% 아래로 내려감). 적응형 `TAU_GATE`의 상한(`TAU_CEILING=0.10`)이 실제로 어떻게 작동하는지는 여전히 검증되지 않았다 — 이번 런에서 `current_tau_gate`는 epoch 59 이후 하한(0.05)에 고정된 채 끝까지 움직이지 않았다.
4. **시드 1개뿐**: HardConcrete는 매 forward마다 `torch.rand_like`로 확률적 샘플링을 하므로, 시드 하나만으로 관찰된 차이(epoch 326 vs 681 등)가 진짜 체계적 효과인지 우연한 변동인지 구분할 수 없다.

---

## 다음 실험 설계 (우선순위)

1. **[최우선] Ablation**: `init_log_alpha`를 0.0으로 고정한 채 적응형 `TAU_GATE`만 켜서 실행 → 프루닝 지연이 여전히 발생하는지 확인. 반대로 `init_log_alpha=2.0`만 적용하고 `TAU_GATE`는 정적 0.05로 유지한 런도 실행 → 이번 관찰된 지연(326→681, 408→1039)이 실제로 `init_log_alpha` 하나로 재현되는지 확인.
2. **[필수] 하드 타겟 테스트**: `mp_rel_err`가 5% 근방에서 정체되도록 의도적으로 어려운 `TARGET_MP`로 한 번 더 실행 — 이것이 `idea_v17_paratune.md`가 원래 풀려던 문제이며, 적응형 `TAU_CEILING`이 실제로 작동하는 유일한 시나리오다. 이번 결과(쉬운 타겟)만으로는 paratune의 실효성을 판단할 수 없다.
3. **[권고] 다중 시드**: 최소 3~5개 시드로 반복 실행해, 이번에 관찰된 프루닝 epoch 차이·collision 개선이 시드 변동 범위 안에 있는지 확인.
4. **[관찰] temperature 스케줄 재검토**: 초반 노이즈가 temperature=1.0에서 기인한다는 가설이 맞다면, `TEMP_INIT`를 약간 낮추거나(예: 0.7) `TEMP_WARMUP_EPOCHS`를 줄여 초반 샘플링 분산을 완화하는 것도 고려할 수 있다 — 단, 이는 다음 라운드의 별도 튜닝 논의로 미룬다.

---

## `.py` 코드 자체에 대한 이전 리뷰 결과 (유효, 재확인 불필요)

- `model.log_alpha.data[idx] = -5.0` 직접 뮤테이션은 v17의 기존 프루닝 트리거 코드(`model.log_alpha.data[pid] = -10.0`)와 동일한 관례이므로 새로운 리스크가 아님.
- `pruning_state`의 in-place mutation 패턴은 현재 스케일(단일 스크립트, `torch.compile` 미사용)에서는 무방하나, 향후 확장 시 명시적 반환값 패턴 고려 권고.
- `argparse.parse_known_args()`는 Jupyter 환경의 `--f=...` 인자를 무시하기 위한 의도된 선택이나, 사용자의 CLI 플래그 오타(`--diagnotic` 등)도 조용히 무시한다는 트레이드오프가 있음.

---

<details>
<summary>숙의 과정 (Synod 세션 상세)</summary>

### 모델 기여
- **Claude (Judge/Validator):** OpenAI의 `sigmoid(0.0)=0.5` vs `sigmoid(2.0)≈0.88` 산술 근거가 Gemini의 서술적 해석보다 우월하다고 판정, "best-feasible 체크포인트가 맞는 비교 기준"이라는 Gemini의 지적은 그대로 채택
- **Gemini (Architect, flash):** 초기 "SPARSE_K+init_log_alpha 복합 효과로 인한 건강한 트레이드오프"라는 서술적 해석 제시(단, 인과관계 입증 없이 상관관계로 결론 내린 부분은 기각됨). Final-epoch 대신 best-feasible 체크포인트를 비교 기준으로 삼아야 한다는 지적은 채택
- **OpenAI (Explorer, o3):** 시작 z_gate 값의 산술 계산으로 프루닝 지연을 `init_log_alpha` 수정에 귀속시키는 정량적 반박 제시, temperature 어닐링이 노이즈의 더 그럴듯한 원인이라는 메커니즘 제시, 교란변수·단일시드 문제를 명확히 지적

### 해결된 주요 쟁점
1. "프루닝 지연이 적응형 TAU_GATE의 결과인가?" → 아니오, `init_log_alpha` 수정(게이트 시작점이 거의 2배 더 열림)이 주된 원인으로 판정
2. "초반 노이즈가 SPARSE_K 완화 때문인가?" → temperature 어닐링(1.0→0.3)이 더 그럴듯한 원인
3. "이번 런으로 적응형 TAU_GATE의 실효성을 확인했는가?" → 아니오, `current_tau_gate`가 epoch 59부터 하한에 고정되어 사실상 정적 로직과 동일하게 작동함

### 신뢰 점수 (최종 라운드 기준)
- Claude: 90
- Gemini: 95 (일부 결론은 Judge가 기각)
- OpenAI: 93

</details>

### 신뢰도: 90%
(이번 리뷰는 실제로 실행된 신버전 코드의 실측 로그를 대상으로 하므로 이전 리뷰보다 신뢰도가 높으나, 단일 시드·단일 타겟 비교라는 근본적 한계가 있어 "적응형 TAU_GATE 자체의 실효성"은 여전히 미검증 상태다. Ablation과 하드 타겟 테스트 전까지는 최종 결론을 유보할 것.)
