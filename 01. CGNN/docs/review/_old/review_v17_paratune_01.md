# review_v17_paratune_01.md

리뷰 대상: `uni-section/code/uni_section_v17_paratune_01.py`의 `paratune_sweep` 실행 결과
`uni-section/results/results_v17_paratunes_01/paratune_sweep_summary.md` (9개 런: anchor 1개 + ablation 4개 + `TARGET_MP` 스케일 4개)

전제: 이 스윕은 이전 리뷰(`review_v17_paratune.md`)가 지적한 "네 가지 변경(`init_log_alpha`/`SPARSE_K`/적응형 `TAU_GATE`/temperature 어닐링)이 동시에 바뀐 상태라 개별 효과를 분리할 수 없다"는 한계를 해소하기 위해 설계된 ablation이다. `anchor`를 기준으로 한 번에 하나씩만 되돌린 4개 런(`abl_alpha0`, `abl_sparseK50`, `abl_tauStatic`, `abl_tempStatic`)과, 동일 설정에서 `TARGET_MP`만 0.5x~1.6x로 바꾼 4개 런(`mp_down_0.7x/0.5x`, `mp_up_1.3x/1.6x`)으로 구성된다.

---

## 1. Ablation 결과: 이전 리뷰의 핵심 판정("프루닝 지연의 원인은 init_log_alpha")이 뒤집힌다

| 런 | part4 DELETED epoch | part3 DELETED epoch | best_feasible epoch (l_collision) | 최종 mp_err | 최종 l_collision |
|---|---|---|---|---|---|
| anchor (전체 신규 설정) | 685 | 1013 | 260 (0.00176) | 0.32% | 0.0035 |
| abl_alpha0 (init_log_alpha=0.0으로 되돌림) | 348 | 479 | 1094 (0.00491) | 0.30% | 0.0042 |
| abl_sparseK50 (SPARSE_K=50.0로 되돌림) | 656 | 891 | 385 (0.00269) | 0.25% | 0.0036 |
| abl_tauStatic (TAU_GATE 정적 0.05로 되돌림) | 709 | 1060 | 935 (0.00284) | 0.31% | 0.0038 |
| abl_tempStatic (temperature 고정 0.5로 되돌림) | 660 | 840 | 988 (0.00522) | 0.29% | 0.0042 |

**이전 리뷰의 판정과 정반대 결과다.** 이전 리뷰는 산술 계산(`sigmoid(0.0)=0.50` vs `sigmoid(2.0)≈0.88`)을 근거로 "프루닝이 2~2.5배 느려진 것은 거의 전적으로 `init_log_alpha` 수정 때문"이라고 결론 내렸다. 그런데 이번 ablation에서 `init_log_alpha`만 0.0으로 되돌린 `abl_alpha0`는 오히려 **가장 빠르게** 프루닝이 발동했다(part4 348, part3 479 — anchor의 685/1013보다 거의 2배 빠름). 나머지 세 ablation(`abl_sparseK50`, `abl_tauStatic`, `abl_tempStatic`)은 anchor와 큰 차이가 없다(656~709 / 840~1060 범위).

즉 **anchor가 다른 4개 ablation보다 뚜렷하게 느린 이유는 `init_log_alpha=2.0` 그 자체가 아니라, `init_log_alpha=2.0`이 다른 세 변경(SPARSE_K=15.0, 적응형 TAU_GATE, temperature 어닐링)과 "함께" 있을 때만 나타나는 상호작용(interaction) 효과일 가능성이 높다.** `abl_alpha0`에서 `init_log_alpha`를 0.0으로 낮추자 다른 세 변경이 그대로인데도 오히려 가장 빠른 프루닝이 나왔다는 것은, 산술적으로 게이트 시작점(0.88 vs 0.50)이 멀다는 이전 근거만으로는 실제 동역학을 설명할 수 없다는 뜻이다. 단일 변수의 "시작점 거리"가 아니라 SPARSE_K·TAU_GATE·temperature와 얽힌 비선형 상호작용이 지배적인 것으로 보인다.

이전 리뷰가 명시적으로 요청했던 정확히 이 실험("`init_log_alpha`를 0.0으로 고정한 채 적응형 TAU_GATE만 켜서 실행")이 `abl_alpha0`에 해당하며, 그 결과는 이전 리뷰의 결론을 지지하지 않는다. **이전 리뷰의 "init_log_alpha가 프루닝 지연의 거의 전적인 원인"이라는 판정은 이번 실측으로 반박되었다고 봐야 한다.**

### best-feasible 체크포인트 기준으로도 같은 그림

`best_feasible_epoch`/`l_collision` 기준으로 봐도 `abl_alpha0`(epoch 1094, l_collision 0.00491)가 anchor(epoch 260, l_collision 0.00176)보다 훨씬 늦게, 더 나쁜 collision 값으로 best-feasible에 도달한다. anchor가 가장 이른 epoch(260)에 가장 낮은 l_collision(0.00176)을 기록한 것은 프루닝이 늦게 트리거되는 것과 별개로, 학습 초반 collision 최적화 자체는 anchor 설정이 4개 ablation 중 가장 우수함을 시사한다.

---

## 2. 어느 단일 변경도 anchor를 단독으로 설명하지 못한다 — 상호작용 가설

네 ablation 각각의 "부분 되돌리기"가 프루닝 타이밍에 준 영향을 정리하면:

- `abl_alpha0`: anchor 대비 **크게 빨라짐** (685→348, 1013→479)
- `abl_sparseK50`: anchor 대비 소폭 빨라짐 (685→656, 1013→891)
- `abl_tauStatic`: anchor와 거의 동일, 오히려 소폭 더 느림 (685→709, 1013→1060)
- `abl_tempStatic`: anchor 대비 소폭 빨라짐 (685→660, 1013→840)

네 값을 단순 합산하면 anchor(685)가 abl 4개보다 일관되게 가장 느리다는 것은, "가장 관대한(가장 늦게 프루닝을 유발하는) 설정 조합"이 정확히 anchor(모든 신규 요소 ON)라는 뜻이다. 이는 각 요소가 독립적으로 프루닝을 늦추는 방향으로 조금씩 기여하되, 그 효과가 산술적으로 단순 가산되지 않고 서로 얽혀 있음을 시사한다 — 특히 `abl_alpha0`만 유독 큰 폭으로 달라진다는 것은 `init_log_alpha`가 나머지 세 요소(특히 적응형 TAU_GATE의 EMA 갱신, SPARSE_K 기울기)와 상호작용할 때 게이트 하강 궤적이 질적으로 달라짐을 의미한다. **이 상호작용의 정확한 메커니즘은 이번 요약 테이블(최종/best-feasible 값만 있음)만으로는 규명할 수 없다** — `ewma_z`/`current_tau_gate`/`temperature`의 epoch별 궤적 로그가 필요하다.

---

## 3. TARGET_MP 스케일 스윕: 원래 문제 제기(하드 타겟에서 프루닝 압력 소실)가 부분적으로 재현됨

| 런 | target_mp | 최종 mp_err | 최종 l_collision | part4/part3 DELETED epoch |
|---|---|---|---|---|
| mp_down_0.7x | 19.2M | 0.20% | 0.0000 | 592 / 1235 |
| mp_down_0.5x | 13.7M | 0.34% | 0.0000 | 591 / 999 |
| anchor (1.0x) | 27.4M | 0.32% | 0.0035 | 685 / 1013 |
| mp_up_1.3x | 35.6M | **1.38%** | 0.0184 | 676 / **None** |
| mp_up_1.6x | 43.9M | 0.70% | 0.0220 | **None / None** |

- `mp_up_1.6x`는 **part3/part4 모두 DELETED에 도달하지 못했다**(`None`). `mp_up_1.3x`도 part3가 `None`이다. 즉 타겟이 어려워질수록(`mp_scale` 상승) 프루닝이 아예 발동하지 않는 경우가 나타난다 — 이것이 바로 `idea_v17_paratune.md`가 원래 문제 제기했던 "mp_rel_err가 정체될 때 프루닝 압력이 꺼진다"는 시나리오이며, 이번 스윕에서 처음으로 재현되었다.
- 다만 `mp_up_1.3x`의 최종 `mp_rel_err`는 1.38%로 다른 런들(0.2~0.7%대)보다 확연히 높다 — 즉 "정체"라기보다 "아직 수렴 안 됨/발산 근처"에 가까운 상태일 수 있다. `TAU_CEILING=0.10`이 실제로 이 상황에서 압력을 유지하는 데 유효했는지, 아니면 애초에 목표가 1500 epoch 안에 못 풀 만큼 어려운 것인지 이 요약만으로는 구분되지 않는다.
- `mp_up_1.6x`는 최종 mp_err가 0.70%로 오히려 `mp_up_1.3x`(1.38%)보다 낮다 — 타겟이 더 어려운데 최종 오차가 더 작다는 것은 단조적이지 않은 결과이며, 두 런 사이에 우연/시드 변동 또는 curriculum-타겟 상호작용이 있을 가능성을 시사한다. 이 지점은 반드시 추가 확인이 필요하다.
- `mp_down_0.5x`/`0.7x`는 둘 다 `l_collision=0.0000`으로 collision 제약이 사실상 완전히 해소되었다 — 타겟이 쉬워질수록(면적이 작아질수록) collision 여유가 커진다는 점에서 물리적으로 합리적이다.

**결론**: 하드 타겟(`mp_up_1.3x`, `1.6x`)에서 프루닝이 실제로 지연/미발동하는 현상이 확인되었다는 점은 적응형 TAU_GATE 도입의 원래 동기(`idea_v17_paratune.md`)가 여전히 유효한 문제였음을 보여준다. 그러나 이번 스윕은 "적응형 TAU_GATE가 이 문제를 해결했는가"를 검증하는 대조군(같은 하드 타겟에서 정적 TAU_GATE=0.05만 쓴 런)이 빠져 있어, **적응형 메커니즘이 실제로 이 상황에서 정적 버전보다 낫다는 것은 여전히 증명되지 않았다.**

---

## 4. 이 결과 자체의 한계

1. **단일 시드**: 9개 런 모두 시드가 명시되지 않았고(스윕 스크립트 확인 결과 시드 고정/변경 로직 없음), HardConcrete의 확률적 샘플링을 고려하면 여기서 관찰된 epoch 차이(예: `abl_alpha0`의 348 vs anchor의 685)가 시드 변동 범위를 벗어난 진짜 효과인지 단정할 수 없다. 특히 상호작용 가설(2번 항목)은 시드 하나로 도출된 것이라 반복 검증이 필요하다.
2. **하드 타겟 스윕에 정적 TAU_GATE 대조군 부재**: `mp_up_1.3x`/`1.6x`가 진짜 "적응형 TAU_GATE 덕분에 그나마 이 정도"인지, "적응형이든 정적이든 이 타겟에서는 원래 이렇게 나오는지" 구분할 방법이 없다.
3. **요약 테이블은 최종/best-feasible 스냅샷만 제공**: `ewma_z`, `current_tau_gate`, `temperature`의 epoch별 궤적이 없어 2번 항목의 상호작용 메커니즘을 직접 검증할 수 없고, 서술적 가설(interaction) 수준에 머무른다.
4. **`mp_up_1.3x`와 `1.6x` 사이의 비단조성**(오차 1.38% vs 0.70%)은 이 표만으로는 원인을 알 수 없다 — curriculum 스케줄(`GATE_ACTIVE_EPOCH` 등)이 `TARGET_MP` 절대값에 의존하는 부분이 있는지 코드 확인이 필요하다.

---

## 다음 실험 설계 (우선순위)

1. **[최우선] 상호작용 분리**: `init_log_alpha=0.0` + `SPARSE_K=15.0` + 적응형 TAU_GATE + temperature 어닐링을 모두 켠 조합(`abl_alpha0`는 이미 이 조합) 외에, 2-factor 조합(예: `init_log_alpha=0.0` & `SPARSE_K=50.0`만, 나머지는 anchor와 동일)을 몇 개 더 실행해 `init_log_alpha`가 정확히 무엇과 상호작용해 anchor를 느리게 만드는지 좁혀야 한다.
2. **[필수] 하드 타겟 대조군**: `mp_up_1.3x`/`1.6x`와 동일한 `TARGET_MP`에서 `TAU_GATE`를 정적 0.05로 고정한 런을 추가해, 적응형 TAU_GATE가 실제로 하드 타겟에서 정적 버전보다 프루닝을 더 신뢰성 있게 발동시키는지 직접 비교해야 한다.
3. **[권고] 다중 시드**: 최소 anchor·abl_alpha0·mp_up_1.3x 세 런만이라도 시드 3~5개로 반복해, 이번에 관찰된 epoch 차이가 시드 변동 폭 안에 있는지 확인.
4. **[관찰] epoch별 궤적 로깅**: `ewma_z`/`current_tau_gate`/`temperature`의 epoch별 값을 별도 CSV로 남기면, 상호작용 가설을 요약 테이블이 아닌 실제 궤적으로 검증할 수 있다.

---

### 신뢰도: 70%
(이번 분석은 `paratune_sweep_summary.md`의 최종/best-feasible 스냅샷 수치만을 근거로 하며, 개별 CLI 모델 교차검증 없이 직접 수행했다. ablation 간 비교의 방향성[abl_alpha0가 가장 빠르다는 것]은 표에 나온 수치 자체이므로 확실하나, 그 원인을 "상호작용"으로 귀속한 부분은 단일 시드·궤적 로그 부재 상태의 가설이라 확정적이지 않다.)
