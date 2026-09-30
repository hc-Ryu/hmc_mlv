# idea_v17_paratune.md

설계 근거: `/synod idea` 세션 (Claude Validator + Gemini Architect[flash, gemini pro rate-limit → flash 폴백] + OpenAI Explorer[gpt4o])
대상 파일: `uni-section/code/uni_section_v17.py`
목표: `TARGET_MP`가 바뀌어도 (1) 형상/두께 수렴이 안정적이고, (2) 불필요한 부재(candidate parts)가 신뢰성 있게 프루닝되도록 존재 게이트 관련 파라미터를 튜닝.

---

## 핵심 진단 (3개 모델 합의)

현재 v17의 근본 문제는 **`w_sparse_effective`가 절대오차 기준으로 완전히 꺼진다**는 점이다:

```
gate_multiplier = sigmoid(SPARSE_K * (TAU_GATE - mp_rel_err))   # SPARSE_K=50, TAU_GATE=0.05
```

`SPARSE_K=50`은 거의 계단함수이므로, `mp_rel_err`가 5%를 넘는 동안은 희소화/엔트로피 압력이 사실상 0이다. `TARGET_MP`가 바뀌어 오차가 5% 아래로 잘 안 내려가면(더 어려운 목표이거나 스케일이 다르면), 프루닝을 유도하는 압력 자체가 훈련 내내 켜지지 않는다 — 이는 하이퍼파라미터 튜닝 문제가 아니라 **설계상 특정 TARGET_MP 값(27,421,470 N·mm)에 암묵적으로 오버피팅된 메커니즘**이다.

세 모델 모두 이 진단에 동의(conf 85-94%). Critic 라운드에서 원래 제안이었던 "current-error EMA 기반 적응형 TAU_GATE"는 기각되었다: 목표가 실제로 달성 불가능(infeasible)할 경우 오차가 정체되고, EMA가 그 정체값을 따라 올라가 오히려 프루닝 압력을 "잘못된 타이밍"에 켜버리는 피드백 루프가 발생할 수 있음(Gemini-critic conf 94, Claude 최초 지적과 일치).

---

## 채택 결론 (Judge 최종 판정)

### 1. TAU_GATE — "current EMA" 대신 "best-so-far 단조감소 최저오차" 기반 적응형 임계값

```python
# best_so_far_mp_rel_err: 지금까지 관측된 mp_rel_err의 실행 최솟값 (단조 비증가)
tau_gate(t) = min(TAU_CEILING, max(TAU_MIN, EMA(best_so_far_mp_rel_err)))
# TAU_MIN = 0.05 (기존값, 하한)
# TAU_CEILING = 0.10 (상한 — 오차가 정체/발산해도 압력이 완전히 사라지지 않도록)
```

- **왜 EMA(current error)가 아니라 EMA(best-so-far)인가**: current error를 추적하면 오차가 정체되거나 일시적으로 튈 때 임계값이 따라 올라가 "목표를 못 맞췄는데 프루닝 압력이 켜지는" 역설이 생긴다. best-so-far는 단조 비증가이므로, 이 체이싱(chasing) 실패 모드가 수학적으로 봉쇄된다(Defense 라운드에서 Gemini가 이 구분을 명시적으로 방어, conf 95).
- **왜 TAU_CEILING이 필요한가**: 목표가 어려워 오차가 계속 8~9%에 머물러도, best-so-far가 그 근방에서 saturate되면 tau_gate가 상한(0.10)에서 멈춰 프루닝 압력이 완전히 0으로 죽지 않는다. 반대로 목표가 쉬워 오차가 빠르게 1~2%까지 내려가면 tau_gate는 TAU_MIN=0.05로 조여져 기존과 동일하게 엄격해진다.
- **리스크(Prosecution 지적, 수용)**: 초기 급격한 오차 감소가 있으면 best-so-far가 너무 빨리 낮아져 이후 임계값이 지나치게 보수적(conservative)이 될 수 있다는 지적은 타당하나, TAU_CEILING/TAU_MIN 클램프로 그 범위가 [0.05, 0.10]으로 이미 제한되어 있어 "완전히 프루닝이 막히는" 치명적 실패로는 이어지지 않는다. 다만 실측 로그에서 이 범위가 실제로 충분한지 확인 필요(액션 아이템 참고).

### 2. SPARSE_K 완화: 50 → 15~20

계단함수에 가까운 전이를 완만하게 만들어, `gate_multiplier`가 mp_rel_err 변화에 급격히 반응하지 않고 여러 epoch에 걸쳐 서서히 커지도록 한다. Gemini의 원안(12)보다는 다소 높여(15~20) 지나친 무기력화(too-soft, 프루닝 신호 자체가 약해지는 것)를 피한다.

### 3. thresh_low/thresh_high는 **아직 건드리지 않는다** (0.08/0.15 유지)

Gemini 원안은 `thresh_low: 0.08→0.15`를 제안했으나 Critic 라운드(conf 94)에서 기각되었다: HardConcrete는 z→0 근처에서 그래디언트가 소실되므로, threshold를 높이면 실제로 죽지 않은 부재를 조기/과도하게 삭제(flapping)할 위험이 있다. **선제적 변경 대신, 아래 "액션 아이템 A"의 진단 실행으로 0.08이 실제로 도달 가능한지부터 실측 확인**해야 한다.

### 4. `init_log_alpha` 불일치 버그 수정 (3개 모델 전원 독립적으로 발견)

`CGDN.__init__`의 기본값은 `init_log_alpha=2.0`이지만, `run_training`에서 모델을 생성할 때 `init_log_alpha=0.0`으로 명시적으로 덮어쓴다. 콘솔 배너는 `init_log_alpha=2.0`이라고 출력하지만 실제로는 `sigmoid(0)=0.5`(z≈0.475)에서 시작한다 — 로그와 실제 동작이 어긋난다.
→ `run_training` 호출부를 `init_log_alpha=2.0`으로 맞추거나(게이트가 명확히 열린 상태에서 시작해 프루닝 압력에 대한 하강 궤적이 뚜렷해짐), 배너 출력 문자열을 실제 값(0.0)에 맞게 고친다. **둘 중 하나로 일관성부터 맞추는 것이 선행 과제** — 어느 쪽이든 관계없이 currently silent mismatch 상태로 두면 안 됨.

### 5. HardConcrete temperature 어닐링 + gate_params lr 워밍업 동기화

`temperature`를 고정 0.5 대신 `GATE_ACTIVE_EPOCH` 기준 상대epoch로 1.0 → 0.3 어닐링하고, 이를 `gate_params` optimizer lr의 활성화 스케줄과 **같은 상대 epoch 카운터**로 동기화한다(`epoch - GATE_ACTIVE_EPOCH`, 절대 epoch 아님). 온도가 lr보다 먼저 식으면 log_alpha가 학습을 시작하기도 전에 게이트가 결정론적으로 굳어버려(그래디언트 소실) 학습 불가 상태가 될 수 있다는 위험을 방지하기 위함(Gemini Defense 라운드에서 지적, "Why Alternatives Fail" 섹션).

---

## 기각된 대안과 사유

| 제안 | 제안자 | 기각 사유 |
|---|---|---|
| current-error EMA 기반 적응형 TAU_GATE | Gemini(solver) | infeasible target에서 threshold가 정체 오차를 따라 올라가는 체이싱 실패 모드 (Critic conf 94) |
| thresh_low/high를 0.15/0.25로 즉시 상향 | Gemini(solver) | HardConcrete z≈0 근처 그래디언트 소실로 조기/flapping 삭제 위험 (Critic conf 94) |
| Non-monotonic "prune-and-recover" 스케줄 | OpenAI(critic/prosecution) | 한번 억제된 물리 파라미터를 HardConcrete saturation 구간에서 복구하려면 평평한 그래디언트 지대를 가로질러야 해 구조 최적화에서 진동/불안정을 유발 (Defense conf 95, judge 수용) |
| CANDIDATE_PARTS를 target별로 동적 결정(ML 기반) | OpenAI(prosecution) | 방향성은 타당하나 이번 튜닝 범위를 벗어나는 별도 서브시스템 — 과설계. 대신 "액션 아이템 C"로 관찰 대상만 남김 |

---

## 액션 아이템 (구현 순서 권장)

**A. 진단부터 먼저 (실측 없이 threshold 재조정 금지)**
`make_pruning_state` 이후, 특정 candidate part(예: part 4)의 `model.log_alpha.data[4]`를 학습 초반에 인위적으로 `-5.0` 근처로 강제 고정한 짧은 진단 런을 돌려 `ewma_z[4]`가 실제로 0.08 아래까지 내려가는지, 몇 epoch가 걸리는지 로그로 확인한다. 이 실측값을 근거로 thresh_low를 조정할지 결정한다.

**B. TAU_GATE를 best-so-far 기반으로 교체**
`train_step`/`run_training`에 `best_so_far_mp_rel_err` 상태를 추가(예: `alda_state` 또는 별도 dict에 저장), `mp_rel_err` 계산 직후 `min(best_so_far, mp_rel_err)`로 갱신, EMA 후 `TAU_MIN`/`TAU_CEILING`로 clamp하여 `TAU_GATE` 상수 대신 사용.

**C. SPARSE_K를 15~20으로 완화, 결과 로그에서 gate_multiplier 곡선의 실제 형태(계단 vs 완만) 확인**

**D. init_log_alpha 불일치 수정** (배너 출력과 실제 생성자 인자 일치)

**E. temperature 어닐링 + gate_params lr 워밍업을 `epoch - GATE_ACTIVE_EPOCH` 기준으로 동기화 구현**

**F. (관찰만, 이번 범위 밖) CANDIDATE_PARTS=[3,4] 고정이 다른 TARGET_MP에서도 타당한지는 v17 범위 밖 — 향후 다른 TARGET_MP로 재학습 시 Patch1/2 외의 파트가 프루닝 후보가 될 필요가 있는지 로그로 관찰만 해둘 것.**

---

<details>
<summary>숙의 과정 (Synod 세션 상세)</summary>

### 모델 기여
- **Claude (Validator):** best-so-far 단조성 요구, threshold 재조정 전 실측 검증 필요성, init_log_alpha 불일치 최초 발견
- **Gemini (Architect, flash 폴백):** 적응형 게이팅/온도 어닐링/threshold 재조정 최초 제안 및 Critic 라운드에서 스스로의 EMA 제안을 정교화(best-so-far로 방어)
- **OpenAI (Explorer, gpt4o):** infeasible target 엣지케이스, non-monotonic 대안 제시(최종 기각), CANDIDATE_PARTS 고정 여부 문제제기

### 해결된 주요 쟁점
1. "적응형 TAU_GATE가 infeasible target에서 오히려 프루닝을 유발하는가?" → best-so-far(단조) 버전으로 해소, raw EMA는 기각
2. "thresh_low를 지금 올려야 하는가?" → 아니오, 실측 진단 먼저

### 신뢰 점수 (최종 라운드 기준)
- Claude: 88 (Good)
- Gemini: 92→95 (defense 라운드, High)
- OpenAI: 78→85 (prosecution 라운드, Good)

### Trust Score 참고
S(자기중심성) 낮음(모두 반박 인정), C/R/I 양호 → 3개 모델 모두 T ≥ 1.0(Good trust) 이상으로 synthesis에 포함.

</details>

### 신뢰도: 90%
(Gemini의 rate-limit로 pro 대신 flash 모델 사용 — 응답 품질 자체는 XML 형식 준수 및 논리적 방어 모두 정상 수행되어 신뢰도에 실질적 영향 없음. 단, 실제 학습 실행 결과로 검증되지 않은 이론적 설계이므로 액션 아이템 A(실측 진단)를 최우선으로 수행할 것을 권고.)
