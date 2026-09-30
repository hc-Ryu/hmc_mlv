# command_v17_paratune.md

설계 근거: `docs/idea/idea_v17_paratune.md`(/synod idea 세션, conf 90) → 본 `/synod idea` 세션(Claude Validator + Gemini Architect[flash 폴백] + OpenAI Explorer[gpt4o], 3라운드: Solver→Critic→Defense/Prosecution, conf 95/85 → judge 최종 91)에서 idea 문서를 구현 가능한 순서로 재정렬.
대상 파일: `uni-section/code/uni_section_v17.py`

이 문서는 코딩 에이전트가 그대로 따라 실행할 수 있는 **순서가 있는** 구현 지시서다. 각 Phase는 이전 Phase가 완료·검증된 후에만 시작한다 — 순서를 바꾸면 안 되는 이유가 각 Phase 서두에 명시되어 있다.

---

## 순서가 중요한 이유 (Judge 최종 판정 요지)

Critic/Defense 라운드에서 Gemini 원안의 5-phase 순서(diagnostic hook을 마지막 Phase 5에 배치)가 기각되고, **diagnostic hook을 Phase 2로 앞당기는 안**이 채택되었다:
- diagnostic run은 "새로운 temperature/adaptive-threshold 로직이 이미 다 들어간 상태"를 테스트하면 안 된다 — 여러 변수가 동시에 바뀐 상태에서 문제가 생기면 원인 특정이 불가능해진다(diagnostic의 존재 이유가 무력화됨).
- 반대로 diagnostic이 "완성된 시스템 반영이 안 돼 결과가 무의미하다"는 Prosecution의 반박도 일리가 있어, **Judge는 절충안으로 diagnostic을 두 번 실행**하기로 한다: Phase 2(격리된 baseline 확인용, 필수 gate)와 Phase 6(전체 변경 완료 후 최종 회귀 확인용, 저비용 재확인).
- `best_so_far_mp_rel_err`는 State-Only Reset 직후 `PRUNING_COOLDOWN_EPOCHS` 동안 **하드 프리즈**한다 — Prosecution이 제안한 "높은 tolerance" 방식은 새로운 미조정 하이퍼파라미터(허용폭이 얼마인지, 감쇠 방식 등)를 추가로 도입해 상태 기계를 복잡하게 만들 뿐이고, State-Only Reset 직후 구간은 원래 손실/오차가 격렬하게 요동치는 구간이라 어떤 tolerance를 걸어도 신뢰할 수 없다(Defense conf 95, judge 수용).

---

## Phase 0 — `init_log_alpha` 일관성 수정 (선행 필수)

**왜 가장 먼저인가**: 이후 Phase 2의 diagnostic run이 관측하는 `ewma_z` 궤적의 시작점(z_gate at epoch 0)이 이 값에 의존한다. 나중에 고치면 diagnostic을 다시 돌려야 한다.

1. `CGDN.__init__`의 `init_log_alpha` 기본값 `2.0`은 그대로 둔다.
2. `run_training()` 내 모델 생성 호출부(`init_log_alpha=0.0`으로 덮어쓰는 부분, 약 1103행 부근)를 `init_log_alpha=2.0`으로 변경 — 콘솔 배너(`print(f"... init_log_alpha=2.0")`, 약 1185행)와 실제 동작을 일치시킨다.
3. **검증**: 학습 스크립트를 1 epoch만 돌려(`max_epochs=1`) 콘솔에 찍히는 `init_log_alpha=2.0`과 실제 `model.log_alpha` 초기값(`sigmoid(2.0)≈0.88` 근방)이 일치하는지 확인. 기존 command_v17.md 동작(프루닝 미발동 로그 패턴 등)이 이 변경만으로 크게 달라지지 않는지 짧게 확인.

---

## Phase 1 — 상수 추가 및 `pruning_state` 딕셔너리 확장

**왜 Phase 0 다음인가**: 이후 모든 Phase가 이 상태 필드/상수에 의존하므로 먼저 뼈대를 만든다.

### 1.1 모듈 상수 (654~666행 근처 블록에 추가)
```python
# 기존 값 변경
SPARSE_K = 15.0          # 기존 50.0 → 완화 (계단함수에 가까운 전이를 완만하게)

# 신규 추가 — 적응형 TAU_GATE
TAU_MIN = 0.05            # 기존 TAU_GATE와 동일한 하한(가장 엄격한 목표 달성 시)
TAU_CEILING = 0.10        # 상한(목표가 어려워 오차가 정체돼도 압력이 완전히 죽지 않도록)
TAU_EMA_ALPHA = 0.15      # best-so-far EMA 평활 계수 (ewma_alpha와 동일 스케일 사용)

# 신규 추가 — HardConcrete 온도 어닐링
TEMP_INIT = 1.0
TEMP_MIN = 0.3
TEMP_WARMUP_EPOCHS = 30   # GATE_ACTIVE_EPOCH 기준 상대epoch로 이 기간 동안 TEMP_INIT→TEMP_MIN 선형 감쇠
```
`TAU_GATE = 0.05` 상수 자체는 코드에서 제거하지 말고 유지 — Phase 4까지는 fallback/구버전 대조용으로 남겨둔다(Phase 4에서 실제 사용처만 `pruning_state['current_tau_gate']`로 교체).

### 1.2 `make_pruning_state()` 확장 (약 230행)
```python
def make_pruning_state(num_parts=5, candidate_parts=None):
    ...
    state = {
        'ewma_z': torch.ones(num_parts),
        'state': ['ALIVE'] * num_parts,
        'pending_duration': torch.zeros(num_parts),
        'ewma_alpha': 0.15,
        'thresh_low': 0.08,     # Phase 2 진단 없이는 변경 금지
        'thresh_high': 0.15,    # 상동
        'confirm_epochs': 20,
        's_protect': S_PROTECT,
        'candidate_parts': candidate_parts,
        'cooldown_remaining': 0,
        # ── 신규: 적응형 TAU_GATE 상태 ──
        'best_so_far_mp_rel_err': float('inf'),
        'ema_best_so_far': TAU_CEILING,   # 초기값은 상한에서 시작(가장 관대한 상태)
        'current_tau_gate': TAU_CEILING,
    }
    return state
```

### 1.3 검증
`make_pruning_state()`를 단독 호출해 새 키들이 기대한 초기값으로 채워지는지 단위 확인. 기존 키(`thresh_low` 등)가 실수로 변경되지 않았는지 diff로 확인.

---

## Phase 2 — 진단 훅 구현 및 1차 실행 (thresh_low/high 변경 이전 필수 게이트)

**왜 지금인가(Phase 3/4보다 먼저)**: temperature 어닐링·adaptive TAU_GATE 로직이 아직 없는 "가장 단순한" 상태에서 `ewma_z` 하강 궤적의 순수한 baseline을 확보한다. 여러 변수를 한꺼번에 바꾼 뒤 문제가 생기면 원인 특정이 사실상 불가능해진다(Defense 라운드에서 확정, conf 95).

### 2.1 CLI 플래그 추가 (`__main__` 블록 또는 별도 argparse 도입)
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--diagnostic', action='store_true',
                     help='candidate part의 log_alpha를 강제로 낮춰 ewma_z 하강 궤적을 관찰하는 진단 모드')
parser.add_argument('--diagnostic-part-id', type=int, default=4,
                     help='진단 대상 part_id (CANDIDATE_PARTS 중 하나, 기본 4=Patch2)')
args = parser.parse_args()
```
(v17이 현재 `argparse` 없이 하드코딩 실행 구조이므로, 기존 `if __name__ == "__main__":` 블록의 최상단에 위 파서를 추가하고 `run_training(...)`에 `diagnostic=args.diagnostic, diagnostic_part_id=args.diagnostic_part_id`를 전달하도록 시그니처를 확장한다.)

### 2.2 `run_training()`에 진단 강제 로직 삽입
`epoch == GATE_ACTIVE_EPOCH` 분기(`gate_stage_done` 처리, 약 1208행) 바로 다음에 추가:
```python
if diagnostic and epoch == GATE_ACTIVE_EPOCH:
    with torch.no_grad():
        idx = CANDIDATE_PARTS.index(diagnostic_part_id)  # CANDIDATE_PARTS 내 위치 확인 후 사용
        model.log_alpha.data[diagnostic_part_id] = -5.0
    print(f"[DIAGNOSTIC] epoch {epoch}: part {diagnostic_part_id} log_alpha 강제 -5.0 "
          f"(ewma_z 하강 궤적 관찰 시작)")
```
`update_pruning_state` 호출 부근(프루닝 판정 로직) 근처에 다음 한 줄을 추가해 매 20epoch 로그에 진단 대상 파트의 `ewma_z`를 별도로 출력:
```python
if diagnostic and (epoch + 1) % 20 == 0:
    print(f"  [DIAGNOSTIC] part {diagnostic_part_id} ewma_z = {pruning_state['ewma_z'][diagnostic_part_id]:.4f} "
          f"(thresh_low={pruning_state['thresh_low']})")
```

### 2.3 1차 진단 실행 및 결과 기록 (필수, 다음 Phase 진행 전 완료)
```bash
python uni_section_v17.py --diagnostic --diagnostic-part-id 4
```
- `ewma_z[4]`가 `thresh_low=0.08` 아래로 실제 내려가는지, 몇 epoch가 걸리는지 로그에서 확인.
- **내려가지 않거나 지나치게 오래 걸린다면**: `thresh_low`를 올리는 근거가 되므로 이 실측 로그를 근거로만 Phase 2.5(선택)에서 `thresh_low`/`thresh_high` 조정을 검토한다. **로그 없이 임의로 0.08/0.15를 바꾸지 않는다** (idea 문서 결론 그대로 유지).
- 이 실행 결과를 `docs/hmc/idea/`(또는 결과 로그 저장 위치)에 텍스트로 남겨 Phase 6 최종 회귀 비교의 기준선으로 삼는다.

---

## Phase 3 — Temperature 배관(plumbing) 및 어닐링

**왜 지금인가**: Critic/Prosecution 라운드에서 3개 모델 전원이 지적한 핵심 갭 — 현재 `compute_gates(temperature=0.5)`는 고정 기본값이고 어떤 호출부도 이를 실제로 오버라이드하지 않는다. 이 배관을 먼저 만들지 않으면 이후 "temperature 1.0→0.3 어닐링"은 죽은 코드가 된다.

### 3.1 시그니처 확장
- `compute_gates(log_alpha, training=True, temperature=0.5, ...)` → 그대로 유지(기본값은 호출부에서 항상 명시적으로 넘기도록 강제할 것이므로 기본값 존재 자체는 무해).
- `CGDN.forward(...)`의 인자에 `temperature=0.5`를 추가하고, 내부에서 `compute_gates(self.log_alpha, training=self.training, temperature=temperature, s_protect=self.s_protect)`로 전달(현재는 `temperature`를 아예 넘기지 않는 상태 — 이 한 줄이 누락된 배관의 핵심).

### 3.2 `train_step()`에서 온도 계산 및 전달
`gate_active = epoch >= GATE_ACTIVE_EPOCH` 계산 직후에 추가:
```python
rel_epoch = epoch - GATE_ACTIVE_EPOCH
warmup = max(1, TEMP_WARMUP_EPOCHS)   # 0-division 가드 (Critic 라운드 conf 92 지적 반영)
if rel_epoch < 0:
    current_temp = TEMP_INIT
else:
    progress = min(1.0, rel_epoch / warmup)
    current_temp = TEMP_INIT + (TEMP_MIN - TEMP_INIT) * progress   # 선형 감쇠
```
`model(...)` 호출부에 `temperature=current_temp` 인자를 추가 전달. `train_step` 반환 dict에도 `"temperature": current_temp`를 추가해 `run_training`의 `history['temperature']`에 누적한다(신규 history 키 — 4.3 참고).

### 3.3 검증
`--diagnostic` 없이 짧은 런(예: `max_epochs=50`)을 돌려 `GATE_ACTIVE_EPOCH` 이전엔 `temperature==TEMP_INIT`, 이후 `TEMP_WARMUP_EPOCHS`에 걸쳐 `TEMP_MIN`까지 선형 감소하는지 로그로 확인. Phase 2 diagnostic 결과와 z_gate 초기 거동이 크게 달라지지 않는지(온도만 추가했을 뿐 아직 TAU_GATE는 안 바꿨으므로) 대조.

---

## Phase 4 — 적응형 TAU_GATE 로직 (best-so-far + 쿨다운 프리즈)

### 4.1 매 `train_step` 호출 후 (또는 `mp_rel_err` 계산 직후) 갱신
```python
# mp_rel_err 계산 직후, gate_multiplier 계산 이전에 위치
if pruning_state['cooldown_remaining'] == 0:   # ── 하드 프리즈: 쿨다운 중엔 갱신 자체를 skip ──
    pruning_state['best_so_far_mp_rel_err'] = min(
        pruning_state['best_so_far_mp_rel_err'], mp_rel_err)
    pruning_state['ema_best_so_far'] = (
        TAU_EMA_ALPHA * pruning_state['best_so_far_mp_rel_err']
        + (1.0 - TAU_EMA_ALPHA) * pruning_state['ema_best_so_far']
    )
    pruning_state['current_tau_gate'] = min(
        TAU_CEILING, max(TAU_MIN, pruning_state['ema_best_so_far']))
# 쿨다운 중에는 current_tau_gate를 이전 값 그대로 유지(재계산하지 않음)
```
**주의**: `train_step()`은 현재 `pruning_state`를 인자로 받지 않는다(시그니처: `train_step(model, data, optimizer, target_mps, target_area, epoch, max_epochs, weights, curriculum, curriculum_ratio, collision_spec, alda_state)`). 이 로직을 위해 `train_step` 시그니처에 `pruning_state`를 추가하거나, `mp_rel_err`/`gate_multiplier` 계산을 `run_training`의 epoch 루프로 옮겨야 한다. **권장: `train_step`에 `pruning_state` 인자를 추가**(가장 적은 코드 변경).

### 4.2 `gate_multiplier` 계산부 교체
```python
# 변경 전: gate_multiplier = float(torch.sigmoid(torch.tensor(SPARSE_K * (TAU_GATE - mp_rel_err))).item())
# 변경 후:
gate_multiplier = float(torch.sigmoid(
    torch.tensor(SPARSE_K * (pruning_state['current_tau_gate'] - mp_rel_err))).item())
```

### 4.3 history/로그 반영
`run_training()`의 `history` dict 초기화부에 `'temperature': []`, `'current_tau_gate': []`, `'best_so_far_mp_rel_err': []` 키를 추가하고 매 epoch 루프에서 append. **`visualize_training`/`visualize_epoch_snapshots`는 이번 범위에서 새 패널을 추가하지 않으므로 이 키들을 실제로 플롯할 필요는 없다 — 단, 키 자체가 없으면 향후 플롯 추가 시 KeyError가 나므로 지금 키만 확보해둔다** (Critic 라운드 지적, 필수는 아니나 권장).

### 4.4 검증
쿨다운이 한 번도 발동하지 않는 짧은 런에서 `current_tau_gate`가 `TAU_CEILING=0.10`에서 시작해 `mp_rel_err`가 개선됨에 따라 `TAU_MIN=0.05` 쪽으로만 단조 감소하는지(절대 증가하지 않는지) 확인. 만약 State-Only Reset이 발동하는 긴 런이라면, reset 직후 `PRUNING_COOLDOWN_EPOCHS` 동안 `best_so_far_mp_rel_err`/`current_tau_gate`가 로그상 완전히 고정(freeze)되어 있는지 확인.

---

## Phase 5 — 체크포인트 직렬화 (선택, 재개 학습을 쓰는 경우 필수)

현재 v17에 명시적 체크포인트 저장/재개 로직이 없다면 이 Phase는 스킵 가능. 만약 향후 재개(resume) 학습을 지원하게 되면, `best_feasible`/`best_mp` 딕셔너리에 저장하는 `state_dict`와 함께 `pruning_state` 전체(신규 필드 포함)를 같이 저장해야 재개 시 적응형 임계값이 기본값으로 리셋되지 않는다. 이번 v17 범위에서는 **`best_feasible['pruning_state']`, `best_mp['pruning_state']` 저장부(약 1311행, 1322행)에 신규 3개 키가 이미 dict 전체 복사이므로 자동 포함되는지만 확인**(현재 코드는 `ewma_z`/`state`/`pending_duration`만 선택적으로 복사하고 있어 신규 필드는 포함되지 않음 — 포함하도록 복사 로직에 세 줄 추가 필요).

```python
best_feasible['pruning_state'] = {
    'ewma_z': pruning_state['ewma_z'].clone(),
    'state': list(pruning_state['state']),
    'pending_duration': pruning_state['pending_duration'].clone(),
    'best_so_far_mp_rel_err': pruning_state['best_so_far_mp_rel_err'],
    'ema_best_so_far': pruning_state['ema_best_so_far'],
    'current_tau_gate': pruning_state['current_tau_gate'],
}
```
(동일 패턴을 `best_mp['pruning_state']` 저장부에도 적용)

---

## Phase 6 — 전체 통합 회귀 확인 (최종 진단 재실행)

**왜 필요한가**: Prosecution이 지적한 "diagnostic이 완성된 시스템을 반영하지 못한다"는 우려에 대한 저비용 절충안(judge 채택). Phase 2의 격리된 baseline 대신, 지금은 temperature/adaptive TAU_GATE가 모두 적용된 상태에서 같은 진단을 다시 돌려 비교한다.

```bash
python uni_section_v17.py --diagnostic --diagnostic-part-id 4
```
- Phase 2에서 기록해둔 baseline 로그와 이번 로그의 `ewma_z[4]` 하강 궤적을 비교. 완만해졌는지(temperature 어닐링으로 그래디언트 흐름이 개선됐는지)/오히려 느려졌는지(adaptive TAU_GATE가 초반에 압력을 더 낮췄기 때문일 수 있음) 확인해 다음 튜닝 라운드의 입력으로 삼는다.
- `mp_rel_err` 수렴 곡선, `l_collision`, State-Only Reset 발동 여부 등 기존 command_v17.md 판정 기준(feasibility, double-shock 여부)이 이번 변경으로 악화되지 않았는지 최종 확인.
- 이 시점에도 `thresh_low=0.08`/`thresh_high=0.15`는 그대로다 — 두 진단 실측 로그(Phase 2, Phase 6)를 모두 근거로 삼아야 다음 이터레이션에서 임계값 조정 여부를 판단할 수 있다(이번 command 문서의 범위 밖).

---

## 범위 밖 (다음 이터레이션 관찰 대상)

- `CANDIDATE_PARTS=[3,4]` 고정을 target-dependent 동적 선택으로 바꾸는 것은 이번 범위에서 명시적으로 제외(over-engineering으로 기각, idea 문서 결론 유지). 향후 다른 `TARGET_MP`로 재학습 시 로그만 관찰.
- `thresh_low`/`thresh_high` 값 자체의 변경은 Phase 2/6 실측 로그가 축적된 이후 별도 이터레이션에서 결정.

---

<details>
<summary>숙의 과정 (Synod 세션 상세)</summary>

### 모델 기여
- **Claude (Validator):** init_log_alpha 수정이 diagnostic보다 선행해야 하는 데이터 의존성 지적, best_so_far 쿨다운 프리즈 필요성 확인
- **Gemini (Architect, flash 폴백):** 5-phase 초안 작성(추후 diagnostic 위치 재배치), temperature 어닐링 수식, Defense 라운드에서 하드 프리즈/격리된 diagnostic의 우수성을 방어
- **OpenAI (Explorer, gpt4o):** temperature 배관 누락이라는 가장 중요한 갭 최초 발견, best_so_far/State-Only-Reset 상호작용 문제 제기, Prosecution 라운드에서 "완성된 시스템 기준 재진단" 필요성 제기(→ Phase 6으로 절충 채택)

### 해결된 주요 쟁점
1. "diagnostic hook을 언제 실행해야 하는가?" → Phase 2(격리된 baseline)와 Phase 6(통합 회귀) 두 번 실행으로 절충
2. "best_so_far를 쿨다운 중에도 갱신할 것인가?" → 하드 프리즈로 확정(tolerance 방식은 불필요한 하이퍼파라미터 추가로 기각)

### 신뢰 점수 (최종 라운드 기준)
- Claude: 87
- Gemini: 95 (Defense 라운드)
- OpenAI: 85 (Prosecution 라운드)

</details>

### 신뢰도: 91%
(Gemini pro가 재차 rate-limit되어 flash로 폴백했으나 응답 품질/XML 형식은 정상. 이 문서는 이론적 설계이며 실제 코드 수정 및 학습 실행 후 Phase 2/Phase 6 로그로 검증하기 전까지는 확정이 아님 — 특히 `train_step` 시그니처에 `pruning_state`를 추가하는 부분은 호출부 전체를 함께 수정해야 하므로 구현 시 주의.)
