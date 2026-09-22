# uni_section_v4 학습 실패 원인 분석 및 해결 방안

`uni-section/results/uni_section_v4.md` 학습 로그와 `uni_section_v4.py` 코드를 근거로,
"단면 형상 붕괴 / 충돌 방지 미작동 / 전소성 모멘트 전혀 미충족" 3가지 증상의 근본 원인을 분석하고,
"물리 제약(충돌 방지) → 전소성 모멘트 → 나머지 loss" 순으로 우선순위를 재구성하는 방안을 제시한다.

---

## 1. 증거: 실제 로그 데이터

```
Epoch ||  Loss  ||  Phys  |  Smth  |  Area  |  Mass  |  Coll
00000 || 428.0568 || 0.6868 | 0.3333 | 1162.9 | 0.0012 | 42.1189
00019 || 482.2459 || 0.6869 | 0.3392 | 1166.6 | 0.0012 | 47.5377
00039 || 433.4933 || 0.6915 | 0.3476 | 1170.0 | 0.0012 | 42.6571
00059 || 282.1672 || 0.7026 | 0.3464 | 1161.1 | 0.0012 | 27.5119
00079 || 225.7935 || 0.7452 | 0.3808 | 1149.2 | 0.0011 | 21.8304
00099 || 300.5475 || 0.8580 | 0.1015 | 1117.4 | 0.0011 | 29.1956
00119 || 203.5092 || 0.8728 | 0.0975 | 1140.7 | 0.0011 | 19.4770
00139 || 402.3585 || 0.8906 | 0.1059 |  979.3 | 0.0010 | 39.3440
00159 || 220.6576 || 0.8910 | 0.1091 | 1064.2 | 0.0011 | 21.1736
00179 || 300.2043 || 0.8918 | 0.1074 | 1171.9 | 0.0012 | 29.1274

최종 결과 요약
  Target Mp     :     47,421,470 N·mm
  Final pred_mp :     10,304,236 N·mm
  Final Error   :  78.27%
```

핵심 관찰:
- **`l_phys`가 학습이 진행될수록 오히려 증가한다** (0.6868 → 0.8918). `l_phys = sqrt(abs(err_ratio))`이므로 0.89² ≈ 0.79, 즉 최종 오차 78~79%와 정확히 일치 — Mp 오차가 전혀 줄지 않고 오히려 악화됨.
- **Total Loss가 단조 감소하지 않고 200~480 사이를 계속 진동**한다 — 학습이 특정 방향으로 수렴하지 못하고 있다는 뜻.
- **`l_collision`(raw ~20~47)이 다른 모든 항보다 압도적으로 크다.** `w_collision=10`을 곱하면 매 epoch loss의 90% 이상이 collision 항 하나에서 나온다.

---

## 2. 근본 원인 분석

### 원인 1 (핵심): Collision loss가 가중치 적용 후 전체 loss를 지배 — phys 학습 신호를 압도

epoch 0 기준:
- `w_phys * l_phys = 10 * 0.6868 = 6.87`
- `w_collision * l_collision = 10 * 42.1189 = 421.19`

→ **총 loss(428.06) 중 98.4%가 collision 항 하나에서 발생.** `loss.backward()`가 수행하는 gradient descent는 사실상 "collision loss만 최소화"하는 것과 다름없고, phys 항의 gradient 기여는 노이즈 수준으로 묻힌다. `command_v3.md`에서 제안한 `w_phys=10, w_collision=10` 가중치는 **두 loss의 원시 스케일(raw magnitude)이 전혀 다르다는 점을 고려하지 않았다** — phys는 `sqrt(abs(err))` 형태로 [0,1] 범위인데 반해, collision은 mm 단위의 침투 거리 합산이라 스케일이 수십 배 크다. 동일한 weight=10을 곱해도 실질 영향력은 전혀 다르다.

### 원인 2: 커리큘럼이 collision을 처음부터 100% 활성화 — "phys 우선 학습" 설계 의도와 정면 충돌

`get_curriculum_weights()`는 `s_collision = 1.0`으로 **항상 고정**되어 있고 ramp 대상이 아니다(`s_phys`도 1.0 고정). `command_v3.md` §3에서 "Phase 1은 phys만 집중, collision은 처음부터 1.0 고정 유지"라고 설계했지만, 실제로는 이 설계가 **collision이 phys보다 스케일상 압도적으로 크다는 사실과 충돌**한다. "먼저 물리 수렴에 집중" 하려던 phase1 의도가 무력화되고, 실제로는 "처음부터 collision만 학습"이 되어버렸다.

### 원인 3: `compute_collision_loss_v3`가 인접하지 않은 파트 쌍까지 전부 검사 → 구조적으로 0에 수렴 불가능한 loss

`parts_order_in_sections = {0: [0, 2, 1, 3, 4]}`로 정의된 순서에서, 코드는 `i < j`인 **모든 조합**(Outer-Plate, Outer-Patch1, Outer-Patch2, InnerHat-Patch1, ... 등 10개 쌍 × 2방향)에 대해 침투를 검사한다. 그런데 `build_bpillar_section()`의 실제 좌표를 보면:

- Plate(#03, part 1): y ≈ 8.05~28.05
- Inner Hat(#06, part 2): y ≈ 9.65~45
- Patch1(#07, part 3): y ≈ 9.55~16.5 (Plate와 y범위가 **의도적으로 겹치는** "샌드위치" 구조)
- Patch2(#08, part 4): y ≈ 7.45 (Plate 우측 플랜지와 **의도적으로 겹치는** 위치)

즉 Patch1/Patch2는 원래 설계부터 Plate와 같은 y 위치에 "붙어 있는" 보강재이며, 서로 침투하지 않는 것이 오히려 비정상이다. 그런데 전역 order `[0, 2, 1, 3, 4]` 기준으로 모든 비인접 쌍(Outer-Patch1, InnerHat-Patch2 등)까지 "층 순서 위반"으로 계산하면서, **설계상 정상인 배치조차 collision loss가 0으로 수렴할 수 없는 구조**가 된다. 이것이 로그에서 collision loss가 20 epoch가 지나도 20~47 사이에서 줄지 않고 진동하는 이유다 — 모델이 아무리 노드를 밀어내도 구조적으로 만족 불가능한 제약을 풀려고 하다가 형상이 계속 흔들리는 것.

### 원인 4 (결과): 형상 붕괴는 원인 1~3의 필연적 결과

Total loss gradient의 대부분이 "말이 안 되는 collision 제약을 풀라"는 방향으로 흐르기 때문에, 모델은 물리(Mp) 목표나 형상 매끄러움과 무관하게 노드를 이리저리 밀어내는 방향으로만 학습된다. `s_smooth`, `s_mass`가 curriculum phase1(epoch<18) 구간에서 0으로 억제되어 있는데도 형상이 무너지는 것은, **smooth loss가 꺼진 상태에서 collision gradient가 형상을 지배**하기 때문이다.

---

## 3. 해결 방안: "물리 제약 → 전소성 모멘트 → 나머지 loss" 순서로 재설계

사용자 요청대로 (1) 충돌 방지 등 물리적 제약조건을 우선 만족시키고, (2) 그 다음 전소성 모멘트를 최대한 만족시킨 후, (3) 나머지 loss(smooth/mass)를 학습시키는 **3단계 순차 커리큘럼**으로 재구성한다.

### 3.1 Collision pair 정의를 인접 레이어만으로 축소

전역 order의 모든 조합이 아니라, **`parts_order_in_sections`의 인접한 (i, i+1) 쌍만** 검사하도록 `compute_collision_loss_v3`를 수정한다. B-Pillar처럼 패치가 특정 파트와 의도적으로 겹치는 구조에서는, "바로 아래/위 레이어와의 침투만" 억제하는 것이 물리적으로 타당하다.

```python
# 변경 전: for j in range(i + 1, len(ordered_parts))  → 모든 비인접 쌍까지 검사
# 변경 후: 인접 쌍만
for i in range(len(ordered_parts) - 1):
    outer_part_id = ordered_parts[i]
    inner_part_id = ordered_parts[i + 1]   # i+1 고정 → 인접 레이어만
    ...
```

이렇게 하면 원래 설계상 겹쳐야 하는 비인접 파트(Outer-Patch1 등)는 더 이상 침투로 취급되지 않고, collision loss가 실제로 0에 수렴 가능한 목표가 된다.

### 3.2 Loss 스케일 정규화 (raw magnitude 기준 가중치 재산정)

정적 가중치(`weights` dict)를 그대로 곱하기 전에, 각 loss 항을 **비슷한 스케일로 정규화**한다. 가장 간단한 방법은 학습 시작 시(첫 1~5 epoch) 각 항의 평균 raw magnitude를 측정해 자동으로 스케일을 맞추는 것이지만, 우선은 로그에 나온 실측 스케일을 근거로 정적 가중치를 재조정한다.

| 항목 | 관측 raw 스케일 (epoch 0 기준) | 기존 weight | 기존 기여도 | 제안 weight | 제안 기여도 |
|---|---|---|---|---|---|
| l_phys | ~0.7 | 10.0 | 6.9 | 20.0 | 14.0 |
| l_smooth | ~0.33 | 0.1 | 0.03 | 0.5 | 0.17 |
| l_mass | ~0.001 | 1.0 | 0.001 | 1.0 | 0.001 |
| l_collision | ~42 | 10.0 | 421 | **0.3** (인접쌍 축소 후 재측정 필요) | ~10~15 목표 |

collision은 인접쌍 축소(§3.1) 이후 raw 값이 다시 측정되어야 하므로, 위 표의 0.3은 잠정값이며 §3.4의 순차 학습으로 실제 값을 관찰한 뒤 재조정해야 한다. 원칙은 **"어떤 항도 total loss의 70~80%를 넘지 않도록"** 스케일을 맞추는 것이다.

### 3.3 Curriculum을 "물리 제약 → Mp → 나머지" 3단계로 재설계

기존 `get_curriculum_weights`는 phys와 collision을 둘 다 1.0 고정, smooth/mass만 ramp하는 2단 구조였다. 이를 아래처럼 3단계로 바꾼다.

- **Stage A (예: epoch 0 ~ 15%)**: `s_collision=1.0` (물리적 침투는 처음부터 강하게 억제), `s_phys=1.0`, `s_smooth=s_mass=0.0`. 이 구간에서는 "형상이 겹치지 않으면서 Mp에 접근"하는 것만 학습.
- **Stage B (15% ~ 60%)**: collision이 §3.1 수정으로 실제로 낮아졌는지 확인 후, `s_phys` 가중치를 유지하며 `s_smooth`, `s_mass`를 기존 sine ramp(0→1)로 서서히 도입.
- **Stage C (60% ~ 100%)**: 모든 항 가중치 1.0 고정, 미세조정.

```python
def get_curriculum_weights_v4(epoch, total_epochs, curriculum_ratio):
    """3-stage: collision+phys 우선 → smooth/mass 순차 도입"""
    stage_a_end = int(total_epochs * curriculum_ratio[0])   # 예: 0.15
    stage_b_end = int(total_epochs * curriculum_ratio[1])   # 예: 0.60

    s_phys = 1.0
    s_collision = 1.0  # 물리 제약은 항상 최우선 유지

    if epoch < stage_a_end:
        s_smooth = s_mass = 0.0
    elif epoch < stage_b_end:
        x = (epoch - stage_a_end) / max(stage_b_end - stage_a_end, 1)
        progress = 0.5 * (1 + math.sin(math.pi * (x - 0.5)))
        s_smooth = s_mass = progress
    else:
        s_smooth = s_mass = 1.0

    return s_phys, s_smooth, s_mass, s_collision
```

이는 기존 `get_curriculum_weights`와 반환 시그니처가 동일하므로 `train_step` 수정 없이 함수만 교체하면 된다.

### 3.4 (선택) 2-Phase 물리 검증 후 학습 재개 방식

더 안전하게 가려면, 코드 구조를 다음처럼 두 단계로 완전히 분리하는 것도 고려할 수 있다.

1. **Phase 1 스크립트**: `loss = w_phys * l_phys + w_collision * l_collision` 두 항만으로 별도 학습 실행 → `l_phys` 수렴 곡선과 최종 오차(%)를 확인. 목표: 수십 epoch 내 오차 10% 이하로 감소하는지 확인.
2. Phase 1이 안정적으로 수렴함을 확인한 뒤에만 §3.3의 3-stage curriculum(전체 4항 통합)을 적용한 Phase 2 학습으로 넘어간다.

이 방식은 "형상 붕괴"와 "Mp 미충족"이 collision/스케일 문제 때문인지, 아니면 모델 구조(CGDN)나 학습률 등 다른 요인 때문인지 분리해서 진단할 수 있다는 장점이 있다.

### 3.5 로그에 weight-adjusted 기여도 출력 추가 (재발 방지)

현재 로그는 raw loss 값만 출력해서 "어느 항이 total loss를 지배하는지"가 즉시 보이지 않는다 — 이번 문제도 사후 분석으로 raw 로그를 다시 계산해서야 발견됐다. `train_step`의 반환값에 `weights['w_*'] * l_* * s_*` (가중치 적용된 기여도)를 추가하고, `run_training`의 print 문에 이를 표시하면 향후 동일한 "한 항이 loss를 지배하는" 실패를 학습 중간에 즉시 감지할 수 있다.

```python
contrib_phys      = weights['w_phys']      * l_phys_total.item() * s_phys
contrib_collision = weights['w_collision'] * l_collision.item()  * s_collision
# ... loss.item() 대비 각 항의 비율(%)을 함께 출력
```

---

## 4. 적용 순서 요약

1. `compute_collision_loss_v3`의 pair 생성 로직을 인접 레이어(i, i+1)만 검사하도록 수정 (§3.1).
2. 수정된 collision loss의 raw 스케일을 epoch 0~5 구간에서 재측정하고, phys 대비 70~80%를 넘지 않도록 `w_collision`을 재조정 (§3.2).
3. `get_curriculum_weights`를 3-stage 버전으로 교체: collision+phys 우선 학습 → smooth/mass 순차 도입 (§3.3).
4. (권장) Phase 1(phys+collision only)을 별도로 먼저 돌려 Mp 수렴이 정상 작동하는지 확인한 뒤 전체 통합 학습으로 전환 (§3.4).
5. `train_step`/`run_training` 로그에 가중치 적용 후 기여도(%)를 출력하도록 추가해 향후 동일 문제를 조기에 발견 (§3.5).

이 순서를 모두 적용한 뒤에도 `l_phys`가 개선되지 않는다면, collision/스케일 문제가 아니라 모델(CGDN) 용량·학습률·`max_displacement=50.0` 클램프 등 다른 요인을 추가로 점검해야 한다.
