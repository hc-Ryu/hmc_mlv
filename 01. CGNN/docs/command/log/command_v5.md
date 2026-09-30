# uni_section_v5 재실패 원인 분석: 20260401_yj.py와의 구조적 차이

`uni-section/results/uni_section_v5.md` 로그와 `20260401_yj.py`(잘 수렴하는 원본)를 직접 비교해
v5가 여전히 형상 붕괴/Mp 미충족을 겪는 근본 원인을 재진단한다. **command_v4.md의 "collision을
인접쌍만 검사하도록 축소"라는 처방은 방향이 틀렸다** — 아래 실측 로그가 이를 증명한다.

---

## 1. 증거: v5 로그, 처방이 의도와 반대로 작동함

```
Epoch ||  Loss  ||  Phys  |  Smth  |  Area  |  Mass  |  Coll  ||  Contrib%(phys/smth/mass/coll)
00000 || 44.7665 || 0.6868 | 0.3333 | 1162.9 | 0.0012 | 103.4347 ||  30.7% /   0.0% /   0.0% /  69.3%
00019 || 109.4527 || 0.1614 | 0.1735 | 1542.7 | 0.0015 | 354.0826 ||   2.9% /   0.0% /   0.0% /  97.1%
00039 || 51.1999 || 0.6922 | 0.0938 | 1533.6 | 0.0015 | 124.5126 ||  27.0% /   0.0% /   0.0% /  73.0%
...
00179 || 35.2973 || 0.7502 | 0.1245 | 1595.1 | 0.0016 | 67.4295 ||  42.5% /   0.2% /   0.0% /  57.3%

최종 결과 요약
  Final pred_mp :     20,584,200 N·mm
  Final Error   :  56.59%
```

command_v4.md는 "collision pair를 all-pairs(10쌍) → 인접쌍(4쌍)으로 줄이면 raw 값이 줄어들 것"이라 가정했다. 그러나 실측 결과는 정반대다.

| | v4 (all-pairs, 10쌍) | v5 (인접쌍만, 4쌍) |
|---|---|---|
| `l_collision` raw (epoch 0) | 42.1 | **103.4** (2.5배 증가) |
| `w_collision` | 10.0 | 0.3 (33배 축소) |
| collision이 total loss에서 차지하는 비율 | ~98% | **여전히 57~97%** |
| Area (epoch 179) | 1171.9 | **1595.1** (초기 대비 +37%, 형상이 더 크게 팽창) |
| Final Mp Error | 78.27% | 56.59% (개선됐지만 여전히 미수렴) |

`w_collision`을 33배나 줄였는데도 collision이 여전히 loss의 절반 이상을 차지한다는 것은, **raw magnitude 자체가 줄어든 게 아니라 오히려 늘었기 때문**이다. 이는 "인접쌍만 남기면 값이 작아질 것"이라는 가정이 틀렸음을 뜻한다.

---

## 2. 왜 인접쌍 축소가 collision을 더 악화시켰는가

`compute_collision_loss_v3`는 각 쌍의 침투 손실을 **`valid_pairs_count`(검사한 쌍의 개수)로 나눠 평균**을 낸다. all-pairs(10쌍) 방식에서는 "구조적으로 전혀 겹치지 않는" 먼 쌍(예: Outer-Patch2)까지 포함되어 있어서, 이런 쌍들의 loss=0이 평균을 크게 희석시켰다. 반면 인접쌍(4쌍)만 남기면 **희석 효과가 사라지고, 진짜로 겹치는 쌍(Plate↔Patch1 등)의 큰 위반량이 그대로 평균에 반영**된다. 즉 command_v4.md의 수정은 "잡음(항상 만족되는 먼 쌍)을 제거"한 것이 아니라 "핵심 문제(구조적으로 만족 불가능한 인접 쌍)를 더 선명하게 드러낸 것"이었다 — 문제를 숨기던 평균화 효과가 사라지면서 원인이 명확해졌을 뿐, 근본 문제(특정 파트 쌍이 애초에 collision-free 형상으로 수렴할 수 없음)는 그대로였다.

---

## 3. 20260401_yj.py는 왜 같은 코드(all-pairs, 동일 가중치 비율)로도 수렴하는가

`20260401_yj.py` 메인 실행부(line 1502)의 가중치는 `w_phys=10, w_collision=10`으로 **v4와 완전히 동일한 비율**이고, `compute_collision_loss_v3`도 **all-pairs 버전 그대로**다(별도 인접쌍 최적화 없음, line 613: `for j in inner_range: range(i+1, len(ordered_parts))`). 그런데도 원본은 잘 수렴한다고 알려져 있다. 즉 **문제는 "가중치 비율"도 "pair 선택 방식"도 아니라, 두 모델이 다루는 단면 지오메트리 자체가 다르다는 데 있다.**

### 핵심 차이: "레이어 중첩(shell nesting)" vs "라미네이트 보강(patch lamination)"

`20260401_yj.py`의 17-section B-Pillar는 각 섹션마다 Outer(#00) / Reinf(#03류) / Inner(#06류) 가 **Y축 상에서 서로 겹치지 않는 별도 밴드**를 형성하도록 좌표가 설계되어 있다(`lower_section`, `upper_section` 좌표 테이블 참고 — 각 파트가 뚜렷이 분리된 Y 범위를 차지). 이런 구조에서는 `compute_collision_loss_v3`가 "각 레이어가 서로의 영역을 침범하지 않는다"는 물리적으로 달성 가능한 목표를 학습시킨다.

반면 `pna_solver_validate_v9.py` 기반의 **단일 섹션 5-part 구조**(v4/v5가 그대로 물려받은 `build_bpillar_section()`)는 다음과 같이 Patch1/Patch2가 Plate와 **같은 Y 범위에 의도적으로 겹치도록** 설계되어 있다(라미네이트 보강재 — "판 위에 덧대는 패치"):

- Plate(#03, part 1): y ≈ 8.05 ~ 28.05
- Patch1(#07, part 3): y ≈ 9.55 ~ 16.5 → **Plate의 y 범위(8.05~28.05) 안에 완전히 포함**
- Patch2(#08, part 4): y ≈ 7.45 (고정) → Plate 우측 플랜지(y=8.05)와 거의 동일 위치

이건 "레이어가 겹치면 안 되는 중첩 구조"가 아니라 "패치가 판 표면에 용접/라미네이트되는 보강 구조"다. `compute_collision_loss_v3`는 이 둘을 구분하지 못하고 Patch1-Plate, Patch1-InnerHat 등 모든 인접 쌍을 "겹치면 안 되는 레이어"로 취급하기 때문에, **애초에 목표 형상 자체가 collision loss=0을 만족할 수 없는 기하학적 모순** 상태에 있다. all-pairs든 인접쌍이든 이 모순은 사라지지 않으며, 오직 평균화로 희석되는 정도만 다를 뿐이다(§2).

---

## 4. 근본 해결 방안: Patch를 collision 대상에서 제외 (구조적 재분류)

### 4.1 parts_order_in_sections에서 Patch(3, 4)를 collision 검사 대상에서 제외

Patch1/Patch2는 "침투하면 안 되는 별도 레이어"가 아니라 "숙주 파트(Plate)에 부착되는 보강재"이므로, collision loss 자체를 계산할 때 제외하는 것이 물리적으로 타당하다.

```python
## 계층 순서: collision은 "겹치면 안 되는 shell"만 대상으로 함
## Patch1(3)/Patch2(4)는 Plate(1) 위에 라미네이트되는 보강재이므로 collision 대상에서 제외
parts_order_in_sections = {
    0: [0, 2, 1],   # Outer → InnerHat → Plate 만 검사 (Patch 제외)
}
```

`compute_collision_loss_v3`는 `parts_order_in_sections`에 없는 part_id는 애초에 순서 배열에 포함되지 않으므로 자동으로 검사에서 빠진다(코드 수정 불필요, 설정값만 변경).

### 4.2 (선택) Patch에는 "부착 손실(attachment loss)"을 별도로 적용

Patch가 숙주 파트에서 너무 멀어지는 것은 막아야 하므로, "침투 금지"가 아니라 "숙주와의 거리 최소화" 형태의 손실로 대체할 수 있다.

```python
def compute_attachment_loss(new_coords, part_ids, host_pairs):
    """
    host_pairs: [(patch_part_id, host_part_id), ...]
    패치의 각 노드가 숙주 파트의 최근접 노드와 너무 멀어지지 않도록 유도.
    """
    loss = torch.tensor(0.0, device=new_coords.device)
    for patch_id, host_id in host_pairs:
        patch_coords = new_coords[part_ids == patch_id]
        host_coords  = new_coords[part_ids == host_id]
        if patch_coords.shape[0] == 0 or host_coords.shape[0] == 0:
            continue
        dist = torch.cdist(patch_coords, host_coords).min(dim=1)[0]
        loss = loss + dist.mean()
    return loss
```

이 손실은 §4.1과 별도로 작은 가중치(`w_attach`)로 추가할 수 있으며, 이번 command_v5 범위에서는 필수는 아니고 §4.1만으로도 충분히 문제가 해소되는지 먼저 확인해야 한다.

### 4.3 §4.1 적용 후 raw 스케일 재측정 및 weight 재조정 (command_v4.md §3.2 절차 재수행)

Patch를 제외하면 `l_collision`의 대상이 Outer-InnerHat-Plate 3파트 2쌍(양방향 4항)으로 줄어든다. 이 구조는 원본(`20260401_yj.py`)처럼 Y-band가 어느 정도 분리되어 있으므로 raw 값이 크게 낮아질 가능성이 높다. 다만 실제로는 재측정 전까지 확신할 수 없으므로, §4.1 적용 후 반드시:

1. `w_collision`을 원래처럼(예: v4 이전 값 10.0 근처로) 되돌리거나 유지한 채 epoch 0~10 구간의 raw `l_collision` 값을 관찰.
2. `Contrib%(coll)`이 여전히 70% 이상이면 그때 `w_collision`을 낮춘다 (collision이 진짜로 원본처럼 낮은 스케일로 떨어졌다면 낮출 필요가 없을 수도 있음).

즉, command_v4.md의 "가중치를 미리 축소"하는 접근 대신, **먼저 기하학적 모순(패치-플레이트 오검사)을 제거한 뒤 → 그 결과로 나온 실측 raw 스케일을 보고 가중치를 조정**하는 순서로 바꾼다.

### 4.4 Area 팽창(1163→1595, +37%)에 대한 추가 점검

v5 로그에서 `Area`가 학습 내내 계속 커지는 것은 `l_mass`가 사실상 무의미하기 때문이다(`target_area=None`이면 `l_mass = area * 1e-6`로, 그냥 면적에 비례하는 forcing 없는 약한 항이라 면적 증가를 막지 못함). §4.1로 collision 모순이 해소된 뒤에도 형상이 계속 팽창한다면, `target_area`에 `build_bpillar_section()`의 초기 면적을 명시적으로 대입해 `l_mass`가 실제 "면적 보존" 역할을 하도록 바꿔야 한다.

```python
init_area, _ = compute_section_area(base_coords, t_full, edge_index_cpu)
history = run_training(..., target_area=init_area, ...)
```

---

## 5. 적용 순서 요약 (uni_section_v6 방향)

1. `parts_order_in_sections`에서 Patch1(3)/Patch2(4)를 제거하고 `[0, 2, 1]`(Outer-InnerHat-Plate)만 collision 대상으로 지정 (§4.1). **가장 먼저, 단독으로 적용하고 재실행해 raw `l_collision`이 실제로 낮아지는지 확인.**
2. §4.1만으로 `l_collision`이 낮은 스케일(예: 원본과 비슷한 한 자릿수~두 자릿수 초반)로 떨어지면, `w_collision`을 v4 이전 값(10.0)으로 복원 — command_v4.md에서 축소했던 가중치는 되돌린다.
3. `target_area`를 초기 면적으로 명시해 `l_mass`가 실질적인 면적 보존 제약으로 작동하도록 수정 (§4.4).
4. 3-stage curriculum(§3.3, command_v4.md 그대로 유지)은 그대로 두되, Stage A 구간에서 `l_collision`이 정상적으로 낮은 값에서 시작하는지 로그로 재확인.
5. 그래도 Mp가 수렴하지 않는다면, 그 다음으로 의심할 지점은 (a) `parts_order_in_sections`의 Outer-InnerHat-Plate 순서가 실제 Y 배치와 맞는지, (b) `20260401_yj.py`처럼 여러 섹션에 걸친 `l_phys` 평균화가 아니라 단일 섹션이라 gradient 신호가 상대적으로 약한 것은 아닌지(예: `n_iter`, learning rate) 순으로 점검한다.

**핵심 교훈**: command_v4.md는 "어느 loss가 total loss를 지배하는가"라는 증상만 보고 가중치/pair 개수를 조정했지만, 실제 원인은 **collision loss가 애초에 이 지오메트리에서 0으로 수렴 불가능한 목표를 검사하고 있었다는 것**이었다. 가중치 튜닝으로는 근본 모순을 해결할 수 없고, 검사 대상 자체(어떤 파트 쌍이 "진짜 겹치면 안 되는 레이어"인지)를 지오메트리 설계 의도에 맞게 재정의해야 한다.
