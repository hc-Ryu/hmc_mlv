# AI_design_v2.py의 "동적 두께 + Ghost Layer" 방식을 uni_section_v5.py에 통합하는 아이디어

`AI_design_v2.py`" C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\AI_design_v2.py
`uni_section_v5.py`: C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\uni-section\uni_section_v5.py

`AI_design_v2.py`가 실제로 어떻게 "재료 두께를 고려해 접합부에서도 겹치지 않게" 만드는지 코드를 근거로 분석하고,
이를 `uni_section_v5.py`(단일 섹션 구조 유지)에 통합해 `command_v5.md`에서 진단한 "Patch1/Patch2가 Plate와
설계상 겹치는데 collision loss가 이를 오판한다" 문제를 **더 원리적으로** 해결하는 방안을 제시한다.

---

## 1. AI_design_v2.py가 실제로 하는 일: "margin을 두껍게" 가 아니라 "두께 자체를 학습·소멸시킴"

먼저 짚어야 할 점: `AI_design_v2.py`는 "접합부의 물리적 마진(margin)을 재료 두께만큼 넓힌다"는 방식이 **아니다**. `compute_collision_loss_v3`의 `margin` 파라미터는 v5와 동일하게 상수(`margin=2`)로 고정되어 있다(line 448, 481). 대신 이 파일이 실제로 두께를 이용해 겹침 문제를 해결하는 방식은 **"겹쳐야 하는 파트의 두께를 모델이 스스로 0에 가깝게 줄여서(Ghost 상태), 그 쌍의 침투 손실 자체를 무력화"** 하는 것이다.

### 1.1 CGDN이 좌표뿐 아니라 두께(t)도 예측하도록 확장 (line 232~348)

```python
# [CMD-2] 두께 헤드 (물리 스케일 분리, Tanh로 [-1,1] 출력)
self.thickness_decoder = nn.Sequential(
    nn.Linear(hidden_channels, 32), nn.GELU(), nn.Linear(32, 1), nn.Tanh(),
)
...
t_new = t_initial + delta_t_part            # 파트 단위로 두께 변화량 통일 (제조 제약)
t_new = torch.clamp(t_new, min=0.1, max=3.0)

# Soft Ghost Gate — sigmoid: 0=ghost, 1=active
alpha   = torch.sigmoid(self.GHOST_STEEPNESS * (t_new - self.GHOST_THRESHOLD))
t_final = t_new * alpha + t_new * 0.05      # Leaky Ghost: 최소 5% 그래디언트 유지
return new_coords, delta_coords, t_final, alpha
```

즉 원래 고정값이던 `t`(두께)가 이제 **학습 가능한 출력**이 되고, 두께가 임계값(`GHOST_THRESHOLD=0.7mm`) 아래로 떨어지면 `alpha→0`이 되어 그 파트는 사실상 "유령(ghost)" 상태가 된다 — 실제로 존재하지 않는 것처럼 취급된다.

### 1.2 Collision loss를 두께(ghost) 상태로 가중 (line 481~525)

```python
def compute_collision_loss_v3_with_alpha(new_coords, part_ids, section_ids, alpha, margin=2, ...):
    ...
    alpha_o = alpha[mo].mean().detach()
    alpha_i = alpha[mi].mean().detach()
    ghost_weight = alpha_o * alpha_i
    ...
    total_loss = total_loss + (loss_oi + loss_io) * ghost_weight
```

두 파트 중 하나라도 ghost 상태(alpha≈0)이면 `ghost_weight≈0`이 되어 **그 쌍의 침투 손실이 사실상 꺼진다.** 즉 "이 파트는 두께가 거의 없으니 다른 파트와 겹쳐도 물리적으로 문제없다"는 논리를 손실 함수 레벨에서 구현한 것이다.

### 1.3 `.detach()`로 "두께를 깎아 충돌을 회피하는 꼼수" 차단 (CMD-2v)

`alpha_o`, `alpha_i`에 `.detach()`를 걸어, collision loss의 그래디언트가 thickness_decoder로 역전파되지 않게 막는다. 이게 없으면 모델이 "형상을 올바르게 만드는 대신 그냥 두께를 0으로 만들어 collision loss를 회피"하는 trivial solution으로 빠질 수 있다(파일 상단 주석 "Trivial Solution 방지 패치" 참고). 두께 감소는 **mass loss와 phys loss(Mp)에는 그대로 반영**되어야 하므로, collision loss만 이 신호를 못 보게 막는 것이다.

---

## 2. 왜 이 방식이 command_v5.md의 "패치 제외" 방식보다 원리적인가

`command_v5.md` §4.1은 "Patch1/Patch2를 `parts_order_in_sections`에서 통째로 빼서 collision 검사 대상에서 제외"하는 수동적 방법이었다. 이는 동작은 하지만 두 가지 한계가 있다.

1. **어떤 쌍을 제외할지 사람이 미리 정해야 한다** — 지오메트리가 바뀌면 다시 손으로 조정해야 함.
2. **"패치가 완전히 자유롭게 겹쳐도 된다"는 극단적 가정**이 된다 — 실제로는 패치가 plate 범위 안에서 자유롭게 아무 위치나 침투해도 되는 게 아니라, "라미네이트로 붙는 위치"에서만 허용되어야 한다.

AI_design_v2의 Ghost 방식은 이 문제를 **모델이 스스로 판단하게** 만든다: 학습 중 특정 위치에서 패치의 두께가 물리적으로 얇아져도(또는 0에 가까워져도) 무방하다고 모델이 판단하면 그 지점에서만 collision loss가 자동으로 완화되고, 나머지 지점에서는 여전히 침투가 억제된다. 즉 "이 파트 전체를 제외" (v5) 가 아니라 "이 파트가 겹쳐야 하는 국소적 상황에서만 물리적으로 타당한 방식(두께 축소)으로 겹침을 허용" (v1) 하는 더 세밀한 해법이다.

---

## 3. uni_section_v5.py에 통합하는 아이디어 (uni-section 단일 섹션 구조 유지)

`uni_section_v5.py`의 구조(단일 섹션, 5-part B-Pillar, `build_bpillar_section()` 초기 좌표 100% 유지)는 그대로 두고, 아래 요소만 이식한다.

### 3.1 CGDN에 두께 디코더 + Ghost Gate 추가

`uni_section_v5.py`의 `CGDN.forward()`는 현재 `(new_coords, delta_coords)`만 반환한다. `AI_design_v2.py`의 `thickness_decoder` + Ghost Gate 블록(§1.1)을 그대로 이식해 `(new_coords, delta_coords, t_final, alpha)`를 반환하도록 확장한다.

- **주의**: uni_section_v5는 단일 섹션이라 `x[:, 5]`(section_id)가 항상 0이지만, `composite_key = section_ids * max_parts + part_ids` 로직은 section_id가 상수여도 `part_ids` 그룹핑만으로 정상 동작하므로 수정 없이 그대로 사용 가능하다.
- Part별 두께를 하나로 통일하는 `scatter_add_` 로직(제조 제약)도 그대로 유지 — 5개 part(Outer/Plate/InnerHat/Patch1/Patch2) 각각 균일한 두께를 갖는다는 물리적 가정과 맞다.

### 3.2 collision loss를 alpha-가중 버전으로 교체

`command_v5.md`가 만든 `compute_collision_loss_v3`(인접쌍 버전)에 `_with_alpha` 변형을 추가한다. v5의 인접쌍 로직(§3.1, command_v4.md에서 도입)과 v1의 alpha 가중 로직(§1.2)을 합친다.

```python
def compute_collision_loss_v3_with_alpha(new_coords, part_ids, section_ids, alpha,
                                          margin=2, parts_order_in_sections=None):
    ...
    for i in range(len(ordered_parts) - 1):          # command_v5 인접쌍 유지
        outer_part_id = ordered_parts[i]
        inner_part_id = ordered_parts[i + 1]
        ...
        alpha_o = alpha[mask_outer].mean().detach()   # AI_design_v2 ghost weight
        alpha_i = alpha[mask_inner].mean().detach()
        ghost_weight = alpha_o * alpha_i
        total_loss = total_loss + (loss_oi + loss_io) * ghost_weight
```

이렇게 하면 **`parts_order_in_sections`에 Patch(3, 4)를 다시 포함시켜도** 된다 — command_v5.md처럼 통째로 빼지 않고, 모델이 Patch의 두께를 국소적으로 줄여 스스로 침투 문제를 해소하도록 맡길 수 있다.

### 3.3 phys / mass loss에 고정 `t` 대신 `t_final` 사용

현재 `train_step`은 `t = x[:, 6].unsqueeze(1)`로 고정 두께를 `calculate_mpl`(Mp 계산)과 `compute_mass_loss`에 넘긴다. 두께가 학습 대상이 되면 이 두 곳 모두 `t_final`을 사용하도록 바꿔야 물리적 일관성이 생긴다 — 두께가 얇아진 부분은 실제로 Mp 기여도와 질량 계산에서도 줄어들어야 "두께 축소로 겹침을 회피"하는 것이 물리적으로 타당해진다.

```python
new_coords, delta_coords, t_final, alpha = model(...)
pred_mp_section = calculate_mpl(coords_section, t_final[section_mask], fy_section, edge_index_section)
area, l_mass = compute_mass_loss(new_coords, t_final, edge_index, edge_attr, target_area)
l_collision  = compute_collision_loss_v3_with_alpha(new_coords, part_ids, section_ids, alpha,
                                                      parts_order_in_sections=parts_order_in_sections)
```

### 3.4 커리큘럼: collision도 progress로 ramp (AI_design_v2 CMD-4v 반영)

`command_v4.md`/`command_v5.md`에서는 `s_collision`을 항상 1.0으로 고정했다(물리 제약이니 처음부터 강하게 억제해야 한다는 논리였음). 그런데 `AI_design_v2.py`는 오히려 **`s_collision = progress`로 두께 학습이 먼저 자리 잡을 시간을 준 뒤에 collision을 서서히 강화**한다(CMD-4v 주석: "지연된 충돌 페널티 부과"). 이는 Ghost Gate가 의미 있는 판단을 내리려면 두께 디코더가 먼저 "이 위치는 얇아도 된다"는 신호를 학습할 최소한의 시간이 필요하기 때문으로 해석된다. uni_section_v5의 `get_curriculum_weights_v5`도 이 순서를 참고해 재검토할 필요가 있다 — 물리 제약(충돌 방지) 자체를 초반에 강하게 걸되, "두께를 통한 회피"라는 자유도가 아직 학습되지 않은 시점에 너무 강하게 걸면 여전히 형상을 억지로 밀어내는 붕괴가 재발할 수 있다.

### 3.5 Trivial-solution 방지 장치도 함께 이식

- `alpha`에 `.detach()`를 반드시 유지해 collision loss가 두께 디코더를 직접 줄이는 방향으로 학습되지 않게 한다(§1.3). 이게 없으면 "모든 파트를 얇게 만들어 collision을 0으로 만드는" 붕괴가 재발할 수 있다 — v5의 형상 붕괴 증상과 유사한 새로운 형태의 실패 모드가 될 위험이 있다.
- `GHOST_THRESHOLD`, `DELTA_SCALE`, thickness_decoder의 bias 초기화(`+0.07` → 시작 시 alpha≈0.8, active 상태)도 그대로 가져와, 학습 초반부터 모든 파트가 ghost로 죽어버리는 것(dead node)을 방지한다.
- Leaky Ghost(`t_final = t_new*alpha + t_new*0.05`)도 유지 — ghost 상태에서도 최소 5% 그래디언트 경로를 남겨, 한번 ghost가 된 파트가 다시 active로 복귀할 수 있는 경로를 보장한다.

---

## 4. 예상되는 효과와 리스크

### 기대 효과
- command_v5.md가 지적한 "Patch1/Patch2-Plate 구조적 겹침" 문제를, 파트를 통째로 제외하지 않고도 물리적으로 타당한 방식(국소적 두께 축소)으로 해소할 수 있다.
- 두께가 실제 설계 변수로 편입되므로, 향후 "재료 두께 최적화"까지 함께 학습하는 확장이 자연스러워진다.

### 리스크 / 확인 필요 사항
- **자유도 증가에 따른 학습 난이도 상승**: 좌표(2D) + 두께(1D)를 동시에 학습해야 하므로, command_v4.md/v5.md에서 이미 겪은 "여러 loss 항이 서로 충돌하며 진동"하는 문제가 더 복잡해질 수 있다. §3.4의 커리큘럼 순서를 신중히 설계해야 한다.
- **Ghost 상태가 물리적으로 말이 되는지 검증 필요**: Patch1/Patch2가 정말로 "얇아져도 되는" 위치인지, 아니면 구조 안전상 최소 두께를 반드시 유지해야 하는지는 도메인 지식으로 확인이 필요하다 — `t_new`의 `clamp(min=0.1, max=3.0)`이 최소 두께 하한을 두고 있으나, 이 하한값(0.1mm)이 B-Pillar 보강재로서 타당한지 별도 검토 필요.
- **`GHOST_THRESHOLD=0.7mm`, `GHOST_STEEPNESS=20.0` 등 하이퍼파라미터**는 `AI_design_v2.py`가 다루는 지오메트리(다른 초기 두께 스케일)에 맞춰 튜닝된 값일 수 있으므로, `uni_section_v5`의 실제 두께값(Outer 2.30mm, Plate 1.60mm, Patch1 1.40mm, Patch2 1.60mm)에 맞게 재조정이 필요할 수 있다.

---

## 5. 적용 순서 요약 (uni_section_v6 방향)

1. `CGDN`에 `thickness_decoder` + Ghost Gate 이식, `forward()` 반환값을 `(new_coords, delta_coords, t_final, alpha)`로 확장 (§3.1).
2. `compute_collision_loss_v3_with_alpha`를 v5의 인접쌍 로직과 결합해 신규 작성 (§3.2).
3. `train_step`에서 `calculate_mpl`, `compute_mass_loss`에 고정 `t` 대신 `t_final`을 사용하도록 교체 (§3.3).
4. `parts_order_in_sections`를 command_v5.md의 `[0, 2, 1]`(Patch 제외)에서 다시 `[0, 2, 1, 3, 4]`(전체 포함)로 복원 — Ghost 메커니즘이 Patch 겹침을 자체적으로 처리하게 함.
5. `get_curriculum_weights_v5`의 `s_collision` 고정(1.0) 정책을 재검토하고, AI_design_v2의 `s_collision=progress` 방식과 비교 실험 (§3.4).
6. `alpha.detach()` 등 trivial-solution 방지 장치를 반드시 포함해서 이식하고, 학습 로그에 `alpha`의 평균/최솟값(어느 파트가 ghost化되는지)을 추가로 출력해 모니터링한다.
