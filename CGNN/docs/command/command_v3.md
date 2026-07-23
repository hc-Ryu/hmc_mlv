# pna_solver_validate_v9 고도화 학습법 적용 아이디어

`20260401_yj.py`의 다목적 손실(loss) 구성과 커리큘럼 학습 방식을 분석하고,
이를 `validation/pna_solver_validate_v9.py`에 적용하기 위한 설계 아이디어를 정리한다.

**전제 조건 (변경하지 않음)**
- 노드 구성: v9의 `build_bpillar_section()` 그대로 (fix point, 노드/엣지 개수, 5-part 단일 단면)
- 섹션 수: 1개 섹션(section_id=0)만 사용 — 섹션 간 학습 순서/커리큘럼 없음
- keepout_loss(설계 제한 영역) 항은 이번 단계에서 설정하지 않음 (항 자체는 남겨두되 `keepout=None`으로 비활성)

---

## 1. 현재 상태 비교

| 항목 | `20260401_yj.py` (고도화) | `pna_solver_validate_v9.py` (현재) |
|---|---|---|
| Loss 항 | phys, smooth(angle), mass, collision_v3, continuity, shape, keepout — 7종 가중합 | `((pred_mp - target_mp)/target_mp)**2` 단일항 |
| 가중치 | `weights = {w_phys:10, w_smooth:0.1, w_mass:1.0, w_collision:10, w_continuity:0.01, w_shape:0.1, w_keepout:10}` | 없음 |
| 학습 순서 | 커리큘럼(`get_curriculum_weights`)으로 phase1(안정화)→phase2(sine ramp 0→1)→phase3(전항 활성) | 없음 (전 epoch 동일 loss) |
| phys loss 형태 | `sqrt(sum(abs(err_ratio))) / num_sections` (섹션 합산 후 sqrt) | `((pred-target)/target)**2` (단일 섹션 MSE 형태) |
| Optimizer | AdamW, `clip_grad_norm_(max_norm=1.0)` | AdamW, `clip_grad_norm_(max_norm=1.0)` (이미 동일) |

핵심 차이는 "단일 물리 손실"에서 "형상 안정성(smooth/shape)·질량·간섭 회피를 커리큘럼으로 서서히 반영하는 다목적 손실"로 바뀐다는 점이다. v9는 섹션이 1개뿐이므로 `compute_section_continuity_loss`, `compute_shape_continuity_loss`는 구조적으로 항상 0이 되어 도입해도 실효가 없다 — 이 두 항은 이번 적용에서 제외하는 것이 자연스럽다.

---

## 2. v9에 적용 가능한 손실 항 (섹션 1개 기준)

1개 섹션만 존재할 때 의미 있게 작동하는 항만 선별한다.

- **`l_phys`** (필수, 이미 존재) — solver Mp vs target Mp
- **`l_smooth`** (`compute_smoothness_loss_angle`) — 노드 좌우 엣지 각도 급변 억제. 단면 형상이 급격히 꺾이지 않도록 함. 섹션 무관하게 엣지 기반이라 그대로 적용 가능.
- **`l_mass`** (`compute_mass_loss`) — 목표 면적(`target_area`) 대비 오차. v9는 `target_area=None`이므로 기본 스케일(`area * 1e-6`)로 시작 가능, 추후 목표 면적 지정 시 오차형으로 전환.
- **`l_collision`** (`compute_collision_loss_v3`) — part 간 침투 방지. v9는 5-part 단일 섹션이므로 `parts_order_in_sections = {0: [0, 1, 2, 3, 4]}` 형태로 Outer→Plate→Inner→Patch1→Patch2 기하학적 순서를 정의해 적용 가능 (겹침 순서는 실제 좌표 상 바깥→안쪽 배치와 일치해야 함, 코드 수정 시 재확인 필요).
- **`l_continuity`, `l_shape`** — 섹션이 1개이므로 **제외** (unique_sections 루프가 1회뿐이라 항상 0, 계산 비용만 발생).
- **`l_keepout`** — 이번 단계에서 **제외** (요청사항). 다만 함수 시그니처와 `weights['w_keepout']=0.0`, `keepout=None`으로 자리만 남겨두면 이후 설계 제한 영역 도입 시 커리큘럼 로직을 재사용할 수 있다.

---

## 3. 커리큘럼 학습 적용 아이디어

`get_curriculum_weights(epoch, total_epochs, curriculum_ratio)`를 그대로 이식하되, v9에 없는 항(continuity/shape/keepout)에 대한 반환값은 사용하지 않는다.

- **Phase 1** (`epoch < total_epochs * curriculum_ratio[0]`): `s_phys=1.0` 고정, `s_smooth=s_mass=0.0` — 먼저 물리(Mp) 수렴에만 집중시켜 형상이 무너지기 전에 목표 모멘트 근사치를 잡는다.
- **Phase 2** (`curriculum_ratio[0] ~ curriculum_ratio[1]`): sine 기반으로 `s_smooth`, `s_mass`가 0→1로 완만→급격 증가. 이 구간에서 형상 정규화와 질량 페널티가 서서히 개입해 phys loss와 충돌하며 발산하는 것을 방지.
- **Phase 3** (`epoch >= total_epochs * curriculum_ratio[1]`): 모든 활성 항 가중치 1.0 고정 — 최종 미세 조정 단계.
- **`s_collision`은 커리큘럼 대상에서 제외하고 처음부터 1.0 고정** — `20260401_yj.py`와 동일한 설계 의도(간섭은 학습 초반부터도 절대 허용 불가한 hard-ish 제약)를 유지.

권장 초기값: `curriculum_ratio = [0.1, 0.5]` (원본 그대로) — total_epochs가 v9 기준 181 epoch로 상대적으로 짧으므로, 필요 시 `[0.15, 0.6]`처럼 phase1을 조금 더 길게 주는 것도 고려할 수 있다 (실험적으로 확인 필요).

---

## 4. 가중치(weights) 초기값 제안

원본 비율을 유지하되 keepout/continuity/shape를 제거한 4항 체계로 축소:

```python
weights = {
    'w_phys':      10.0,
    'w_smooth':      0.1,
    'w_mass':        1.0,
    'w_collision':  10.0,
}
```

- `w_phys`와 `w_collision`을 동일하게 가장 높게 두는 것은 원본 설계 의도(물리 정확도와 간섭 방지가 최우선)를 그대로 반영.
- `w_smooth`, `w_mass`는 부차적 정규화 항으로 낮게 유지.
- v9의 `l_phys`는 원본과 스케일이 다르므로(원본은 `sqrt(sum abs err)/num_sections`, v9는 `((pred-target)/target)**2`) 가중치 비율을 그대로 가져오면 실제 기여도가 달라질 수 있다 — phys 항 형태를 원본과 동일하게 `abs(err_ratio)` 기반으로 맞추거나(§5 참고), 도입 후 loss curve를 보고 `w_phys`를 재조정할 필요가 있다.

---

## 5. phys loss 형태 정합성

원본은 `l_phys_section = abs((pred_mp - target_mp)/target_mp)` 이후 여러 섹션을 합산하고 `sqrt(...) / num_sections`를 취한다. v9는 섹션이 1개뿐이므로 두 방식이 다음과 같이 수렴한다.

- 원본 방식 적용 시: `l_phys = sqrt(abs((pred_mp - target_mp)/target_mp))` (섹션 1개이므로 나눗셈 생략)
- 기존 v9 방식: `l_phys = ((pred_mp - target_mp)/target_mp) ** 2`

두 방식은 오차가 작을 때(<1) sqrt 방식이 더 큰 그래디언트를, 제곱 방식이 더 작은 그래디언트를 줌 — 학습 후반 미세조정에서 수렴 속도 차이가 날 수 있다. **원본과 동일한 학습 거동을 재현하려면 sqrt(abs(...)) 형태로 교체하는 것을 권장**하며, 이 경우 §4의 `w_phys=10.0` 비율도 원본과 일관되게 맞출 수 있다.

---

## 6. 적용 순서 (구현 시 권장 단계)

1. `compute_smoothness_loss_angle`, `compute_mass_loss`, `compute_collision_loss_v3`, `get_curriculum_weights`를 v9로 이식 (keepout/continuity/shape 함수는 이식하지 않거나 미사용 상태로 보류).
2. v9에 `parts_order_in_sections = {0: [0, 1, 2, 3, 4]}` 정의 — 실제 좌표상 파트 배치 순서와 일치하는지 시각적으로 재확인.
3. `run_training()` 내부의 단일 `loss = ((pred_mp - target_mp_val)/target_mp_val)**2`를 §4의 가중합 형태로 교체, `train_step`처럼 curriculum 분기 추가.
4. `history` 딕셔너리에 `l_smooth`, `l_mass`, `l_collision` 항목 추가해 기존 시각화(`visualize_results`)에 서브패널로 확장.
5. Phase 1 구간에서 phys-only 수렴이 v9 기존 결과(약 몇 % 오차 수준)와 비슷한지 먼저 검증한 뒤, Phase 2/3에서 형상·간섭 항이 개입했을 때 Mp 오차가 크게 튀지 않는지 확인.
6. keepout 항은 이번 범위에서 제외하되, `weights['w_keepout']=0.0`과 `keepout=None`을 남겨두어 이후 설계 제한 영역 도입 시 최소 수정으로 확장 가능하도록 준비.

---

## 7. 리스크 및 확인 필요 사항

- **단일 섹션에서 collision loss 유효성**: `compute_collision_loss_v3`는 섹션별로 파트 순서를 따라 인접 파트 쌍만 검사한다. v9의 5-part 구조(#00/#03/#06/#07/#08)가 실제로 순차적으로 겹치는 배치인지 좌표를 재확인해야 하며, 아니라면 `parts_order_in_sections`를 실제 형상에 맞게 조정해야 한다.
- **phys loss 형태 변경의 영향**: §5에서 언급한 sqrt vs square 방식 차이는 기존 v9 검증 결과(수렴 곡선, 그래디언트 정확도 비교)와 직접 비교 기준이 달라지므로, 변경 전/후 두 버전을 모두 돌려 비교하는 것을 권장.
- **curriculum_ratio 튜닝**: v9의 `max_epochs=181`은 원본(`max_epochs=500`)보다 짧다. 동일한 `[0.1, 0.5]` 비율을 쓰면 phase1이 약 18 epoch, phase3 전환이 약 90 epoch 시점으로 상대적으로 이른데, 물리 수렴이 충분히 이뤄지기 전에 형상 항이 개입할 위험이 있어 phase 경계를 조정하거나 max_epochs를 늘리는 것을 함께 고려해야 한다.
