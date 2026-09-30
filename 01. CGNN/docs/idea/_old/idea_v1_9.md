# idea_v1_9.md — 구조적 붕괴(스파이크) 해소를 위한 v1.9 설계 아이디어

작성일: 2026-08-09 (`/synod idea` 세션, Gemini flash conf92→95→98 + OpenAI gpt4o conf85→85→75)
근거: `docs/review/review_v1_8.md`(§1~9, 코드 라인 수준으로 확정된 9가지 붕괴 원인)
범위: `AI_design_v1_8.py`(메인, 로컬 오버라이드 상수·함수 패턴 유지), `uni-section/code/uni_section_v19.py`
(원칙적으로 최소 수정 — 이 파일은 v1.4~v1.8 내내 "거의 불변" 관례를 따랐다)

**세션 개요**: Solver 라운드에서 Gemini는 `compute_smoothness_lse_v2()`를 좌표를
`(17, nodes_per_section, 2)`로 단순 reshape하는 방식으로 제안했으나, Critic 라운드에서
Gemini 자신과 OpenAI 둘 다 이것이 `neighbor_map`의 실제 자료구조(부분집합 flat 인덱스)와
맞지 않음을 지적해 scatter/그룹핑 기반으로 정정했다. Defense 라운드에서 Gemini가 외부
의존성 없는 순수 PyTorch 구현(17섹션 Python 루프 + STE)을 제시했고, OpenAI가 Prosecutor로서
`W_SMOOTH_LSE`를 5~10까지 올리는 것은 `w_mass`/`w_area_floor`(10.0)와 같은 자릿수가 되어
Mp 수렴을 불안정하게 할 위험이 있다고 반박해 3.0 수준의 보수적 시작값을 역제안했다(채택).
아래 §1~§5는 이 논쟁의 최종 결과이며, Gemini가 제시한 코드 중 `neighbor_map` 필드명이
실제 코드(`left_idx`/`right_idx`, `(M,K)` 패딩 아님)와 다른 부분은 Claude가 Judge로서
직접 정정했다(§부록 참고).

---

## §1. [P0] `compute_smoothness_lse_v2()` — LSE 통합 재설계 (review §1·②·③ 동시 해소)

**해소 원인**: review_v1_8.md ①(가중치 불균형), ②(전역 단일 스칼라 whack-a-mole), ③(DELETED
Ghost Gradient Hijacking).

**설계**: 기존 `compute_smoothness_lse()`(AI_design_v1_8.py, v1.7 도입)를 대체하는 신규 함수를
같은 파일에 로컬 오버라이드로 추가한다(`uni_section_v19.py`는 건드리지 않음 — 이 손실 자체가
이미 v1.7부터 `AI_design_v1_8.py`에만 존재하는 "보강 손실"이므로 관례상 자연스럽다).

- **섹션별 그룹핑**: `neighbor_map['node_idx']`가 가리키는 노드들의 `section_id = x[node_idx, 5]`를
  조회해, 17개 섹션에 대해 **명시적 Python 루프**로 섹션별 LogSumExp를 각각 계산한 뒤 평균한다
  (`torch_scatter`는 이 프로젝트의 의존성에 없고, 설치 이식성 문제가 있어 기각 — Defense 라운드
  결론). 17회 루프는 노드 수(510개) 규모에서 비용이 무시할 만하다.
- **노드 단위 마스킹**: `t_final < eps`(예: `eps=0.1mm`)인 노드는 "중심 노드"든 "이웃 노드"든
  LSE 대상에서 제외한다. **섹션 전체를 마스킹하지 않는다** — 연속 파트(0/1/2)는 항상 alive이므로
  섹션 단위 마스킹은 같은 섹션의 살아있는 연속 파트 노드까지 잘못 제외시킬 위험이 있다(Critic
  라운드에서 두 모델 모두 이 구분을 확정).

```python
# AI_design_v1_8.py — compute_smoothness_lse()를 대체 (v1.7 함수는 v1.9에서 제거)
def compute_smoothness_lse_v2(x, new_coords, neighbor_map, t_final, num_sections=NUM_SECTIONS,
                               eps=0.1, temperature=SMOOTH_LSE_TEMPERATURE):
    """[v1.9] compute_smoothness_lse()(v1.7)를 대체한다.
    변경점: (1) 17섹션 전체를 하나의 스칼라로 묶던 것을 섹션별로 분리해 whack-a-mole 방지(review §2),
    (2) t_final 기준 노드 단위(섹션 단위 아님) 마스킹으로 DELETED 노드의 Ghost Gradient 차단(review §3).
    neighbor_map은 build_static_neighbor_map()이 반환하는 그대로(node_idx/left_idx/right_idx,
    이미 정확히 2개의 물리 이웃을 가진 노드만 필터링된 상태 — 패딩/-1 처리 불필요)."""
    node_idx  = neighbor_map['node_idx']
    left_idx  = neighbor_map['left_idx']
    right_idx = neighbor_map['right_idx']
    if node_idx.numel() == 0:
        return new_coords.sum() * 0.0

    t_flat = t_final.squeeze(-1) if t_final.dim() > 1 else t_final
    # 노드 단위 활성 마스크: 중심/좌/우 셋 다 살아있어야 이 쌍을 사용(review §3 대응)
    active = (t_flat[node_idx] > eps) & (t_flat[left_idx] > eps) & (t_flat[right_idx] > eps)
    if not active.any():
        return new_coords.sum() * 0.0

    node_idx, left_idx, right_idx = node_idx[active], left_idx[active], right_idx[active]
    section_ids = x[node_idx, 5].long()

    p_i, p_left, p_right = new_coords[node_idx], new_coords[left_idx], new_coords[right_idx]
    deviation = p_i - 0.5 * (p_left + p_right)
    sq_deviation = torch.sum(deviation ** 2, dim=-1)   # (M_active,)

    section_lse = []
    for sec in range(num_sections):
        sec_mask = section_ids == sec
        if not sec_mask.any():
            continue
        d = sq_deviation[sec_mask]
        section_lse.append(temperature * torch.logsumexp(d / temperature, dim=0))
    if not section_lse:
        return new_coords.sum() * 0.0
    return torch.stack(section_lse).mean()
```

- **가중치**: `W_SMOOTH_LSE`를 기존 `0.25`에서 **`3.0`으로 상향**한다(Gemini 원안의 5~10은
  Prosecution 라운드에서 OpenAI가 "`w_mass`/`w_area_floor`(10.0)와 같은 자릿수가 되면 Mp 수렴
  자체를 방해할 위험"을 근거로 반박해 채택되지 않음 — 3.0은 물리 손실(10.0)의 약 30% 수준으로,
  기존 0.25(2.5%) 대비 12배 강화하면서도 Mp 압력을 압도하지 않는 절충값). §6 검증 계획에서
  ablation으로 재보정할 것을 전제로 한 시작값이다.
- **기존 `compute_smoothness_loss_angle()`(불변 파일, mean 희석)와의 관계**: 이 함수는 그대로
  두되(review §1 표에서 `contrib_smooth`는 유지), LSE 항이 실질적 억제력을 갖게 되므로 상대적
  비중이 자연히 낮아진다. 별도 수정은 이번 버전 범위 밖(불변 파일 원칙 유지).

---

## §2. [P0 동급] Stage 4 이진화 완화 — 점진적 게이트 보간 + STE (review §5 해소)

**해소 원인**: review_v1_8.md ⑤(Stage4 hard-flip 불연속 충격 + optimizer 모멘텀 초기화).

**설계**: `run_training_multi()`의 Stage 4 진입부(`model.log_alpha_candidates[k, c] = 5.0 if
hard_exists[k, c] else -5.0`로 단일 스텝 강제하던 부분)를 다음으로 교체한다.

```python
# AI_design_v1_8.py — Stage 4 진입부 수정
STAGE4_GATE_WARMUP_EPOCHS = 50   # [v1.9] hard-flip을 이 기간에 걸쳐 선형 보간으로 완화

# (as-is) model.log_alpha_candidates[k, c] = 5.0 if hard_exists[k, c] else -5.0
#          model.log_alpha_candidates.requires_grad_(False)
# (to-be) 목표값은 동일(±5.0)하되, 즉시 대입하지 않고 매 epoch 선형 보간으로 접근시킨다.
la_init = model.log_alpha_candidates.detach().clone()               # [v1.9] 보간 시작점
la_target = torch.where(hard_exists, torch.tensor(5.0), torch.tensor(-5.0))
# requires_grad_(False)는 보간이 끝난 뒤(stage4_end 직전)에만 건다 — 보간 도중에는 여전히
# 학습 가능한 파라미터로 두어, 이 구간에도 다른 손실(collision 등)의 피드백을 받을 수 있게 한다.

# Stage4 for-loop 내부, 매 epoch 시작 시:
progress = min(1.0, (ep - stage4_start) / STAGE4_GATE_WARMUP_EPOCHS)
with torch.no_grad():
    model.log_alpha_candidates.copy_((1 - progress) * la_init + progress * la_target)
if progress >= 1.0 and model.log_alpha_candidates.requires_grad:
    model.log_alpha_candidates.requires_grad_(False)   # 보간 완료 후 동결(§5.7 종료 assert와 호환)
```

- **optimizer 모멘텀 보존**: Stage4 진입 시 `optimizer = optim.AdamW(...)`로 완전히 새로 만들던
  것을, `main_params`/`thick_params`에 대해서는 **기존 optimizer의 `state`를 재사용**하도록
  수정한다(파라미터 그룹만 재구성하고 `optimizer.state`는 이전 옵티마이저의 것을 복사).
  ```python
  old_state = {id(p): optimizer.state[p] for p in main_params + thick_params if p in optimizer.state}
  optimizer = optim.AdamW([...], lr=lr * STAGE4_LR_SCALE, weight_decay=1e-4)
  for p in main_params + thick_params:
      if id(p) in old_state:
          optimizer.state[p] = old_state[id(p)]
  ```
- **STE는 불필요**: Critic/Defense 라운드에서 Straight-Through Estimator 적용이 제안됐으나,
  Judge 검토 결과 이 프로젝트의 `compute_gates_multi()`는 이미 `usv18.compute_gates()`의
  HardConcrete **soft** 샘플링을 그대로 사용하고 있고(§5 진입 전까지), Stage4에서
  `log_alpha_candidates`를 선형 보간하는 위 방식은 파라미터 자체를 부드럽게 이동시키므로 forward
  경로의 `compute_gates_multi()`가 항상 미분 가능한 soft 값을 계산한다 — 이산적 hard sampling을
  강제로 우회하는 STE 트릭이 필요 없다. STE는 이 설계에서 채택하지 않는다(불필요한 복잡도로 기각).

---

## §3. [P1] `compute_shape_continuity_loss()` — 인덱스 대응 기반으로 교체 (review §6 해소)

**해소 원인**: review_v1_8.md ⑥(Chamfer-style 최근접-이웃 매칭이 스파이크를 놓침).

**근거**: 17섹션 모두 `build_bpillar_section()`을 1회씩 호출해 동일 토폴로지를 스케일링만 다르게
사용하므로(`build_bpillar_17section()`), 인접 섹션 간 같은 로컬 인덱스(같은 노드 순번)는 항상
같은 물리적 위치(예: "part 0의 5번째 노드")를 가리킨다 — `torch.cdist` + `min` 없이 인덱스로
직접 대응시킬 수 있다.

```python
# AI_design_v1_8.py — compute_shape_continuity_loss() 교체
def compute_shape_continuity_loss_v2(new_coords, section_ids, part_ids, threshold=2.0):
    """[v1.9] Chamfer-style min-distance 매칭(review §6)을 인덱스 대응 방식으로 교체.
    두 섹션이 build_bpillar_section()의 동일 토폴로지를 스케일링만 다르게 쓴다는 사실(로컬 노드
    순번이 섹션 간 대응됨)을 이용 — 노드를 (section, part, local_rank) 튜플로 재정렬해 1:1 비교."""
    device = new_coords.device
    total = torch.tensor(0.0, device=device)
    n_terms = 0
    unique_secs = torch.unique(section_ids).long().sort()[0]

    for i in range(len(unique_secs) - 1):
        k_a, k_b = int(unique_secs[i].item()), int(unique_secs[i + 1].item())
        s_a, s_b = scale_factor(k_a), scale_factor(k_b)
        for pid in torch.unique(part_ids):
            pid_int = int(pid.item())
            # 각 섹션 내부에서 노드가 build_bpillar_section() 생성 순서 그대로 유지된다는 전제
            # (build_bpillar_17section()이 노드를 이어붙이기만 하고 재정렬하지 않음 — 코드 확인됨)
            idx_a = ((section_ids == k_a) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            idx_b = ((section_ids == k_b) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            if idx_a.numel() == 0 or idx_b.numel() == 0 or idx_a.numel() != idx_b.numel():
                continue   # 노드 수가 다르면(이 프로젝트에서는 발생하지 않음) 안전하게 스킵
            ca = new_coords[idx_a] / s_a
            cb = new_coords[idx_b] / s_b
            dist = torch.norm(ca - cb, dim=-1)          # 대응 인덱스끼리 직접 거리(더 이상 min 아님)
            violation = torch.clamp(dist - threshold, min=0.0)
            total = total + torch.mean(violation ** 2)
            n_terms += 1
    if n_terms > 0:
        total = total / n_terms
    return total
```

- **트레이드오프**: `idx_a.numel() != idx_b.numel()`인 경우(현재 코드베이스에서는 발생하지 않지만,
  향후 섹션별로 노드 수가 달라지는 변경이 생기면) 이 함수는 조용히 스킵한다 — 안전장치이자
  잠재적 사각지대이므로, §6 검증 계획에 "노드 수 불일치 assert 추가"를 포함한다.
- **적용 범위**: candidate 파트(3/4)에도 동일하게 적용된다. DELETED 확정된 run 경계에서는 두께가
  0이라 형상 오차가 있어도 제조에 영향이 없지만, 이 손실 자체는 좌표를 다루므로 §1의 `t_final`
  마스킹과 별개로 그대로 둔다(연속성 손실은 원래도 "제조 무관 파트"에 걸려도 무해하다는 것이
  기존 docstring의 명시적 설계였음 — 이 전제를 유지).

---

## §4. [P1 병행] `l_phys_total` / Collision 손실의 평균 희석 방지 (review §7·⑧·⑨ 해소)

**해소 원인**: review_v1_8.md ⑦(`l_phys_total`의 섹션 간 mean 희석), ⑨(collision 손실의 노드 단위
평균 희석). ⑧(`z_gate_part_avg`의 부당한 희석)은 별도 §5에서 다룬다.

### §4.1 `l_phys_total` — Top-K 풀링으로 교체 (기존 §1의 Order 손실 패턴 재사용)

review_v1_8.md §1에서 이미 `compute_mesh_order_loss()`에 적용된 Top-K 풀링 패턴을 그대로
재사용한다 — **새 메커니즘을 발명하지 않고, 이 프로젝트에 이미 검증된 패턴을 다른 위치에
적용**하는 것이 최소 변경 원칙에 부합한다(OpenAI가 Solver 라운드에서 제안한 softmax 가중 평균은
"이미 있는 Top-K 패턴과 다른 새 메커니즘을 도입한다"는 이유로 기각 — 하나의 프로젝트에 두 가지
국소-이상치 억제 전략이 공존하면 유지보수 부담이 커진다).

```python
# AI_design_v1_8.py — train_step_multi() 내 l_phys_total 계산부 교체
# (as-is) l_phys_total = torch.stack(l_phys_terms).mean()
# (to-be)
PHYS_TOPK_FRACTION = 0.3   # [v1.9] 17섹션 중 최악 30%(~5개 섹션)에 그래디언트 집중

l_phys_stack = torch.stack(l_phys_terms)                      # (17,)
k_phys = max(1, int(l_phys_stack.numel() * PHYS_TOPK_FRACTION))
l_phys_topk, _ = torch.topk(l_phys_stack, k=k_phys, largest=True)
l_phys_total = torch.mean(l_phys_topk)
```

- `top_k_fraction=0.3`(17섹션 중 5개)은 `compute_mesh_order_loss`의 `0.02`보다 훨씬 크게 잡았다
  — 물리 엣지(~2992개) 중 2%(~60개)와 섹션(17개) 중 2%(<1개)는 성격이 다르므로, "섹션 수가 원래
  적다"는 점을 감안해 절대 개수 기준(최소 3~5개 섹션은 항상 반영)으로 시작하고 §6에서 재보정한다.

### §4.2 Collision 손실 — 동일 패턴을 `uni_section_v19.py`에 적용

`compute_collision_loss_v5`/`compute_collision_penalty_unclamped`의 방향별 위반량 집계
(`violation.sum() / valid.sum()`, 평균)를 Top-K 평균으로 교체한다. **이 함수들은
`uni_section_v19.py`(compute_collision_loss_v5)와 `AI_design_v1_8.py`(compute_collision_penalty_unclamped)
에 나뉘어 있다** — 전자는 "거의 불변" 원칙 때문에 최소 수정만 적용한다.

```python
# uni_section_v19.py — compute_collision_loss_v5() 내부, 방향별 loss_dir 계산부만 수정
# (as-is) loss_dir = (violation ** 2).sum() / valid.float().sum()
# (to-be) 위반 노드가 극소수(1~2개)인 국소 관통을 살리기 위해 Top-K(최대 violating 30%, 최소 1개)로 교체
v = (violation ** 2)[valid]
if v.numel() > 0:
    k_col = max(1, int(v.numel() * 0.3))
    top_v, _ = torch.topk(v, k=k_col, largest=True)
    loss_dir = top_v.mean()
else:
    loss_dir = torch.tensor(0.0, device=violation.device)
```
동일한 수정을 `AI_design_v1_8.py`의 `compute_collision_penalty_unclamped()`에도 적용한다(구조가
거의 동일하므로 코드 중복 — 이번 버전에서는 두 함수 통합을 시도하지 않는다, §6 향후 과제로 남김).

- **트레이드오프**: 두 함수 모두 이제 이 패턴을 반복하게 되어 코드 중복이 늘어난다. 향후 버전에서
  "국소 이상치를 Top-K로 잡는" 공용 헬퍼(`topk_mean(tensor, fraction, min_k=1)`)로 리팩터링할
  것을 §6에 남긴다 — 이번 v1.9는 review_v1_8.md의 진단을 빠르게 해소하는 것이 우선이므로 리팩터링은
  범위 밖으로 둔다.

---

## §5. [검토, 낮은 난이도] `z_gate_part_avg`를 생존 섹션 평균으로 정밀화 (review §8 해소)

**해소 원인**: review_v1_8.md ⑧(`z_gate.mean(dim=0)`이 17섹션 전체 평균이라, candidate 파트가
7/17에서만 생존하면 생존 섹션에서도 collision 억제력이 약 60% 깎임).

```python
# AI_design_v1_8.py — train_step_multi() 내 z_gate_part_avg 계산부 수정
# (as-is) z_gate_part_avg = z_gate.mean(dim=0)  # (5,)
# (to-be)
alive_mask = torch.ones_like(z_gate, dtype=torch.bool)          # 연속 파트는 항상 전부 alive 취급
cand_idx_t = torch.tensor(CANDIDATE_PARTS, device=z_gate.device)
if pruning_state is not None:
    alive_cand = (pruning_state['state'] != STATE_DELETED).to(z_gate.device)  # (17, n_cand)
    alive_mask[:, cand_idx_t] = alive_cand
# 생존 섹션만의 평균 (분모 0 방지를 위해 clamp)
z_gate_part_avg = (z_gate * alive_mask).sum(dim=0) / alive_mask.sum(dim=0).clamp(min=1)
```

- **효과**: DELETED 확정 섹션은 애초에 collision 계산에서 제외되므로, 생존 섹션의 z_gate 평균이
  1.0에 가까워져(생존 섹션들의 z_gate가 이미 대부분 높다는 전제) collision 억제력이 정상화된다.
- **부작용**: `pruning_state`가 없는 호출 경로(Stage 0 dry-run 등)에서는 기존과 동일하게 전체
  평균으로 폴백한다 — 위 코드는 이미 `if pruning_state is not None` 가드로 이를 보장한다.

---

## §6. 검증 계획 (review_v1_8.md 권장사항 4·9를 v1.9에 맞게 구체화)

1. **로깅 강화(최우선, 코드 변경과 무관하게 먼저 적용 가능)**: `l_smooth_lse_v2`(섹션별 값 중
   max/mean), `l_phys_total`(Top-K 적용 전/후 비교용으로 mean도 병기), `l_col_aux` 등을 epoch별로
   CSV 또는 `history` dict에 남기고 **콘솔 출력뿐 아니라 파일로 저장**한다 — v1.5~v1.8 리뷰가 4번
   연속 "저장된 로그가 없다"는 한계를 지적했다.
2. **Ablation**: (a) §1만 적용, (b) §1+§2, (c) §1+§2+§3+§4+§5 전체 — 3가지 조합으로 최소 200 epoch
   스모크 테스트를 돌려 `AI_design_v1_8_3d.html`의 스파이크 잔존 여부를 육안 비교한다.
3. **가중치 재보정**: `W_SMOOTH_LSE=3.0`, `PHYS_TOPK_FRACTION=0.3`은 모두 시작값이다 — Mp 수렴
   속도(`mp_rel_err` 곡선)가 v1.8 대비 눈에 띄게 느려지면 단계적으로 낮춘다.
4. **Stage4 보간(§2) 검증**: `STAGE4_GATE_WARMUP_EPOCHS=50` 적용 후 `§5.7 종료 assert`(z_gate가
   정확히 0/1로 포화하는지)가 여전히 통과하는지 반드시 확인한다 — 보간이 끝나기 전에 루프가
   종료되지 않도록 `STAGE4_EPOCHS >= STAGE4_GATE_WARMUP_EPOCHS`(200 >= 50, 현재 값으로 이미 충족)
   assert를 추가한다.

---

## 부록 — Solver/Defense 라운드에서 나왔으나 기각되거나 수정된 제안

- **Gemini Solver 원안의 `(17, nodes_per_section, 2)` reshape**: Critic 라운드에서 Gemini·OpenAI
  둘 다 `neighbor_map`이 이미 필터링된 flat 부분집합 인덱스임을 근거로 반증 — §1의 섹션별 Python
  루프 방식으로 교체.
- **Gemini Defense의 `neighbor_idx (M, K)` 패딩 형식**: 실제 `build_static_neighbor_map()`은
  `left_idx`/`right_idx` 두 개의 분리된 `(M,)` 텐서를 반환하며 이미 "정확히 2개 이웃" 조건으로
  필터링돼 있어 패딩(`-1`)이 필요 없다 — Judge(Claude)가 §1 코드에서 실제 형식에 맞게 수정.
- **Gemini Defense의 Straight-Through Estimator(STE) 도입**: §2에서 "파라미터 자체를 선형 보간하면
  forward가 항상 soft 값을 계산하므로 불필요"라는 이유로 Judge가 기각 — 복잡도 대비 이득이 없음.
- **OpenAI Solver 원안의 `dynamic_weight_adjustment()`(epoch에 따라 w_mass를 낮추고 w_smooth를
  올리는 시간 가변 스케줄)**: 이미 `get_curriculum_weights_v10()`이 `s_phys`/`s_smooth`로 유사한
  역할을 하고 있어(review_v1_8.md 2차 검토에서 확인 — s_smooth는 epoch 350 이후 1.0 고정) 중복
  메커니즘으로 판단해 채택하지 않음.
- **OpenAI Prosecution의 `W_SMOOTH_LSE=3` 제안**: 유일하게 채택된 Prosecution 주장 — Gemini의
  5~10 상향안보다 보수적인 값으로 §1에 반영.
- **Gemini Solver 원안의 `l_phys` softmax 가중 집계**: §4.1에서 이미 검증된 Top-K 패턴 재사용을
  이유로 기각(새 메커니즘 중복 도입 방지).
