# command_v1_9.md — 구조적 붕괴(스파이크) 해소 5종 세트 구현 지시서

작성일: 2026-08-09 (`/synod design` 세션, Gemini flash conf98→100→100 + OpenAI o3 conf92→84→71)
참조: `docs/idea/idea_v1_9.md`(설계 근거, 신뢰도 검증됨), `docs/review/review_v1_8.md`(§1~9, 원인 진단),
`AI_design_v1_8.py`(기반 코드), `uni-section/code/uni_section_v19.py`(**collision Top-K만 수정 —
v20으로 갱신**)

> **앵커 규칙**: 본 문서의 위치 지정은 **함수/변수 이름 기준**이다. 라인 번호는 참고용이며
> 코드 변경으로 어긋날 수 있다.

> **범위 확장 승인**: `uni_section_v19.py`를 `uni_section_v20.py`로 복사해 `compute_collision_loss_v5()`
> 단 하나만 고친다(§4-C) — command_v1_8.md의 v18→v19 전례와 동일한 패턴. **기존 `uni_section_v19.py`
> 파일은 삭제하지 않고 그대로 둔다**(과거 버전 참조/재현성 보존 — v1.1~v1.8 전 버전이 이 방식을
> 따랐다).

> **불변 규칙(계속 유지)**: `SMOOTH_LSE_TEMPERATURE=1.5`(v1.8에서 물리적으로 재보정된 값)와
> `compute_mesh_order_loss()`의 `top_k_fraction=0.02`는 이번 버전 범위 밖이며 **절대 변경하지
> 않는다** — Solver/Critic 라운드에서 두 모델 모두 이 두 상수의 "기존 값"을 잘못 인용(0.1/0.7,
> 0.2)했으나 실제 값은 각각 1.5, 0.02이다(Judge가 실제 코드로 재확인).

---

## §0. 목표 및 산출물

`AI_design_v1_8.py`를 복사해 **`AI_design_v1_9.py`**를, `uni-section/code/uni_section_v19.py`를
복사해 **`uni-section/code/uni_section_v20.py`**를 만든다(프로젝트 관례 — command_v1_8.md §0과
동일한 파일 버전업 패턴).

```bash
cp AI_design_v1_8.py AI_design_v1_9.py
cp uni-section/code/uni_section_v19.py uni-section/code/uni_section_v20.py
```

| idea_v1_9.md 조항 | 해소 원인(review_v1_8.md) | 대상 파일 |
|---|---|---|
| §1 `compute_smoothness_lse_v2()` | ①②③(가중치 불균형/전역 LSE/Ghost Gradient) | `AI_design_v1_9.py` |
| §2 Stage4 게이트 점진 보간 + optimizer state 이전 | ⑤(hard-flip 불연속) | `AI_design_v1_9.py` |
| §3 `compute_shape_continuity_loss_v2()` | ⑥(Chamfer 매칭 루프홀) | `AI_design_v1_9.py` |
| §4 `l_phys_total`/collision Top-K | ⑦⑨(평균 희석) | `AI_design_v1_9.py` + `uni_section_v20.py` |
| §5 `z_gate_part_avg` 생존 섹션 평균 | ⑧(z_gate 희석) | `AI_design_v1_9.py` |

**적용 순서·원자성(Critic 라운드 수렴)**: §1과 §2는 **반드시 같은 커밋**으로 묶어 적용한다.
`W_SMOOTH_LSE`가 0.25→3.0으로 12배 상향되어 손실 스케일 자체가 바뀌는데, Stage4의 기존
"한 스텝 강제 이진화"가 그대로 남아 있으면 스케일 변화와 이산 점프가 겹쳐 그래디언트가 불안정해질
위험이 있다(Solver 라운드 공통 지적 — 단, 구체적 발생 확률 수치는 이 세션에서 실행된 적 없는
근거 없는 인용이었으므로 배제하고, 메커니즘적 근거만 채택). §3~§5는 §1·§2와 독립적이므로 순서
무관하나, 이번 버전에서는 함께 적용하고 함께 검증한다(리뷰 문서가 이미 9개 원인을 하나의 버전
범위로 묶어 진단했으므로 분리 적용의 실익이 적다 — idea_v1_9.md 부록 참고).

이 커밋 규율은 이 저장소에 CI가 없으므로(사실 확인됨) **문서상 MANDATORY 표기 + 코드 리뷰 시
확인**으로 강제한다(GitHub Actions 등 실재하지 않는 CI 게이트를 전제하지 않는다).

---

## §1. `compute_smoothness_lse_v2()` — 섹션별 LSE + 노드 단위 마스킹 (AI_design_v1_9.py)

### §1.1 신규 함수 추가, 기존 함수 제거

**대상**: `AI_design_v1_9.py`의 `compute_smoothness_lse()`(v1.7 도입, `build_static_neighbor_map()`
바로 아래).

```python
# 변경 전
def compute_smoothness_lse(new_coords, neighbor_map, temperature=SMOOTH_LSE_TEMPERATURE):
    """[v1.7 §1] 정적 이웃 쌍에 대한 Laplacian형 편차를 LogSumExp로 집계.
    ...(DELETED 노드도 그대로 포함한다는 독스트링)..."""
    node_idx, left_idx, right_idx = (neighbor_map['node_idx'], neighbor_map['left_idx'],
                                      neighbor_map['right_idx'])
    if node_idx.numel() == 0:
        return new_coords.sum() * 0.0
    p_i, p_left, p_right = new_coords[node_idx], new_coords[left_idx], new_coords[right_idx]
    deviation = p_i - 0.5 * (p_left + p_right)
    sq_deviation = torch.sum(deviation ** 2, dim=-1)
    return temperature * torch.logsumexp(sq_deviation / temperature, dim=0)

# 변경 후 — 함수를 완전히 대체한다(레거시 래퍼 없음, 프로젝트에 그런 전례가 없다).
def compute_smoothness_lse_v2(x, new_coords, neighbor_map, t_final, num_sections=NUM_SECTIONS,
                               eps=0.1, temperature=SMOOTH_LSE_TEMPERATURE):
    """[v1.9 §1] compute_smoothness_lse()(v1.7)를 대체한다. 두 가지를 동시에 고친다:
    (1) 17섹션 전체를 하나의 스칼라로 묶던 것을 섹션별로 분리해 whack-a-mole 방지
        (review_v1_8.md §2).
    (2) t_final 기준 노드 단위(섹션 단위 아님) 마스킹으로 DELETED 노드의 Ghost Gradient
        차단(review_v1_8.md §3). 연속 파트(0/1/2)는 항상 alive이므로 섹션 전체를 죽은
        것으로 마스킹하면 안 된다 — 반드시 t_final 기준 노드별 판정.
    neighbor_map은 build_static_neighbor_map()이 반환하는 그대로(node_idx/left_idx/right_idx,
    이미 정확히 2개의 물리 이웃을 가진 노드만 필터링된 상태 — 패딩/-1 처리 불필요)."""
    node_idx  = neighbor_map['node_idx']
    left_idx  = neighbor_map['left_idx']
    right_idx = neighbor_map['right_idx']
    if node_idx.numel() == 0:
        return new_coords.sum() * 0.0

    t_flat = t_final.squeeze(-1) if t_final.dim() > 1 else t_final
    active = (t_flat[node_idx] > eps) & (t_flat[left_idx] > eps) & (t_flat[right_idx] > eps)
    if not active.any():
        return new_coords.sum() * 0.0

    node_idx, left_idx, right_idx = node_idx[active], left_idx[active], right_idx[active]
    section_ids = x[node_idx, 5].long()

    p_i, p_left, p_right = new_coords[node_idx], new_coords[left_idx], new_coords[right_idx]
    deviation = p_i - 0.5 * (p_left + p_right)
    sq_deviation = torch.sum(deviation ** 2, dim=-1)

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

**호출부 변경**: `train_step_multi()` 내부(§4의 `contrib_smooth_lse` 계산부 바로 위).

```python
# 변경 전
l_smooth_lse = compute_smoothness_lse(new_coords, neighbor_map)

# 변경 후
l_smooth_lse = compute_smoothness_lse_v2(x, new_coords, neighbor_map, t_final)
```

`t_final`은 `train_step_multi()`가 이미 `model(...)` 호출로 계산해 갖고 있는 변수를 그대로 넘긴다
(신규 인자 추가 없음).

### §1.2 가중치 상향

**대상**: `AI_design_v1_9.py` 상단 `W_SMOOTH_LSE` 상수(§3 신규 상수 블록 참고).

```python
# 변경 전
W_SMOOTH_LSE = 0.25

# 변경 후
W_SMOOTH_LSE = 3.0   # [v1.9 §1] 0.25 → 3.0. Prosecution 라운드에서 5~10 상향안이 w_mass/
                      # w_area_floor(10.0)와 같은 자릿수가 되어 Mp 수렴을 방해할 위험이 있다는
                      # 반론으로 채택되지 않고, 물리 손실의 약 30% 수준으로 절충됐다(idea_v1_9.md).
                      # §7 검증 계획에서 실측 후 재보정 필수인 시작값이다.
```

**주의**: 기존 `compute_smoothness_loss_angle()`(불변 파일, mean 희석)와 `contrib_smooth`
(`weights['w_smooth']=0.5`)는 이번 버전에서 건드리지 않는다 — LSE 항이 실질적 억제력을 갖게 되면
상대적 비중이 자연히 낮아진다(idea_v1_9.md §1 범위 확정 사항).

---

## §2. Stage 4 게이트 점진 보간 + Optimizer State 이전 (AI_design_v1_9.py)

### §2.1 신규 상수

```python
STAGE4_GATE_WARMUP_EPOCHS = 50   # [v1.9 §2] hard-flip을 이 기간에 걸쳐 선형 보간으로 완화
assert STAGE4_EPOCHS >= STAGE4_GATE_WARMUP_EPOCHS, \
    "STAGE4_EPOCHS(200)가 WARMUP(50)보다 작으면 보간이 끝나기 전에 Stage4 루프가 종료된다"
```

이 assert는 모듈 레벨(파일 하단, 기존 §7.3 스모크 테스트 블록 근처)에 추가한다 — 값이 바뀌어도
항상 검사되도록.

### §2.2 Stage 4 진입부 수정

**대상**: `run_training_multi()`의 `if enter_stage4:` 블록(`model.log_alpha_candidates[k, c] = 5.0
if hard_exists[k, c] else -5.0`가 있는 위치).

```python
# 변경 전
for k in range(NUM_SECTIONS):
    for c in range(len(CANDIDATE_PARTS)):
        model.log_alpha_candidates[k, c] = 5.0 if hard_exists[k, c] else -5.0
model.log_alpha_candidates.requires_grad_(False)

seg_ids = compute_segment_ids(pruning_state, hard_exists=hard_exists).to(device)

optimizer = optim.AdamW(
    [{'params': main_params,  'name': 'main',              'lr': lr * STAGE4_LR_SCALE},
     {'params': thick_params, 'name': 'thickness_decoder', 'lr': lr * STAGE4_LR_SCALE}],
    lr=lr * STAGE4_LR_SCALE, weight_decay=1e-4)

# 변경 후
# [v1.9 §2] 단일 스텝 강제 이진화 대신 목표값(±5.0)만 정하고, 실제 대입은 Stage4 루프 안에서
# STAGE4_GATE_WARMUP_EPOCHS에 걸쳐 매 epoch 선형 보간한다. requires_grad는 보간이 끝난 뒤에만 끈다.
la_init_v19 = model.log_alpha_candidates.detach().clone()
la_target_v19 = torch.where(hard_exists.to(device), torch.tensor(5.0, device=device),
                             torch.tensor(-5.0, device=device))
# requires_grad는 여기서 끄지 않는다 — 보간 도중에도 다른 손실(collision 등)의 피드백을 받도록
# 열어 둔다. compute_gates_multi()는 log_alpha 값을 그대로 sigmoid에 넣으므로 보간 중에도
# 항상 미분 가능한 soft 값을 계산한다(STE 불필요 — idea_v1_9.md 부록에서 기각된 대안).

seg_ids = compute_segment_ids(pruning_state, hard_exists=hard_exists).to(device)

# [v1.9 §2] optimizer state 이전: 이 시점까지 model의 파라미터 객체(main_params/thick_params)는
# 한 번도 재생성되지 않았다(Stage4가 모델을 다시 만들지 않고 optimizer만 새로 만듦) — 따라서
# 기존 optimizer.state는 여전히 "같은 Parameter 객체"를 키로 갖고 있다. id()나 이름 매칭 같은
# 간접 수단이 필요 없다: 파라미터 객체 자체가 dict key로 재사용 가능하다.
old_state_by_param = {p: optimizer.state[p] for p in main_params + thick_params
                       if p in optimizer.state}

optimizer = optim.AdamW(
    [{'params': main_params,  'name': 'main',              'lr': lr * STAGE4_LR_SCALE},
     {'params': thick_params, 'name': 'thickness_decoder', 'lr': lr * STAGE4_LR_SCALE}],
    lr=lr * STAGE4_LR_SCALE, weight_decay=1e-4)

try:
    for p, state in old_state_by_param.items():
        optimizer.state[p] = state
    print(f"[Stage4] optimizer 모멘텀 이전 완료: {len(old_state_by_param)}개 파라미터")
except Exception as e:   # [v1.9 §2] 안전장치 — 실패해도 학습이 죽지 않고 모멘텀 없이 계속 진행
    print(f"[Stage4][WARNING] optimizer state 이전 실패({e}) — 모멘텀 없이 새로 시작합니다")
```

### §2.3 Stage 4 for-loop 내부 수정

**대상**: `for ep in range(stage4_start, stage4_end):` 루프 본문 시작 부분(`train_step_multi()`
호출 이전).

```python
# 변경 후 — 루프 본문 맨 앞에 삽입
progress = min(1.0, (ep - stage4_start) / STAGE4_GATE_WARMUP_EPOCHS)
with torch.no_grad():
    model.log_alpha_candidates.copy_((1.0 - progress) * la_init_v19 + progress * la_target_v19)
if progress >= 1.0 and model.log_alpha_candidates.requires_grad:
    model.log_alpha_candidates.requires_grad_(False)   # 보간 완료 후에만 동결 — §5.7 종료 assert와 호환
```

**§5.7 종료 assert 호환성**: 기존 코드의 `assert torch.allclose(zc_chk, hard_exists.float(),
atol=1e-6)`는 그대로 유지한다 — `STAGE4_EPOCHS=200 >= WARMUP=50`이 보장되므로, 루프가 끝나는
시점(`ep == stage4_end - 1`)에는 항상 `progress >= 1.0`이라 `log_alpha_candidates`가 이미
`la_target_v19`(정확히 ±5.0)와 동일하다 — z_gate가 0/1로 포화한다는 기존 assert는 그대로 통과한다.

---

## §3. `compute_shape_continuity_loss_v2()` — 인덱스 대응 기반 (AI_design_v1_9.py)

**대상**: `compute_shape_continuity_loss()`(`SECTION 3` 헤더 아래).

```python
# 변경 전
def compute_shape_continuity_loss(new_coords, section_ids, part_ids, threshold=2.0):
    """... torch.cdist(ca, cb, p=2) + torch.min(dist, dim=1) 기반 최근접-이웃 매칭 ..."""
    ...

# 변경 후
def compute_shape_continuity_loss_v2(new_coords, section_ids, part_ids, threshold=2.0):
    """[v1.9 §3] compute_shape_continuity_loss()(Chamfer-style 최근접-이웃 매칭)를 대체한다.
    17섹션 모두 build_bpillar_section()의 동일 토폴로지를 스케일링만 다르게 사용하므로
    (build_bpillar_17section() 확인 — 노드를 이어붙이기만 하고 재정렬하지 않음), 인접 섹션 간
    같은 로컬 순번은 항상 같은 물리적 위치를 가리킨다 — cdist+min 없이 인덱스로 직접 대응시킨다.
    review_v1_8.md §6: 기존 방식은 스파이크가 우연히 다른 노드 근처면 위반을 놓칠 수 있었다."""
    device = new_coords.device
    total = torch.tensor(0.0, device=device)
    n_terms = 0
    unique_secs = torch.unique(section_ids).long().sort()[0]

    for i in range(len(unique_secs) - 1):
        k_a, k_b = int(unique_secs[i].item()), int(unique_secs[i + 1].item())
        s_a, s_b = scale_factor(k_a), scale_factor(k_b)
        for pid in torch.unique(part_ids):
            pid_int = int(pid.item())
            idx_a = ((section_ids == k_a) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            idx_b = ((section_ids == k_b) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            if idx_a.numel() == 0 or idx_b.numel() == 0:
                continue
            if idx_a.numel() != idx_b.numel():
                # [v1.9, 안전장치] 이 코드베이스에서는 발생하지 않지만(모든 섹션이 동일 노드 수),
                # 향후 섹션별 노드 수가 달라지는 변경이 생기면 조용히 스킵한다.
                continue
            ca = new_coords[idx_a] / s_a
            cb = new_coords[idx_b] / s_b
            dist = torch.norm(ca - cb, dim=-1)
            violation = torch.clamp(dist - threshold, min=0.0)
            total = total + torch.mean(violation ** 2)
            n_terms += 1
    if n_terms > 0:
        total = total / n_terms
    return total
```

**호출부 변경**: `train_step_multi()` 내부.

```python
# 변경 전
l_continuity = compute_shape_continuity_loss(new_coords, section_ids, part_ids, threshold=2.0)

# 변경 후
l_continuity = compute_shape_continuity_loss_v2(new_coords, section_ids, part_ids, threshold=2.0)
```

**검증 항목 추가(§7)**: `idx_a.numel() != idx_b.numel()`인 경우가 실제로 한 번도 발생하지 않는지
Stage 0 dry-run에 assert를 추가할 것을 §7에 남긴다(이번 버전에서는 조용히 스킵하는 것으로 충분,
assert 추가는 후속 버전 과제).

---

## §4. `l_phys_total` / Collision 손실 Top-K 풀링

### §4-A. `l_phys_total` (AI_design_v1_9.py, `train_step_multi()`)

```python
# 변경 전
l_phys_total = torch.stack(l_phys_terms).mean()

# 변경 후
l_phys_stack = torch.stack(l_phys_terms)                      # (17,)
k_phys = max(1, int(l_phys_stack.numel() * PHYS_TOPK_FRACTION))
l_phys_topk, _ = torch.topk(l_phys_stack, k=k_phys, largest=True)
l_phys_total = torch.mean(l_phys_topk)
```

### §4-B. `compute_collision_penalty_unclamped()` (AI_design_v1_9.py)

**대상**: 방향별 루프 내부 `loss_dir = _huber_collision_term(violation).sum() / valid.float().sum()`.

```python
# 변경 전
loss_dir = _huber_collision_term(violation).sum() / valid.float().sum()   # [v1.5 §2]

# 변경 후
huber_v = _huber_collision_term(violation)[valid.bool()]
if huber_v.numel() > 0:
    k_col = max(1, int(huber_v.numel() * PHYS_TOPK_FRACTION))
    top_huber, _ = torch.topk(huber_v, k=k_col, largest=True)
    loss_dir = top_huber.mean()
else:
    loss_dir = violation.sum() * 0.0
```

### §4-C. `compute_collision_loss_v5()` (uni_section_v20.py — 이 파일에서 유일하게 수정하는 함수)

**대상**: 방향별 루프 내부 `loss_dir = (violation ** 2).sum() / valid.float().sum()`.

```python
# 변경 전
violation = torch.relu(-gap).clamp(max=1.0) * valid.float()
loss_dir = (violation ** 2).sum() / valid.float().sum()

# 변경 후
violation = torch.relu(-gap).clamp(max=1.0) * valid.float()
sq_v = (violation ** 2)[valid.bool()]
if sq_v.numel() > 0:
    k_col = max(1, int(sq_v.numel() * PHYS_TOPK_FRACTION))
    top_sq_v, _ = torch.topk(sq_v, k=k_col, largest=True)
    loss_dir = top_sq_v.mean()
else:
    loss_dir = violation.sum() * 0.0
```

이 함수 외의 `uni_section_v20.py` 내 다른 함수는 `uni_section_v19.py`와 **1바이트도 다르지 않아야
한다**(§5 참고).

### §4-D. 신규 상수

```python
PHYS_TOPK_FRACTION = 0.3   # [v1.9 §4] 17섹션(또는 방향별 위반 노드) 중 최악 30%에 그래디언트 집중.
                           # compute_mesh_order_loss()의 top_k_fraction(0.02, 물리 엣지 ~2992개 기준)
                           # 과는 절대 개수 스케일이 다르므로(섹션 17개, collision 방향은 노드 수가
                           # 훨씬 적음) 의도적으로 더 큰 비율을 시작값으로 잡는다 — §7에서 재보정.
```

---

## §5. `z_gate_part_avg` 생존 섹션 평균화 (AI_design_v1_9.py)

**대상**: `train_step_multi()` 내부 `z_gate_part_avg = z_gate.mean(dim=0)  # (5,)`.

```python
# 변경 전
z_gate_part_avg = z_gate.mean(dim=0)  # (5,)

# 변경 후
if pruning_state is not None:
    alive_mask = torch.ones_like(z_gate, dtype=torch.bool)          # 연속 파트는 항상 alive 취급
    cand_idx_t = torch.tensor(CANDIDATE_PARTS, device=z_gate.device)
    alive_cand = (pruning_state['state'] != STATE_DELETED).to(z_gate.device)  # (17, n_cand)
    alive_mask[:, cand_idx_t] = alive_cand
    z_gate_part_avg = (z_gate * alive_mask).sum(dim=0) / alive_mask.sum(dim=0).clamp(min=1)
else:
    z_gate_part_avg = z_gate.mean(dim=0)  # Stage 0 dry-run 등 pruning_state가 없는 경로는 기존 동작 유지
```

---

## §6. 신규 상수 블록 전체 (AI_design_v1_9.py)

기존 v1.7/v1.8 상수 블록 옆에 다음을 추가/수정한다.

```python
# ══════════════════════════════════════════════════════════════════
# [v1.9] 로컬 오버라이드 상수 — 구조적 붕괴(스파이크) 해소 5종 세트
# 근거: docs/idea/idea_v1_9.md, docs/review/review_v1_8.md, docs/command/command_v1_9.md
# ══════════════════════════════════════════════════════════════════

# §1. compute_smoothness_lse_v2()
# W_SMOOTH_LSE = 0.25 (v1.7 값) → 3.0 (아래 실제 정의 참고, 기존 상수 재사용)
STAGE4_GATE_WARMUP_EPOCHS = 50   # §2. Stage4 hard-flip을 이 기간에 걸쳐 선형 보간으로 완화
PHYS_TOPK_FRACTION = 0.3         # §4. l_phys_total/collision 손실 Top-K 비율

# 주의: SMOOTH_LSE_TEMPERATURE(1.5)와 uni_section_v20.compute_mesh_order_loss()의
# top_k_fraction(0.02)은 이번 버전 범위 밖이므로 변경하지 않는다.
```

`W_SMOOTH_LSE`는 새 상수가 아니라 기존 상수의 값만 바뀐다(§1.2 참고) — 위 블록에는 주석으로만
표기하고 실제 정의는 기존 위치(`W_SMOOTH_LSE = 0.25` 줄)를 직접 수정한다.

`STAGE4_EPOCHS >= STAGE4_GATE_WARMUP_EPOCHS` assert는 §2.1에 이미 명시했다.

---

## §7. 임포트 경로 갱신 (AI_design_v1_8.py → v1.9)

**대상**: 모듈 상단 임포트문.

```python
# 변경 전
import uni_section_v19 as usv18   # [v1.8] compute_mesh_order_loss() Top-K 풀링 적용판

# 변경 후
import uni_section_v20 as usv18   # [v1.9] compute_collision_loss_v5() Top-K 풀링 적용판(§4-C)
```

별칭(`usv18`)은 그대로 유지한다(파일 전체에서 다회 참조 — command_v1_8.md와 동일한 원칙, 불필요한
전체 치환으로 인한 오류 위험 최소화). 과거 이력을 기록하는 docstring 내 문자열 언급은 수정하지 않는다.

---

## §8. 절대 변경 금지 항목

- `SMOOTH_LSE_TEMPERATURE`(1.5), `compute_mesh_order_loss()`의 `top_k_fraction`(0.02) — 이번
  버전 범위 밖.
- §4(중립 초기화+지연 엔트로피), §7(R1~R3 롤백)의 상수·로직(v1.3~v1.4부터 계속 유지) — 여전히
  수정하지 않는다.
- `uni_section_v20.py` 내부에서 `compute_collision_loss_v5()` **이외의 모든 함수**는 원본
  `uni_section_v19.py`와 **1바이트도 다르지 않아야 한다.**
- `uni_section_v19.py` 파일 자체는 삭제/수정하지 않는다(과거 버전 보존).
- `build_bpillar_section()`의 기하 생성 로직, `CGDN17.forward()`의 GNN 레이어 전파 수식 — 이번
  버전과 무관.
- Stage 1/2/3의 커리큘럼 스케줄(`get_curriculum_weights_v10`, `continuity_weight_schedule`)은
  변경하지 않는다 — `W_SMOOTH_LSE`만 상향하고 스케줄 자체는 그대로 둔다.

---

## §9. 파일 헤더 갱신

**`uni_section_v20.py`** 최상단에 변경 이력 추가:

```
[v20 변경, docs/command/command_v1_9.md 실행]
  compute_collision_loss_v5(): 방향별 위반량 sum()/valid.sum() 평균 → Top-K(PHYS_TOPK_FRACTION=0.3)
  평균 풀링. 한두 노드의 깊은 국소 관통(송곳형)이 다수의 정상 노드에 의해 평균에서 희석되던 문제를
  compute_mesh_order_loss()(v19)와 동일한 패턴으로 해소(review_v1_8.md §9).
  그 외 모든 함수는 uni_section_v19.py와 동일.
```

**`AI_design_v1_9.py`** 최상단 docstring(v1.8 항목 바로 위)에 추가:

```
AI_design_v1_9.py — Structural Collapse (Spike) Mitigation Suite (v1.9)
─────────────────────────────────────
[v1.9 변경, docs/command/command_v1_9.md 실행]
  §1 compute_smoothness_lse_v2(): 17섹션 전역 단일 스칼라 LSE → 섹션별 분리 LSE +
     t_final 노드 단위 마스킹. W_SMOOTH_LSE 0.25→3.0. review_v1_8.md ①②③(가중치 불균형/
     whack-a-mole/Ghost Gradient Hijacking) 동시 해소.
  §2 Stage4 게이트 강제 이진화(±5.0 단일 스텝) → STAGE4_GATE_WARMUP_EPOCHS(50)에 걸친 선형
     보간 + optimizer 모멘텀 이전(모델 파라미터가 재생성되지 않으므로 파라미터 객체를 직접
     dict key로 재사용). review_v1_8.md ⑤ 해소.
  §3 compute_shape_continuity_loss_v2(): Chamfer-style 최근접-이웃 매칭 → 인덱스 1:1 대응
     거리 비교(17섹션 동일 토폴로지 전제). review_v1_8.md ⑥ 해소.
  §4 l_phys_total 및 collision 손실(uni_section_v20.compute_collision_loss_v5,
     AI_design_v1_9.compute_collision_penalty_unclamped) 평균 → Top-K(PHYS_TOPK_FRACTION=0.3)
     풀링. review_v1_8.md ⑦⑨ 해소. 기존 §1(v1.8)의 Order 손실 Top-K 패턴을 재사용(새 메커니즘
     도입 없음).
  §5 z_gate_part_avg: 17섹션 전체 평균 → pruning_state 기반 생존 섹션만의 평균. review_v1_8.md
     ⑧ 해소.
  구현: /synod design 세션(Gemini flash conf98→100, OpenAI o3 conf92→84→71)에서 두 모델 모두
  Solver/Critic/Defense 각 라운드에서 실제 코드에 없는 구체적 수치(가짜 실험 로그, 잘못된
  기존 상수값, 존재하지 않는 PyTorch 실패 시나리오)를 반복적으로 인용했다 — Claude가 매 라운드
  실제 소스 코드 대조로 반증했다(SMOOTH_LSE_TEMPERATURE 실제값 1.5, top_k_fraction 실제값 0.02,
  Stage4는 모델을 재생성하지 않으므로 파라미터 객체 직접 매칭이면 충분함 등). OpenAI가 제안한
  compute_smoothness_lse() legacy wrapper 유지 방식은 이 프로젝트에 전례가 없어 기각, Gemini의
  완전 교체 방식을 채택했다. 파일 버전 관리도 OpenAI가 제안한 "기존 파일 그대로 사용"을 프로젝트
  관례 위반으로 기각하고 Gemini의 신규 파일 생성안을 채택했다. optimizer state 이전은 두 모델의
  제안(id() 매핑, state_dict() 재직렬화 기반 매칭) 모두 이 코드베이스의 실제 상황(모델 미재생성)
  에는 불필요하게 복잡하거나 실제로는 키 타입 불일치로 동작하지 않는 코드였음을 Judge가 직접
  확인해, 파라미터 객체를 그대로 dict key로 쓰는 가장 단순한 방식으로 대체했다.
그 외 로직은 AI_design_v1_8.py와 동일. uni_section_v20.py는 compute_collision_loss_v5() 1개
함수만 수정, 나머지는 uni_section_v19.py와 동일.
─────────────────────────────────────
```

---

## §10. 검증 계획

이 프로젝트에는 CI 파이프라인이 없다 — 콘솔 로그 확인, 짧은 스모크 실행, 전체 학습 후 육안 확인의
3단계로 검증한다(command_v1_8.md §7과 동일한 실전 방식).

1. **임포트/문법 확인**: `import uni_section_v20 as usv18` 변경 후 임포트 에러 없는지, Stage 0
   dry-run(`--dry-run-only`)이 통과하는지 확인.
2. **스모크 테스트(20 epoch)**: 콘솔 로그에서 `l_smooth_lse`, `l_continuity`, `l_col_aux`가
   NaN/Inf 없이 정상 출력되는지 확인. 처음 5 epoch 동안 `W_SMOOTH_LSE=3.0` 적용 후 총 손실에서
   `contrib_smooth_lse`가 지배적이지 않은지(다른 항 대비 5~10배 이내) 확인 — 초과하면
   `W_SMOOTH_LSE`를 1.0~2.0으로 낮춰 재시도.
3. **Stage4 보간 확인**: Stage4 진입 로그에서 `log_alpha_candidates`가 매 epoch 목표값에
   점진적으로 접근하는지(옵션: 10epoch 간격으로 `.abs().mean()` 값을 콘솔에 출력), 그리고 §5.7의
   기존 종료 assert(`z_gate`가 0/1로 정확히 포화)가 여전히 통과하는지 확인.
4. **optimizer state 이전 확인**: `[Stage4] optimizer 모멘텀 이전 완료: N개 파라미터` 로그가
   출력되는지, `try/except`의 `[WARNING]` 분기가 (정상 상황에서는) 타지 않는지 확인.
5. **회귀 테스트(전체 학습)**: `reports/v1_9/`에서
   - `l_smooth_lse`가 v1.8보다 낮은 값에서 안정화되는지(섹션별 분리 + 마스킹 효과)
   - `l_phys_total`(Top-K 적용 전/후 비교를 위해 `mean` 버전도 로깅에 병기 권장)이 특정 섹션의
     국소 붕괴에 더 민감하게 반응하는지
   - 최종 3D 형상(`AI_design_v1_9_3d.html`)에서 review_v1_8.md가 지적한 잔존 스파이크가
     추가로 완화됐는지 육안 확인
   - Mp 목표 미달 섹션이 v1.8보다 늘지 않았는지
6. **`PHYS_TOPK_FRACTION` 재보정**: 스모크 테스트 로그에서 `contrib_phys`(Top-K 적용 후)가
   적용 전(mean 버전 참고값) 대비 과도하게(10배 이상) 커지면 `0.3`을 `0.15~0.2` 수준으로
   낮춰 재조정한다.
7. **`compute_shape_continuity_loss_v2` 안전장치 확인**: 학습 전 구간에서
   `idx_a.numel() != idx_b.numel()`로 스킵되는 (섹션, 파트) 쌍이 실제로 0개인지 임시 카운터로
   확인(현재 코드베이스에서는 항상 0이어야 함 — 0이 아니면 `build_bpillar_17section()`의 노드
   수 불일치를 먼저 조사할 것).
