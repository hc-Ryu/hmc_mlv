# command_v1_10.md — 잔존 스파이크(경계 국소 꺾임) 해소 3종 세트 구현 지시서

작성일: 2026-08-10 (`/synod design` 세션, Gemini flash conf98 + OpenAI o3 conf92)
참조: `docs/idea/idea_v1_10.md`(설계 근거), `docs/review/review_v1_9.md`(①②③, 3D 결과 직접
디코딩으로 확정된 잔존 스파이크 정밀 위치), `AI_design_v1_9.py`(기반 코드),
`uni-section/code/uni_section_v20.py`(**이번 버전에서 수정하지 않음**)

> **앵커 규칙**: 본 문서의 위치 지정은 **함수/변수 이름 기준**이다. 라인 번호는 참고용이며
> 코드 변경으로 어긋날 수 있다.

> **범위**: 이번 버전은 `AI_design_v1_9.py`만 수정한다. `uni_section_v20.py`는 idea_v1_10.md의
> 3가지 아이디어 모두 이 파일의 함수(`compute_collision_loss_v5` 등)와 무관하므로 **전혀
> 건드리지 않는다** — v21 신규 파일도 만들지 않는다(Solver 라운드에서 두 모델 공통 결론).

---

## §0. 목표 및 산출물

`AI_design_v1_9.py`를 복사해 **`AI_design_v1_10.py`**를 만든다(프로젝트 관례 — command_v1_9.md
§0과 동일한 파일 버전업 패턴).

```bash
cp AI_design_v1_9.py AI_design_v1_10.py
```

| idea_v1_10.md 조항 | 해소 원인(review_v1_9.md) | 비고 |
|---|---|---|
| §1 `compute_smoothness_lse_v3`(soft-threshold + Top-K) | ①(가중치 부작용 — 두께 소멸 우회) + ③(섹션 내부 whack-a-mole) | `compute_smoothness_lse_v2`(v1.9) 대체 |
| §2 `compute_boundary_continuity_loss`(경계 완충) | ②(Stage4 게이트 보간이 연속 파트 재형성을 방치) | 기존 `compute_shape_continuity_loss_v2`(v1.9)는 그대로 유지, **추가 항** |
| §3 `w_smooth_lse_schedule`(단계적 스케줄) | ①(처음부터 세게 누르는 부작용 완화) | 기존 `continuity_weight_schedule` 패턴 재사용 |

**적용 순서·원자성(Solver 라운드 공통 결론)**: §1과 §3은 **반드시 같은 커밋**으로 묶어 적용한다 —
§3의 목표치(`W_SMOOTH_LSE=3.0`)는 §1의 새 집계 방식(Top-K 평균, soft-threshold)을 전제로 한
스케일이므로, §1 없이 §3만 적용하면 v1.9와 동일한 부작용이 재현될 수 있다. §2(경계 완충 손실)는
개념적으로 독립적이지만 같은 파일의 같은 함수(`train_step_multi`)를 수정하므로, 이번 버전에서는
셋 다 하나의 커밋으로 함께 적용하고 함께 검증한다.

---

## §1. `compute_smoothness_lse_v3` — Soft-threshold + Top-K (AI_design_v1_9.py 기준 위치)

### §1.1 신규 함수 추가, 기존 함수 대체

**대상**: `compute_smoothness_lse_v2()`(v1.9, `build_static_neighbor_map()` 아래) 바로 다음에
신규 함수를 추가한다. 기존 함수는 **삭제하지 않고 그대로 둔다**(command_v1_9.md가
`compute_smoothness_lse`(v1.7)를 완전 삭제했던 것과 달리, 이번엔 두 버전을 나란히 남긴다 — §1.2
호출부만 바꾸면 `compute_smoothness_lse_v2`는 자연히 미사용 상태가 되지만, review_v1_9.md의
정량 분석이 v2 기준이었으므로 향후 ablation 비교 시 참조할 수 있도록 보존한다. 필요시 후속
버전에서 제거).

```python
# AI_design_v1_10.py — compute_smoothness_lse_v2() 바로 아래에 추가

def _soft_active_weight(t_flat, eps=0.1, transition_width=SMOOTH_EPS_TRANSITION_WIDTH):
    """[v1.10 §1.1] compute_smoothness_lse_v2()의 하드 eps 컷오프(`t_flat > eps` 불리언)를
    대체하는 sigmoid soft-threshold. t_flat=0(진짜 DELETED, z_gate=0으로 t_final이 정확히 0)이면
    sigmoid((0-0.1)/0.02)=sigmoid(-5)≈0.0067로 사실상 0 — review_v1_8.md §3 Ghost Gradient
    Hijacking 방지 효과는 그대로 유지된다. Stage4 워밍업 중 t_flat이 eps 근방을 서서히 통과하는
    노드는 가중치가 매끄럽게 0→1로 바뀌어, '살짝만 깎아도 손실 항 전체가 증발'하는 계단 효과
    (review_v1_9.md §1의 우회 경로)를 없앤다."""
    return torch.sigmoid((t_flat - eps) / transition_width)


def compute_smoothness_lse_v3(x, new_coords, neighbor_map, t_final, num_sections=NUM_SECTIONS,
                               eps=0.1, transition_width=SMOOTH_EPS_TRANSITION_WIDTH,
                               top_k_fraction=SMOOTH_TOPK_FRACTION,
                               temperature=SMOOTH_LSE_TEMPERATURE):
    """[v1.10 §1] compute_smoothness_lse_v2()(v1.9)를 대체한다.
    (1) 하드 eps 컷오프 -> sigmoid soft-threshold(_soft_active_weight): DELETED 노드는 여전히
        가중치≈0으로 배제되지만, 전이 구간의 그래디언트 계단 현상을 없앤다.
    (2) 섹션 내부 logsumexp(사실상 max) -> Top-K 평균: review_v1_9.md §3에서 3D 결과를 직접
        디코딩해 확인한 "섹션당 정확히 1곳만 남는" 패턴의 직접 원인을 제거한다."""
    node_idx  = neighbor_map['node_idx']
    left_idx  = neighbor_map['left_idx']
    right_idx = neighbor_map['right_idx']
    if node_idx.numel() == 0:
        return new_coords.sum() * 0.0

    t_flat = t_final.squeeze(-1) if t_final.dim() > 1 else t_final
    w_i = _soft_active_weight(t_flat[node_idx],  eps, transition_width)
    w_l = _soft_active_weight(t_flat[left_idx],  eps, transition_width)
    w_r = _soft_active_weight(t_flat[right_idx], eps, transition_width)
    active_weight = w_i * w_l * w_r          # (M,) — 셋 다 살아있어야 1에 가까움

    section_ids = x[node_idx, 5].long()
    p_i, p_left, p_right = new_coords[node_idx], new_coords[left_idx], new_coords[right_idx]
    deviation = p_i - 0.5 * (p_left + p_right)
    sq_deviation = torch.sum(deviation ** 2, dim=-1) * active_weight

    section_losses = []
    for sec in range(num_sections):
        sec_mask = section_ids == sec
        if not sec_mask.any():
            continue
        d = sq_deviation[sec_mask]
        n = d.numel()
        k = max(1, int(n * top_k_fraction))
        top_d, _ = torch.topk(d, k=k, largest=True)
        section_losses.append(top_d.mean())
    if not section_losses:
        return new_coords.sum() * 0.0
    return torch.stack(section_losses).mean()
```

### §1.2 호출부 교체

**대상**: `train_step_multi()` 내부, 기존 `# [v1.9 §1]` 주석이 달린 줄.

```python
# 변경 전
l_smooth_lse = compute_smoothness_lse_v2(x, new_coords, neighbor_map, t_final)   # [v1.9 §1]

# 변경 후
l_smooth_lse = compute_smoothness_lse_v3(x, new_coords, neighbor_map, t_final)   # [v1.10 §1]
```

---

## §2. `compute_boundary_continuity_loss` — Candidate 경계 섹션 전용 완충 손실

### §2.1 신규 함수 추가

**대상**: `compute_shape_continuity_loss_v2()`(v1.9) 바로 아래에 추가.

```python
def find_candidate_boundary_sections(pruning_state):
    """[v1.10 §2] pruning_state['state']((17, n_cand), ALIVE=0/PENDING=1/DELETED=2 카테고리 값)
    에서 인접 섹션 간 상태가 달라지는 (k, k+1) 쌍을 찾는다. candidate 파트가 살아있다가 죽거나,
    죽어있다가 살아나는 경계를 모두 포함한다. pruning_state가 None이면(Stage 0 dry-run 등)
    빈 리스트를 반환한다."""
    if pruning_state is None:
        return []
    state = pruning_state['state']    # (17, n_cand), CPU
    boundary_pairs = []
    for k in range(NUM_SECTIONS - 1):
        if not torch.equal(state[k], state[k + 1]):
            boundary_pairs.append((k, k + 1))
    return boundary_pairs


def compute_boundary_continuity_loss(new_coords, section_ids, part_ids, boundary_pairs,
                                      threshold=BOUNDARY_CONTINUITY_THRESHOLD):
    """[v1.10 §2] candidate 생존 경계 섹션에서 연속 파트(CONTINUOUS_PARTS=[0,1,2])에 한해
    compute_shape_continuity_loss_v2()와 동일한 인덱스 1:1 대응 방식을 재사용하되(새 메커니즘
    도입 금지), 경계 섹션·연속 파트로만 범위를 좁히고 threshold를 더 엄격하게(1.0mm) 적용한다.
    review_v1_9.md §2: Stage4 게이트 보간은 candidate 파트만 다루고 인접 연속 파트의 재형성은
    별도 완충 장치 없이 방치돼 경계 섹션에서 국소 꺾임이 남았다."""
    device = new_coords.device
    total = torch.tensor(0.0, device=device)
    n_terms = 0
    for k_a, k_b in boundary_pairs:
        s_a, s_b = scale_factor(k_a), scale_factor(k_b)
        for pid_int in CONTINUOUS_PARTS:   # [0, 1, 2]만 — candidate 파트 자체는 이미 §1에서 다룸
            idx_a = ((section_ids == k_a) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            idx_b = ((section_ids == k_b) & (part_ids == pid_int)).nonzero(as_tuple=True)[0]
            if idx_a.numel() == 0 or idx_b.numel() == 0 or idx_a.numel() != idx_b.numel():
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

### §2.2 호출부 추가

**대상**: `train_step_multi()` 내부, `l_continuity`/`contrib_continuity` 계산 직후
(`contrib_continuity = w_continuity * l_continuity` 다음 줄).

```python
# 추가
boundary_pairs = find_candidate_boundary_sections(pruning_state)   # [v1.10 §2] None-safe 내장
l_boundary_continuity = compute_boundary_continuity_loss(new_coords, section_ids, part_ids,
                                                           boundary_pairs)
contrib_boundary_continuity = W_BOUNDARY_CONTINUITY * l_boundary_continuity   # [v1.10 §2]
```

### §2.3 최종 loss 합산식에 추가

**대상**: `train_step_multi()`의 `loss = (contrib_phys + contrib_smooth + L_alda_effective + ...)`
합산식.

```python
# 변경 전 (v1.9)
loss = (contrib_phys + contrib_smooth + L_alda_effective
        + contrib_order + contrib_anchor + contrib_sat
        + contrib_sparse + contrib_entropy + contrib_continuity
        + contrib_mass + contrib_col_aux + contrib_area_floor
        + contrib_smooth_lse)

# 변경 후 (v1.10)
loss = (contrib_phys + contrib_smooth + L_alda_effective
        + contrib_order + contrib_anchor + contrib_sat
        + contrib_sparse + contrib_entropy + contrib_continuity
        + contrib_boundary_continuity                              # [v1.10 §2] 신규
        + contrib_mass + contrib_col_aux + contrib_area_floor
        + contrib_smooth_lse)
```

---

## §3. `w_smooth_lse_schedule` — 단계적 스케줄 (기존 `continuity_weight_schedule` 패턴 재사용)

### §3.1 신규 함수

**대상**: `continuity_weight_schedule()` 바로 아래에 추가.

```python
def w_smooth_lse_schedule(epoch, stage2_end, w_max=W_SMOOTH_LSE, w_min=W_SMOOTH_LSE_FLOOR,
                           beta=0.05):
    """[v1.10 §3] continuity_weight_schedule()과 동일한 sigmoid 성장 패턴(새 메커니즘 도입 금지).
    stage2_end(ASWC_STAGE2_END=400) 근방부터 목표치(W_SMOOTH_LSE=3.0)로 서서히 올라간다 — Stage4
    게이트 확정 이전에는 형상 탐색이 Mp/면적 목표를 우선하도록 완화하고(review_v1_9.md §1 부작용
    대응), 게이트가 정해지는 시점부터 smoothness 억제력을 강화한다."""
    return w_min + (w_max - w_min) / (1.0 + math.exp(-beta * (epoch - stage2_end)))
```

### §3.2 `train_step_multi()` 시그니처 및 내부 — sentinel 패턴

**주의**: 기존 `w_continuity_val` sentinel은 단순 상수 폴백이 아니라 **None이면 스케줄 함수를
그 자리에서 호출**하는 패턴이다(`AI_design_v1_9.py` 1599~1602행 부근). `w_smooth_lse_val`도
정확히 동일한 패턴을 따른다 — 정적 상수로 폴백하면 Stage4 호출부(§3.3에서 인자를 넘기지 않음)가
항상 `W_SMOOTH_LSE` 고정값을 쓰게 되어 스케줄의 의미가 없어진다.

```python
# 변경 전 (시그니처)
def train_step_multi(model, data, optimizer, target_mps, target_area,
                      epoch, max_epochs, weights, curriculum,
                      curriculum_ratio, collision_spec, alda_state,
                      pruning_state=None, segment_ids=None,
                      alpha_ent=ALPHA_ENT_V13, stage4_active=False, frozen_curriculum=None,
                      w_col_aux_val=None, w_continuity_val=None, neighbor_map=None):

# 변경 후
def train_step_multi(model, data, optimizer, target_mps, target_area,
                      epoch, max_epochs, weights, curriculum,
                      curriculum_ratio, collision_spec, alda_state,
                      pruning_state=None, segment_ids=None,
                      alpha_ent=ALPHA_ENT_V13, stage4_active=False, frozen_curriculum=None,
                      w_col_aux_val=None, w_continuity_val=None, neighbor_map=None,
                      w_smooth_lse_val=None):   # [v1.10 §3]
```

```python
# 변경 전 (l_continuity 계산부 근처)
if w_continuity_val is None:                                          # [v1.6 §1]
    w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)
else:
    w_continuity = w_continuity_val
l_continuity = compute_shape_continuity_loss_v2(new_coords, section_ids, part_ids, threshold=2.0)
contrib_continuity = w_continuity * l_continuity

# 변경 후 — 바로 아래에 동일 패턴 추가(§2.2의 boundary 계산보다 앞에 와도 무방)
if w_continuity_val is None:                                          # [v1.6 §1]
    w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)
else:
    w_continuity = w_continuity_val
l_continuity = compute_shape_continuity_loss_v2(new_coords, section_ids, part_ids, threshold=2.0)
contrib_continuity = w_continuity * l_continuity

if w_smooth_lse_val is None:                                          # [v1.10 §3]
    w_smooth_lse = w_smooth_lse_schedule(epoch, ASWC_STAGE2_END)
else:
    w_smooth_lse = w_smooth_lse_val
```

**대상**: 기존 `contrib_smooth_lse = W_SMOOTH_LSE * l_smooth_lse  # [v1.7 §1]` 줄.

```python
# 변경 전
contrib_smooth_lse = W_SMOOTH_LSE * l_smooth_lse                  # [v1.7 §1]

# 변경 후
contrib_smooth_lse = w_smooth_lse * l_smooth_lse                  # [v1.10 §3]
```

### §3.3 `run_training_multi()` 메인 루프 호출부 갱신

**대상**: 메인 학습 루프의 `train_step_multi(...)` 호출(`curr_w_continuity = continuity_weight_schedule(...)`
바로 다음).

```python
# 변경 전
curr_w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)   # [v1.7 §2] 단순화
info = train_step_multi(model, data, optimizer, target_mps, target_area,
                         epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                         collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                         alpha_ent=alpha_ent_schedule(epoch), stage4_active=False,
                         w_col_aux_val=curr_w_col_aux,
                         w_continuity_val=curr_w_continuity,
                         neighbor_map=neighbor_map)  # [v1.4 §4.2][v1.5 §3][v1.7 §1][v1.7 §2]

# 변경 후
curr_w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)   # [v1.7 §2] 단순화
curr_w_smooth_lse = w_smooth_lse_schedule(epoch, ASWC_STAGE2_END)        # [v1.10 §3]
info = train_step_multi(model, data, optimizer, target_mps, target_area,
                         epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                         collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                         alpha_ent=alpha_ent_schedule(epoch), stage4_active=False,
                         w_col_aux_val=curr_w_col_aux,
                         w_continuity_val=curr_w_continuity,
                         w_smooth_lse_val=curr_w_smooth_lse,              # [v1.10 §3]
                         neighbor_map=neighbor_map)  # [v1.4 §4.2][v1.5 §3][v1.7 §1][v1.7 §2][v1.10 §3]
```

**Stage4 호출부(2번째 `train_step_multi` 호출)는 수정하지 않는다** — `w_continuity_val`을
그동안도 Stage4에서 넘기지 않아 왔고(내부 sentinel이 `continuity_weight_schedule(ep, ...)`를
그 자리에서 계산), `w_smooth_lse_val`도 동일한 이유로 넘기지 않아 Stage4의 `ep`(이미
`ASWC_STAGE2_END`를 한참 지난 값) 기준으로 스케줄이 자동으로 거의 `W_SMOOTH_LSE`(목표치)에
포화된 값을 계산한다 — 기존 `w_continuity_val` 처리와 완전히 동일한 동작이므로 별도 예외 처리가
필요 없다.

---

## §4. 신규 상수 블록 (AI_design_v1_9.py 기준, `W_SMOOTH_LSE` 정의 근처)

```python
# ══════════════════════════════════════════════════════════════════
# [v1.10] 로컬 오버라이드 상수 — 잔존 스파이크(경계 국소 꺾임) 해소 3종 세트
# 근거: docs/idea/idea_v1_10.md, docs/review/review_v1_9.md, docs/command/command_v1_10.md
# ══════════════════════════════════════════════════════════════════
SMOOTH_TOPK_FRACTION = 0.3            # §1. 섹션 내 활성 노드 중 상위 30%(최소 1개) 평균
SMOOTH_EPS_TRANSITION_WIDTH = 0.02    # §1. eps 근방 soft-threshold 전이폭(mm)
BOUNDARY_CONTINUITY_THRESHOLD = 1.0   # §2. mm. 일반 continuity threshold(2.0mm)보다 엄격
W_BOUNDARY_CONTINUITY = 0.5           # §2. 시작값 — §5에서 재보정
W_SMOOTH_LSE_FLOOR = 0.5              # §3. 학습 초반 최소값(v1.9의 3.0 고정값보다 낮게 시작)
```

`W_SMOOTH_LSE`(기존 상수, 값은 3.0 그대로 유지)는 이제 `w_smooth_lse_schedule()`의 "도달
목표치"로 의미가 바뀐다 — 상수 정의 자체는 변경하지 않는다.

---

## §5. 절대 변경 금지 항목

- `uni_section_v20.py`는 이번 버전에서 **전혀 수정하지 않는다**(§0 참고).
- `compute_shape_continuity_loss_v2()`(v1.9)는 전 구간 threshold=2.0mm 동작을 그대로 유지한다 —
  §2의 신규 함수는 이를 대체하는 것이 아니라 경계 섹션에만 추가되는 **별도 항**이다.
- `compute_smoothness_lse_v2()`(v1.9)는 삭제하지 않고 코드에 남겨둔다(§1.1 참고, 호출부만 v3로
  교체).
- `SMOOTH_LSE_TEMPERATURE`(1.5), `uni_section_v20.compute_mesh_order_loss()`의
  `top_k_fraction`(0.02), `PHYS_TOPK_FRACTION`(0.3, v1.9에서 도입) — 이번 버전 범위 밖.
- Stage4 진입부의 게이트 선형 보간(`STAGE4_GATE_WARMUP_EPOCHS` 등, v1.9 §2)과 optimizer state
  이전 로직은 변경하지 않는다.
- `NUM_SECTIONS`, `CONTINUOUS_PARTS`, `CANDIDATE_PARTS` 등 전역 구조 상수는 수정하지 않는다.

---

## §6. 파일 헤더 갱신

**`AI_design_v1_10.py`** 최상단 docstring(v1.9 항목 바로 위)에 추가:

```
AI_design_v1_10.py — Residual Spike (Boundary Kink) Mitigation Suite (v1.10)
─────────────────────────────────────
[v1.10 변경, docs/command/command_v1_10.md 실행]
  §1 compute_smoothness_lse_v3(): 하드 eps 컷오프 -> sigmoid soft-threshold(_soft_active_weight),
     섹션 내부 logsumexp(사실상 max) -> Top-K(SMOOTH_TOPK_FRACTION=0.3) 평균. review_v1_9.md
     ①(두께 소멸 우회)·③(섹션당 1곳만 남는 패턴) 동시 해소. compute_smoothness_lse_v2(v1.9)는
     보존(미사용, ablation 참조용).
  §2 compute_boundary_continuity_loss(): candidate 파트 생존/삭제 경계 섹션에서만 연속 파트
     (0/1/2)에 더 엄격한 threshold(1.0mm)로 추가 완충 손실. 기존 compute_shape_continuity_loss_v2
     (전 구간 2.0mm)는 그대로 유지 — 대체 아닌 추가. review_v1_9.md ②(Stage4 보간이 연속 파트
     재형성을 방치) 해소.
  §3 w_smooth_lse_schedule(): 기존 continuity_weight_schedule()과 동일한 sigmoid 성장 패턴
     재사용. W_SMOOTH_LSE_FLOOR(0.5)에서 시작해 ASWC_STAGE2_END 근방부터 목표치(W_SMOOTH_LSE=
     3.0)로 서서히 상승 — review_v1_9.md ①(처음부터 세게 누르는 부작용) 완화.
  구현: /synod design 세션(Gemini flash conf98, OpenAI o3 conf92)에서 §1·§3을 동일 커밋으로
  묶어야 한다는 데 두 모델이 수렴했다(§3의 목표치가 §1의 새 집계 방식을 전제로 재보정된 값이므로).
  OpenAI가 제시한 검증 계획(unit_test_loss_grad.py 등 존재하지 않는 테스트 스크립트, "offline
  실험에서 <1% loss 폭주"라는 근거 없는 인용)은 이 프로젝트에 그런 테스트 인프라나 실행 로그가
  없음을 Judge가 확인해 기각하고, 이 프로젝트가 실제로 써 온 검증 방식(콘솔 로그+스모크 실행+
  전체 학습 후 3D 결과 육안 확인)으로 대체했다. 두 모델 모두 train_step_multi()의 실제 시그니처를
  단순화해 부정확하게 제시했으나, Judge가 실제 코드(AI_design_v1_9.py)를 직접 대조해 정확한
  파라미터 목록과 w_continuity_val sentinel의 정확한 동작 방식(정적 상수 폴백이 아니라 스케줄
  함수를 그 자리에서 호출하는 방식)을 확인하고 w_smooth_lse_val도 동일 패턴으로 구현했다.
그 외 로직은 AI_design_v1_9.py와 동일. uni_section_v20.py는 이번 버전에서 수정하지 않는다.
─────────────────────────────────────
```

---

## §7. 검증 계획

이 프로젝트에는 CI나 별도 테스트 스크립트가 없다 — 콘솔 로그 확인, 짧은 스모크 실행, 전체 학습
후 육안 확인의 3단계로 검증한다(command_v1_9.md §10과 동일한 실전 방식).

1. **Soft-threshold 단위 확인(최우선)**: 코드 실행 전 별도 스크립트로
   `_soft_active_weight(torch.tensor(0.0), eps=0.1, transition_width=0.02)` ≈ 0.0067임을 먼저
   확인 — review_v1_8.md §3 Ghost Gradient Hijacking 버그가 재발하지 않는지의 1차 방어선이다.
2. **임포트/문법 확인**: `AI_design_v1_10.py` 문법 오류 없는지(`py -m py_compile`), Stage 0
   dry-run(`--dry-run-only`)이 통과하는지 확인. `find_candidate_boundary_sections(None)`이 빈
   리스트를 반환하는지(Stage 0에서 `pruning_state=None`) 확인.
3. **스모크 테스트(20 epoch)**: 콘솔 로그에서 `l_smooth_lse`(v3), `l_boundary_continuity`,
   `w_smooth_lse`(현재 스케줄 값)이 NaN/Inf 없이 정상 출력되는지 확인. `len(boundary_pairs)`가
   학습 중 합리적 범위(0~수개)인지 확인.
4. **Top-K 회귀 확인**: `top_k_fraction=1.0`으로 임시로 주면 기존 `logsumexp`(v1.9) 동작과 유사한
   경향을 보이는지(완전히 동일하지는 않음 — mean vs logsumexp 차이) 확인.
5. **회귀 테스트(전체 학습)**: `reports/v1_10/`에서
   - target miss 섹션 수가 v1.9(10개)보다 줄어드는지(v1.8의 7개 수준 이하가 목표)
   - 면적 감량률이 목표(5%) 근처로 돌아오는지(v1.9의 -11.6% 과다 감량 완화)
   - review_v1_9.md가 3D 데이터에서 특정한 7곳의 날카로운 꺾임(#06 섹션 12~15, #00 섹션 10·11·16)
     이 실제로 완화되는지 — **동일한 방식(3D html 직접 디코딩 + 꺾임각 계산)으로 재검증할 것**
     (review_v1_9.md의 분석 스크립트 로직 재사용 가능).
6. **가중치 재보정**: `contrib_boundary_continuity`가 스모크 테스트 로그에서 다른 항(`contrib_phys`,
   `contrib_continuity` 등) 대비 과도하게(10배 이상) 크거나 작으면 `W_BOUNDARY_CONTINUITY`를
   조정한다. `w_smooth_lse_schedule`의 `beta`, `W_SMOOTH_LSE_FLOOR`도 초기 수렴 속도를 보고
   재조정한다.
7. **로깅 강화**: 이번에도 실행 로그를 저장할 것 — v1.5~v1.9 리뷰가 6회 연속 "저장된 콘솔 로그가
   없어 학습 중 곡선을 직접 확인 못했다"고 지적했다. `history` dict에 `l_smooth_lse`,
   `l_boundary_continuity`, `w_smooth_lse`(스케줄 값)를 매 epoch 추가하고 파일로 저장할 것.
