# idea_v1_10.md — 잔존 스파이크(경계 국소 꺾임) 해소를 위한 v1.10 설계 아이디어

작성일: 2026-08-10 (`/synod idea` 세션, Gemini flash conf95→98→100 + OpenAI gpt4o conf85→75→60)
근거: `docs/review/review_v1_9.md`(①②③, 3D 결과 직접 디코딩으로 확정된 잔존 스파이크 정밀 위치)
범위: `AI_design_v1_9.py`(메인, 로컬 오버라이드 패턴 유지). `uni-section/code/uni_section_v20.py`는
이번 범위에서 수정하지 않는다.

**세션 개요**: Solver 라운드에서 Gemini가 제안한 "eps 하드 마스킹을 완전히 제거하자"는 아이디어는
Critic 라운드에서 Gemini 자신과 OpenAI 둘 다 review_v1_8.md §3(Ghost Gradient Hijacking)의 재발
위험으로 기각했다. 대신 "eps 하드 컷오프 대신 sigmoid 기반 soft-threshold로 완화"라는 대안에
수렴했다. Defense 라운드에서 두 모델 모두 실제 코드 수준 구현을 시도했으나, Gemini는
`pruning_state['state']`(ALIVE/PENDING/DELETED 3단계 카테고리 텐서, shape `(17, n_cand)`)를
연속적인 두께 값처럼 오해해 이 프로젝트와 무관한 `nn.Module`을 새로 만들었고, OpenAI는 실제
코드를 전혀 제시하지 못했다 — 이 두 결과물은 채택하지 않고, Claude가 Judge로서 실제 데이터
구조(`neighbor_map`의 flat 텐서, `pruning_state['state']`의 카테고리 값, `t_final`의 실제 의미)에
맞게 직접 구현했다. 개념적 합의(Top-K 다중 노드 집계, 하드 컷오프 대신 soft-threshold, 경계 섹션
전용 완충 손실, `W_SMOOTH_LSE` 스케줄링) 자체는 두 모델과 review_v1_9.md 모두 동일한 방향이었으므로
그대로 유지했다.

---

## §1. [최우선] `compute_smoothness_lse_v2`의 섹션 내부 집계를 Top-K + Soft-threshold로 교체

**해소 원인**: review_v1_9.md ①(가중치 상향의 부작용 — 두께 소멸을 통한 우회 경로 차단)과
③(섹션 내부 whack-a-mole, "섹션당 정확히 1곳" 패턴) 동시 해소.

### §1.1 하드 `eps` 컷오프 → Sigmoid Soft-threshold

**문제**: 기존 `active = (t_flat[node_idx] > eps) & ...`는 불리언 마스크라 `t_flat`이 eps를 넘나드는
순간(Stage4에서 candidate 파트가 살아나는 전이 구간) 그래디언트가 계단식으로 끊긴다. 이것이 review
①에서 지적한 "두께를 깎아 계산 대상에서 아예 빼버리는 우회 경로"를 더 쉽게 만든다 — 살짝만 깎아도
전체 항이 통째로 사라지기 때문이다. Critic 라운드 합의대로 `eps` 마스킹 자체(review_v1_8.md §3
Ghost Gradient Hijacking 방지)는 유지하되, **하드 컷오프를 부드러운 가중치로 대체**한다.

```python
# AI_design_v1_9.py 상단 신규 헬퍼(로컬 오버라이드 패턴 유지)
def _soft_active_weight(t_flat, eps=0.1, transition_width=0.02):
    """[v1.10 §1.1] compute_smoothness_lse_v2의 하드 eps 컷오프를 대체.
    t_flat이 0(진짜 DELETED, z_gate=0으로 t_final이 정확히 0)이면 sigmoid((-0.1)/0.02)=sigmoid(-5)
    ≈0.0067로 사실상 0 — review_v1_8.md §3 Ghost Gradient Hijacking 방지 효과는 그대로 유지된다.
    반면 Stage4 워밍업 중 t_flat이 eps 근방을 서서히 통과하는 노드는 가중치가 매끄럽게 0→1로
    바뀌어, '살짝만 깎아도 손실 항 전체가 증발'하는 계단 효과(review §1의 우회 경로)를 없앤다."""
    return torch.sigmoid((t_flat - eps) / transition_width)
```

### §1.2 섹션 내부 집계를 Top-K로 교체

**문제**: `torch.logsumexp(d / temperature, dim=0)`는 섹션 내부에서 사실상 `max()`로 동작해 가장
심한 노드 1곳에만 그래디언트가 집중된다(review §3). Top-K(k>1) 평균으로 교체해 여러 노드가 동시에
그래디언트를 받도록 한다.

```python
# AI_design_v1_9.py — compute_smoothness_lse_v2() 전체 교체(§1.1+§1.2 통합)
SMOOTH_TOPK_FRACTION = 0.3     # [v1.10 §1.2] 섹션 내 활성 노드 중 상위 30%(최소 1개) 평균
SMOOTH_EPS_TRANSITION_WIDTH = 0.02   # [v1.10 §1.1] mm. eps 근방 soft-threshold 전이폭

def compute_smoothness_lse_v3(x, new_coords, neighbor_map, t_final, num_sections=NUM_SECTIONS,
                               eps=0.1, transition_width=SMOOTH_EPS_TRANSITION_WIDTH,
                               top_k_fraction=SMOOTH_TOPK_FRACTION,
                               temperature=SMOOTH_LSE_TEMPERATURE):
    """[v1.10 §1] compute_smoothness_lse_v2()(v1.9)를 대체한다.
    (1) 하드 eps 컷오프 -> sigmoid soft-threshold(§1.1): DELETED 노드는 여전히 가중치≈0으로
        배제되지만(Ghost Gradient 방지 유지), 전이 구간의 그래디언트 계단 현상을 없앤다.
    (2) 섹션 내부 logsumexp(사실상 max) -> Top-K 평균(§1.2): review_v1_9.md §3에서 확인된
        "섹션당 정확히 1곳만 남는" 패턴의 직접 원인을 제거한다."""
    node_idx  = neighbor_map['node_idx']
    left_idx  = neighbor_map['left_idx']
    right_idx = neighbor_map['right_idx']
    if node_idx.numel() == 0:
        return new_coords.sum() * 0.0

    t_flat = t_final.squeeze(-1) if t_final.dim() > 1 else t_final
    w_i = _soft_active_weight(t_flat[node_idx],  eps, transition_width)
    w_l = _soft_active_weight(t_flat[left_idx],  eps, transition_width)
    w_r = _soft_active_weight(t_flat[right_idx], eps, transition_width)
    active_weight = w_i * w_l * w_r          # (M,) — 셋 다 살아있어야 1에 가까움(기존 AND 조건과 동일한 의도)

    section_ids = x[node_idx, 5].long()
    p_i, p_left, p_right = new_coords[node_idx], new_coords[left_idx], new_coords[right_idx]
    deviation = p_i - 0.5 * (p_left + p_right)
    sq_deviation = torch.sum(deviation ** 2, dim=-1) * active_weight   # 가중 편차

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

**호출부 변경**: `train_step_multi()` 내부 `l_smooth_lse = compute_smoothness_lse_v2(...)`를
`compute_smoothness_lse_v3(...)`로 교체(인자 동일, 추가 인자는 모두 기본값 사용 가능).

**부작용/트레이드오프**: `active_weight`가 always-differentiable해지면서 DELETED 노드 근처에서도
아주 작은(≈0.007) 그래디언트가 흐른다 — review §3의 Ghost Gradient Hijacking을 "완전히 0"으로
차단하지는 않고 "무시할 수준으로 억제"하는 것이라는 점을 명시한다. `transition_width`가 너무 크면
(예: 0.1 이상) DELETED 노드의 가중치가 유의미하게 커져 버그가 재발할 수 있으므로, §4 검증 계획에서
`transition_width`별로 `active_weight`가 DELETED 노드에서 실제로 0.01 미만인지 반드시 확인한다.

---

## §2. [최우선] Candidate 경계 섹션 전용 연속 파트 완충 손실

**해소 원인**: review_v1_9.md ②(Stage4 게이트 보간이 candidate 파트만 다루고 인접 연속 파트의
재형성은 방치).

**설계**: 기존 `compute_shape_continuity_loss_v2()`(§3, v1.9)를 건드리지 않고, **별도의 추가 항**을
candidate 파트의 생존 상태가 바뀌는 경계 섹션에만 적용한다 — 기존 항(threshold=2.0mm, 전 구간 적용)
을 대체하는 것이 아니라 경계 섹션에서만 더 엄격한 threshold(예: 1.0mm)로 추가 페널티를 얹는 방식이라
이중 계산(double-penalize)이 아니라 "경계에서만 더 강하게" 작동한다.

```python
# AI_design_v1_9.py 신규 함수
BOUNDARY_CONTINUITY_THRESHOLD = 1.0   # [v1.10 §2] mm. 일반 threshold(2.0mm)보다 엄격
W_BOUNDARY_CONTINUITY = 0.5           # [v1.10 §2] 시작값 — §4에서 재보정

def find_candidate_boundary_sections(pruning_state):
    """[v1.10 §2] pruning_state['state']((17, n_cand), ALIVE=0/PENDING=1/DELETED=2 카테고리 값)
    에서 인접 섹션 간 상태가 달라지는 (k, k+1) 쌍을 찾는다. candidate 파트가 살아있다가 죽거나,
    죽어있다가 살아나는 경계를 모두 포함한다."""
    state = pruning_state['state']    # (17, n_cand), CPU
    boundary_pairs = []
    for k in range(NUM_SECTIONS - 1):
        if not torch.equal(state[k], state[k + 1]):
            boundary_pairs.append((k, k + 1))
    return boundary_pairs

def compute_boundary_continuity_loss(new_coords, section_ids, part_ids, boundary_pairs,
                                      threshold=BOUNDARY_CONTINUITY_THRESHOLD):
    """[v1.10 §2] candidate 생존 경계 섹션에서 연속 파트(0/1/2)에 한해 인덱스 1:1 대응 거리에
    더 엄격한 threshold를 적용 — compute_shape_continuity_loss_v2()와 동일한 인덱스 대응 방식을
    재사용하되(중복 메커니즘 도입 금지), 경계 섹션·연속 파트로만 범위를 좁히고 threshold만 낮춘다."""
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

**호출부 추가**: `train_step_multi()` 내부, `l_continuity` 계산 직후.

```python
# 추가
boundary_pairs = find_candidate_boundary_sections(pruning_state) if pruning_state is not None else []
l_boundary_continuity = compute_boundary_continuity_loss(new_coords, section_ids, part_ids, boundary_pairs)
contrib_boundary_continuity = W_BOUNDARY_CONTINUITY * l_boundary_continuity
```

`loss = (...)` 합산식에 `+ contrib_boundary_continuity`를 추가한다.

**부작용/트레이드오프**: `find_candidate_boundary_sections`는 매 스텝 `pruning_state['state']`를
비교하므로, candidate 상태가 바뀔 때마다(동적 분할 이벤트) 경계가 자동으로 재조정된다 — 별도의
캐싱/무효화 로직이 필요 없다(`pruning_state`가 이미 매 스텝 최신 상태). 다만 `torch.equal`을
17번(섹션 수) 루프 안에서 호출하는 것은 비용이 미미하지만, GPU-CPU 동기화가 있을 수 있으므로
`pruning_state['state']`가 CPU 텐서임을 그대로 유지한다(기존 코드도 CPU에 둠).

---

## §3. [중요] `W_SMOOTH_LSE` 단계적 스케줄

**해소 원인**: review_v1_9.md ①(가중치 12배 상향의 부작용)을 완화하면서도 §1의 개선 효과를 살린다.

**설계**: 이 프로젝트에는 이미 `continuity_weight_schedule()`(sigmoid 기반 성장 함수)이라는 검증된
패턴이 있다 — 새 스케줄링 방식을 발명하지 않고 동일한 패턴을 재사용한다.

```python
# AI_design_v1_9.py 신규 함수 — continuity_weight_schedule()과 동일한 시그니처 패턴
W_SMOOTH_LSE_FLOOR = 0.5   # [v1.10 §3] 학습 초반(Mp/면적 목표 우선) 최소값
                            # W_SMOOTH_LSE(3.0)는 이제 "도달 목표치"로 의미가 바뀐다

def w_smooth_lse_schedule(epoch, stage2_end, w_max=W_SMOOTH_LSE, w_min=W_SMOOTH_LSE_FLOOR, beta=0.05):
    """[v1.10 §3] continuity_weight_schedule()과 동일한 sigmoid 성장 패턴(새 메커니즘 도입 금지).
    stage2_end(ASWC_STAGE2_END=400) 근방부터 목표치(W_SMOOTH_LSE=3.0)로 서서히 올라간다 — Stage4
    게이트 확정 이전에는 형상 탐색이 Mp/면적 목표를 우선하도록 완화하고, 게이트가 정해지는 시점부터
    smoothness 억제력을 강화한다."""
    return w_min + (w_max - w_min) / (1.0 + math.exp(-beta * (epoch - stage2_end)))
```

**호출부 변경**: `train_step_multi()`에서 `contrib_smooth_lse = W_SMOOTH_LSE * l_smooth_lse`를
`current_w_smooth_lse * l_smooth_lse`로 바꾸고, `current_w_smooth_lse`는
`run_training_multi()`의 메인 루프에서 `curr_w_continuity`(기존 `continuity_weight_schedule` 호출
패턴)와 같은 자리에서 `w_smooth_lse_schedule(epoch, ASWC_STAGE2_END)`로 계산해 `train_step_multi`에
인자로 전달한다(기존 `w_continuity_val` sentinel 패턴 재사용).

**부작용/트레이드오프**: `W_SMOOTH_LSE_FLOOR=0.5`는 v1.8의 0.25보다는 높고 v1.9의 3.0보다는 훨씬
낮다 — v1.9가 "처음부터 세게 누르기"로 부작용을 겪었다는 교훈을 반영해, 초반에는 v1.8보다 살짝만
강화된 수준으로 시작하고 Stage4 근방에서 목표치까지 올라가도록 했다. §4에서 `beta`와 `w_min`을
재보정할 것.

---

## §4. 검증 계획

1. **Soft-threshold 단위 확인(최우선)**: `_soft_active_weight(t_flat=0.0, eps=0.1,
   transition_width=0.02)` ≈ 0.0067임을 실행 전 스크립트로 먼저 확인 — review_v1_8.md §3 버그가
   재발하지 않는지의 1차 방어선이다. `transition_width`를 0.02보다 크게 조정할 경우 반드시 이
   값을 재확인할 것.
2. **Top-K 회귀 확인**: `top_k_fraction=1.0`으로 주면 기존 `logsumexp`(v1.9) 동작과 유사한 경향을
   보이는지(완전히 동일하지는 않음 — mean vs logsumexp 차이) 확인해 새 구현이 극단값에서 합리적으로
   행동하는지 스모크 테스트.
3. **스모크 테스트(20 epoch)**: `l_smooth_lse`(v3), `l_boundary_continuity`, `current_w_smooth_lse`
   콘솔 로그 확인. 경계 섹션 수(`len(boundary_pairs)`)가 학습 중 합리적 범위(0~수개)인지 확인.
4. **회귀 테스트(전체 학습)**: `reports/v1_10/`에서
   - target miss 섹션 수가 v1.9(10개)보다 줄어드는지(v1.8의 7개 수준 이하가 목표)
   - 면적 감량률이 목표(5%) 근처로 돌아오는지(v1.9의 -11.6% 과다 감량 완화)
   - review_v1_9.md가 3D 데이터에서 특정한 7곳의 날카로운 꺾임(#06 섹션 12~15, #00 섹션 10·11·16)이
     실제로 완화되는지 — **이번에도 동일한 방식(3D html 직접 디코딩 + 꺾임각 계산)으로 재검증할 것**
     (review_v1_9.md의 분석 스크립트 로직을 재사용 가능).
5. **로깅 강화**: 이번에도 실행 로그를 저장할 것 — v1.5~v1.9 리뷰가 6회 연속 "저장된 콘솔 로그가
   없어 학습 중 곡선을 직접 확인 못했다"고 지적했다. `history` dict에 `l_smooth_lse`,
   `l_boundary_continuity`, `current_w_smooth_lse`, `active_weight.mean()`을 매 epoch 추가하고
   파일로 저장할 것.

---

## 부록 — Solver/Defense 라운드에서 나왔으나 기각되거나 수정된 제안

- **Gemini Solver 원안 [아이디어2] "eps 마스킹 완전 제거"**: Critic 라운드에서 Gemini·OpenAI 둘 다
  review_v1_8.md §3 Ghost Gradient Hijacking 재발 위험으로 기각 — Gemini 자신이 제시한 트레이드오프
  ("DELETED 노드는 마스킹 아웃해야 한다")가 사실상 원래 마스킹을 다른 이름으로 재도입하는 자기모순
  임을 두 모델 모두 지적했다. §1.1의 soft-threshold로 대체.
- **OpenAI Solver/Defense 원안의 완충 손실 코드**: `model.continuous_parts`, `adjust_for_continuity()`
  등 이 코드베이스에 존재하지 않는 API를 사용 — Critic·Defense 라운드에서 기각. §2는 실제
  `pruning_state['state']`/`compute_shape_continuity_loss_v2`의 인덱스 대응 패턴을 재사용해
  Judge가 직접 작성했다.
- **Gemini Defense의 `PruningOptimizationEngine(nn.Module)` 전체 구현**: `pruning_state['state']`
  (ALIVE=0/PENDING=1/DELETED=2 카테고리 텐서)를 연속적인 두께 값처럼 오해해
  `nn.Parameter(torch.randn(17, n_cand)*0.02+0.5)`로 재정의하는 등 실제 데이터 구조와 무관한
  코드였다 — Judge가 전면 기각하고 §1·§2를 처음부터 다시 작성했다.
- **W_SMOOTH_LSE 스케줄 방식**: Gemini는 코사인 스케줄, OpenAI는 선형 스케줄을 제안했으나, 이
  프로젝트에 이미 `continuity_weight_schedule()`(sigmoid 성장)이라는 검증된 패턴이 있어 §3에서는
  이를 재사용했다(새 메커니즘 도입 최소화 원칙).
