# command_v1_7.md — AI_design_v1_6.py → AI_design_v1_7.py: 노드 스파이크 수정 + 연속성 단순화 지시서

작성일: 2026-08-09 (`/synod idea` 세션, Gemini flash conf 98 + OpenAI gpt4o conf 75)
참조: `docs/idea/idea_v1_7.md`(설계 근거, 신뢰도 89%), `docs/review/review_v1_6.md`(실행 결과 리뷰),
`AI_design_v1_6.py`(기반 코드), `uni_section_v18.py`(**불변 — import만**)

> **앵커 규칙**: 본 문서의 위치 지정은 **함수/변수 이름 기준**이다. 라인 번호는 참고용이며
> 코드 변경으로 어긋날 수 있다.

> **불변 규칙**: `uni_section_v18.py`는 한 줄도 수정하지 않는다(`compute_smoothness_loss_angle()`
> 포함 — 신규 항은 순수 보강). §4(중립 초기화+지연 엔트로피), §7(R1~R3 롤백)의 상수·로직도
> 절대 수정하지 않는다.

> **정정 이력**: 구현 세션에서 OpenAI가 4가지 통합 리스크를 제기했다 — (a) DELETED 파트에
> 속한 노드가 정적 이웃 맵에서 계속 페널티를 받는 문제, (b) 초기 좌표 자체가 이미 애매한
> 경우 "얼린 채로 틀림"이 될 위험, (c) Stage4 호출부에서 신규 항 누락 위험, (d) 신규 항이
> 기존 `s_smooth` 커리큘럼 스케일을 공유하면 초반 스파이크를 놓칠 위험. Judge 판정: (a)는
> **범위 밖으로 기각**(기존 `compute_smoothness_loss_angle()`도 DELETED를 구분하지 않으므로
> 신규 항이 기존 베이스라인과 다르게 행동할 이유가 없다 — 동일 동작 유지가 일관성 있음),
> (b)는 §7 검증 계획에 관찰 항목으로 추가, (c)·(d)는 **채택**해 아래 §1.4·§1.3에 반영했다.

---

## §0. 목표 및 산출물

`AI_design_v1_6.py`를 복사해 **`AI_design_v1_7.py`** 를 만들고, idea_v1_7.md의 두 가지
아이디어를 구현한다.

| 결함 | 근본 원인 | 대응 |
|---|---|---|
| **노드 단위 스파이크(개별 정점이 주변 대비 튀어나옴)** | `compute_smoothness_loss_angle()`(불변)이 (a) `mean()`으로 희석, (b) x좌표 비교 실패 시 검사 자체를 건너뜀 | §1(정적 이웃 맵 + LogSumExp 보강 손실) |
| **v1.6 연속성 폐루프 래칫 버그**(`w_continuity` 상시 1.0 고정, review_v1_6.md에서 확인) | `max(w_base, w_adaptive)` 결합이 편도 래칫 | §2(폐루프 제거, 단순 하한선으로 대체) |

**적용 순서**: §1과 §2는 서로 독립적이므로 순서 무관, 동시 적용 후 함께 재실행해 검증한다.

---

## §1. 정적 이웃 맵 + LogSumExp 보강 스무스니스 손실

### §1.1 신규 상수 블록

기존 v1.6 상수 블록(`GATE_STEEP_MAX` 등) 뒤에 추가한다.

```python
# ══════════════════════════════════════════════════════════════════
# [v1.7] 로컬 오버라이드 상수 — 노드 단위 스파이크 보강 손실
# 근거: docs/idea/idea_v1_7.md, docs/review/review_v1_6.md, docs/command/command_v1_7.md
# ══════════════════════════════════════════════════════════════════
W_SMOOTH_LSE = 0.25   # 보강 손실 가중치 — 기존 weights['w_smooth']와 독립(§1.3 근거)
SMOOTH_LSE_TEMPERATURE = 0.01   # LogSumExp 온도. §7 검증 계획에서 실측 편차 스케일 기준 재보정 필수
```

### §1.2 정적 이웃 맵 구축 함수 (신규)

**대상**: `continuity_weight_schedule()` 정의 부근(모듈 레벨 함수 영역)에 추가.

```python
def build_static_neighbor_map(initial_coords, edge_index):
    """[v1.7 §1] compute_smoothness_loss_angle()(불변)의 두 결함 — (a) mean 희석,
    (b) 매 스텝 x좌표 재비교로 인한 검사 회피 — 을 근본적으로 우회한다. 초기(미변형)
    좌표 기준으로 좌/우 이웃을 딱 한 번만 결정해 영구히 고정한다 — 이후 좌표가 아무리
    뒤틀려도 이 노드가 "검사 대상에서 빠지는" 일이 없다.
    정확히 2개의 물리 이웃을 가지며 초기 좌표에서 좌/우가 명확히 갈리는 노드만 포함한다
    (파트 접합부처럼 이웃이 3개 이상인 노드는 원본 함수와 동일하게 제외 — 범위 확장 없음)."""
    N = initial_coords.shape[0]
    adj = {i: [] for i in range(N)}
    for u, v in edge_index.t().tolist():
        if u != v:
            adj[u].append(v)

    node_idx_list, left_idx_list, right_idx_list = [], [], []
    for i in range(N):
        neighbors = list(set(adj[i]))
        if len(neighbors) != 2:
            continue
        n1, n2 = neighbors
        x_i = initial_coords[i, 0].item()
        x_n1, x_n2 = initial_coords[n1, 0].item(), initial_coords[n2, 0].item()
        if x_n1 < x_i and x_n2 > x_i:
            node_idx_list.append(i); left_idx_list.append(n1); right_idx_list.append(n2)
        elif x_n2 < x_i and x_n1 > x_i:
            node_idx_list.append(i); left_idx_list.append(n2); right_idx_list.append(n1)

    device = initial_coords.device
    return {
        'node_idx': torch.tensor(node_idx_list, dtype=torch.long, device=device),
        'left_idx': torch.tensor(left_idx_list, dtype=torch.long, device=device),
        'right_idx': torch.tensor(right_idx_list, dtype=torch.long, device=device),
    }
```

### §1.3 LogSumExp 보강 손실 함수 (신규)

```python
def compute_smoothness_lse(new_coords, neighbor_map, temperature=SMOOTH_LSE_TEMPERATURE):
    """[v1.7 §1] 정적 이웃 쌍에 대한 Laplacian형 편차를 LogSumExp로 집계.
    mean()이 아니므로 소수의 심한 스파이크가 희석되지 않고(§0(a) 해결), 이웃 판정이
    고정돼 있으므로 스파이크가 생겨도 검사가 무효화되지 않는다(§0(b) 해결). LogSumExp의
    개별 노드 그래디언트는 softmax 출력이라 [0,1] 유계 — hard max와 달리 그래디언트
    폭발 메커니즘이 없다(design 세션에서 수식으로 증명, idea_v1_7.md §1.2 참고).
    [v1.7 §1 구현 세션 판정] weights['w_smooth']의 커리큘럼 스케일(s_smooth, 초반에 약함)을
    공유하지 않고 독립 상수 W_SMOOTH_LSE를 쓴다 — 이 항의 존재 목적 자체가 기존 항이 놓치는
    스파이크를 잡는 것이므로, 같은 초반 약화를 상속하면 목적이 무력화된다(OpenAI 지적,
    채택). DELETED 파트에 속한 노드도 그대로 포함한다 — compute_smoothness_loss_angle()
    (불변)도 DELETED를 구분하지 않으므로 기존 베이스라인과 동일하게 동작시킨다(OpenAI가
    마스킹을 제안했으나 범위 확장으로 판단해 기각)."""
    node_idx, left_idx, right_idx = (neighbor_map['node_idx'], neighbor_map['left_idx'],
                                      neighbor_map['right_idx'])
    if node_idx.numel() == 0:
        return torch.tensor(0.0, device=new_coords.device)
    p_i, p_left, p_right = new_coords[node_idx], new_coords[left_idx], new_coords[right_idx]
    deviation = p_i - 0.5 * (p_left + p_right)
    sq_deviation = torch.sum(deviation ** 2, dim=-1)
    return temperature * torch.logsumexp(sq_deviation / temperature, dim=0)
```

### §1.4 배선 — `run_training_multi()` / `train_step_multi()` / Stage4 호출부

**§1.4.1 맵 구축(1회, 루프 진입 전)** — `collision_spec = build_collision_spec(...)`와 동일한
위치·패턴.

```python
initial_coords = data.x[:, :2].clone().detach()
neighbor_map = build_static_neighbor_map(initial_coords, data.edge_index)   # [v1.7 §1]
```

**§1.4.2 `train_step_multi()` 시그니처** — `neighbor_map`을 필수 인자로 추가(sentinel
패턴 불필요 — `collision_spec`처럼 항상 존재하는 고정 데이터이므로 기본값 없이 위치/키워드
인자로 명시 전달).

```python
def train_step_multi(model, data, optimizer, target_mps, target_area,
                      epoch, max_epochs, weights, curriculum,
                      curriculum_ratio, collision_spec, alda_state,
                      pruning_state=None, segment_ids=None,
                      alpha_ent=ALPHA_ENT_V13, stage4_active=False, frozen_curriculum=None,
                      w_col_aux_val=None, w_continuity_val=None,
                      neighbor_map=None):                              # [v1.7 §1]
    ...
    l_smooth_lse = compute_smoothness_lse(new_coords, neighbor_map)    # [v1.7 §1]
    contrib_smooth_lse = W_SMOOTH_LSE * l_smooth_lse
    ...
    loss = (contrib_phys + contrib_smooth + L_alda_effective
            + contrib_order + contrib_anchor + contrib_sat
            + contrib_sparse + contrib_entropy + contrib_continuity
            + contrib_mass + contrib_col_aux + contrib_area_floor
            + contrib_smooth_lse)                                      # [v1.7 §1]
```

반환 dict에도 진단용으로 추가한다: `"l_smooth_lse": l_smooth_lse.item()`.

**§1.4.3 메인 루프 호출부** — `neighbor_map=neighbor_map` 전달.

**§1.4.4 Stage4 호출부** — **반드시 함께 수정**(구현 세션에서 OpenAI가 지적 — v1.6에서
Stage4 호출부가 일부 kwarg를 sentinel 폴백에 의존해 누락했던 전례와 달리, 이번 `neighbor_map`
은 sentinel 기본값이 없으므로 누락 시 `NoneType` 인덱싱 에러로 즉시 크래시한다). Stage4는
좌표 재수렴을 수행하는 마지막 단계이자 최종 시각화 형상을 결정하는 구간이므로, 오히려 이
보강 손실이 **가장 필요한 지점**이다 — 반드시 `neighbor_map=neighbor_map`을 전달할 것.

### §1.5 콘솔 로깅 추가

기존 epoch 로그 문자열 끝에 추가:
```python
f"| l_smooth_lse={info['l_smooth_lse']:.4f}"
```

---

## §2. 연속성 가중치 단순화 — 폐루프 제어 완전 제거

### §2.1 상수 정리

**삭제**: `CONTINUITY_TAU`, `CONTINUITY_ETA`, `CONTINUITY_EMA_BETA`.
**유지**: `CONTINUITY_W_FLOOR = 0.3`.

### §2.2 `continuity_weight_schedule()` 수정, `get_adaptive_continuity_weight()` 삭제

**대상**: `AI_design_v1_6.py`의 두 함수(L862-884 부근).

```python
# 변경 후 — get_adaptive_continuity_weight()는 함수 전체를 삭제한다.
def continuity_weight_schedule(epoch, stage2_end, w_max=1.0, w_min=0.05, beta=0.1,
                                w_floor=CONTINUITY_W_FLOOR):
    """[v1.7 §2] v1.5 sigmoid 그대로 + 하한선(review_v1_5.md 1차 원인 대응)만 추가.
    v1.6의 폐루프 제어(상태 저장, EMA, eta 튜닝)는 래칫 버그(review_v1_6.md에서 로그로
    확인 — w_continuity가 전체 620 epoch 동안 1.000에 고정)를 이유로 완전히 제거한다."""
    w_base = w_min + (w_max - w_min) / (1.0 + math.exp(beta * (epoch - stage2_end)))
    return max(w_base, w_floor)
```

### §2.3 `train_step_multi()` 내부 — 변경 불필요

v1.6의 sentinel 패턴(`w_continuity_val is None`이면 `continuity_weight_schedule()` 호출)을
그대로 유지한다 — 이미 단순화된 새 `continuity_weight_schedule()`을 호출하므로 자동으로
올바르게 동작한다.

### §2.4 `run_training_multi()` — 상태 초기화·갱신 코드 삭제

**삭제 대상**:
- `continuity_state = {"w_adaptive": 1.0, "last_l_continuity": 0.0}` 초기화 줄
- 메인 루프 내 `curr_w_continuity = get_adaptive_continuity_weight(epoch, ASWC_STAGE2_END, continuity_state)` 호출
- `train_step_multi()` 호출 후의 EMA 갱신 블록(`continuity_state["last_l_continuity"] = ...`)

**대체**:
```python
curr_w_continuity = continuity_weight_schedule(epoch, ASWC_STAGE2_END)   # [v1.7 §2] 단순화
info = train_step_multi(model, data, optimizer, target_mps, target_area,
                         epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                         collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                         alpha_ent=alpha_ent_schedule(epoch), stage4_active=False,
                         w_col_aux_val=curr_w_col_aux,
                         w_continuity_val=curr_w_continuity,
                         neighbor_map=neighbor_map)                       # [v1.7 §1][v1.7 §2]
```

콘솔 로그의 `w_continuity={curr_w_continuity:.3f}` 출력은 그대로 유지 — 이제 진짜로 감쇠하는
값이 찍혀야 정상이다(§7 검증 항목).

### §2.5 §7(v1.6) 스모크 테스트 코드 정리

`AI_design_v1_6.py` 하단의 `get_adaptive_continuity_weight()` 경계값 assert 블록은
함수 삭제와 함께 제거한다.

---

## §3. 절대 변경 금지 항목 (재확인)

- §4(중립 초기화+지연 엔트로피): `INIT_LOG_ALPHA_V14`, `ALPHA_ENT_START_EPOCH`,
  `ALPHA_ENT_ANNEAL_EPOCHS`, `alpha_ent_schedule()` 로직.
- §7(R1~R3 롤백): `DEATH_MP_THRESHOLD`, `DEATH_BURST_RATIO`, R1/R2/R3 raise 조건문.
- `uni_section_v18.py` 전체 — `compute_smoothness_loss_angle()`, `compute_gates()`,
  `compute_gate_temperature()`, `build_collision_spec()`, `_signed_projections()` 모두
  원본 그대로 호출만 한다. `compute_smoothness_lse()`는 이를 대체하는 것이 아니라 **보강**
  하는 신규 항이다 — 기존 `contrib_smooth`(`weights['w_smooth']*l_smooth*s_smooth`)는
  그대로 유지한다.
- v1.6의 §2(게이트 사전 첨예화, `GATE_STEEP_*` 상수와 `compute_gates_multi()` 로직)는
  이번 세션과 무관하므로 그대로 유지한다.

---

## §4. 파일 헤더 갱신

```
AI_design_v1_7.py — Node-Level Spike Fix + Continuity Simplification (v1.7)
─────────────────────────────────────
[v1.7 변경, docs/command/command_v1_7.md 실행]
  §1 정적 이웃 맵 + LogSumExp 보강 스무스니스 손실: compute_smoothness_loss_angle()(불변)의
     mean 희석·x좌표 검사 회피 결함을 우회하는 신규 보강 항 추가. 초기 좌표 기준으로 좌/우
     이웃을 1회만 고정, LogSumExp로 집계(그래디언트 [0,1] 유계 — 수식으로 검증됨). Stage4
     호출부에도 반드시 배선(누락 시 크래시).
  §2 연속성 가중치 단순화: v1.6의 폐루프 제어(get_adaptive_continuity_weight, 상태 저장,
     EMA)를 래칫 버그(review_v1_6.md 확인)를 이유로 완전 제거, 단순 하한선(max(sigmoid,
     0.3))으로 대체.
  구현: /synod idea 세션(Gemini flash conf98, OpenAI gpt4o conf75)에서 OpenAI가 제기한
  DELETED 파트 마스킹 제안은 기존 베이스라인과의 일관성을 이유로 기각, Stage4 호출부 누락
  위험과 신규 항의 독립 가중치 필요성은 채택했다.
그 외 로직은 AI_design_v1_6.py와 동일.
─────────────────────────────────────
```

---

## §5. 검증 계획

1. **단위 테스트**: `build_static_neighbor_map()`을 간단한 사각형 메쉬(4~6노드)로 실행해
   좌/우 인덱스가 기대대로 나오는지 확인. `compute_smoothness_lse()`에 고의로 하나의 노드만
   심하게 이동시킨 좌표를 넣어, `mean()` 버전 대비 손실 값이 유의미하게 커지는지(희석되지
   않는지) 확인.
2. **온도 파라미터 재보정(§1.1)**: `SMOOTH_LSE_TEMPERATURE=0.01`은 초기 추정값이다.
   실측 스파이크 편차가 수십 mm 스케일이므로 `sq_deviation`은 수백~수천(mm²) 단위가 될 수
   있다 — `sq_deviation/temperature`가 지나치게 커서 사실상 순수 max처럼 동작하는지,
   혹은 너무 작아 여전히 mean처럼 희석되는지 재실행 로그(`l_smooth_lse`)로 확인 후 재조정.
3. **스모크 테스트(20 epoch)**: 콘솔 로그에 `l_smooth_lse`와 (제대로 감쇠하는) `w_continuity`
   가 정상 출력되는지 확인.
4. **회귀 테스트(전체 학습)**: `reports/v1_7/`에서
   - `w_continuity`가 v1.6처럼 1.000에 고정되지 않고 epoch 400 이후 실제로 감쇠하는지
   - 최종 3D 형상에서 노드 단위 스파이크가 사라지거나 크게 줄었는지(육안 확인)
   - Mp 오차가 기존 대비 크게 나빠지지 않는지
   - Stage4 구간(마지막 좌표 재수렴)에서도 `l_smooth_lse`가 정상적으로 낮게 유지되는지
5. **관찰 항목(구현 세션 정정 이력 (b) 참고)**: 초기 좌표 자체에 이미 애매한 x좌표 배치가
   있는 노드가 존재하는지 `build_static_neighbor_map()` 실행 시 `node_idx` 개수를 전체
   노드 수(1581)와 비교해 확인 — 지나치게 적으면(예: 절반 이하) 초기 형상 자체의 이웃 판정
   품질을 별도로 재검토.
