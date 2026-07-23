# command_v10.md — uni_section_v8 → v10 개선 지시서

> 출처: /synod review 세션 `synod-20260710-103702-cee330`
> (Claude Validator + Gemini flash/high + OpenAI o3/medium, Solver→Critic→Defense 3라운드 실제 병렬 교차검증)
> 최종 신뢰도: **89%** | command_v9.md의 제안 5건(1, 2, 3, A, B)을 전수 검토한 결과.

## 0. 판정 요약표

| v9 제안 | 판정 | 비고 |
|---------|------|------|
| 1. T_MAX 2.5mm 클램프 | **수정 채택** | bias/scale 재보정 동반 필수 |
| 2. 동적 두께 margin + 제곱 페널티 | **수정 채택 (핵심)** | center-distance 대신 v8 segment 투영 유지, α 삭제 |
| 3. Mesh Order Loss (dx sign) | **수정 채택** | 1D dx → 2D 엣지 방향 내적으로 일반화 |
| A. 2단계 분리 학습 (하드 스위치) | **수정 채택** | freeze + 그룹별 optimizer 국소 리셋 + sigmoid 램프 |
| B. 비대칭 phys loss (5× err²) | **수정 채택** | 5× 그대로는 기각 → 비대칭 Huber형으로 대체 |

---

## 1. [§4.1] 두께 상한 2.5mm — 재보정 동반 (제안 1)

`T_MAX = 4.0 → 2.5`. 단, v8의 soft clamp는 `sigmoid(logit(t_init_frac) + delta_t)` 구조이므로 상한만 줄이면 동일 delta_t가 상대적으로 더 큰 두께 변화를 만들고, 초기 bias(0.07)가 초기 두께를 의도보다 키운다 (OpenAI 지적, 전원 동의).

```python
T_MIN = 0.1
T_MAX = 2.5                     # 핫스탬핑 상용 두께 한계
DELTA_SCALE = 1.35              # 1.5 → 1.35 (범위 축소에 비례해 완화)
nn.init.constant_(self.thickness_decoder[-1].bias, 0.03)   # 0.07 → 0.03
```

주의: Outer Hat 초기 두께 2.30mm는 새 상한 2.5mm에 근접 → logit이 상한 벽에 가까워 증가 방향 gradient가 작다. 이는 의도된 효과("두께 인플레이션 차단")이므로 그대로 둔다.

## 2. [§4.2] Collision v5 — 두께 연동 surface-gap + 제곱 소프트컨택 (제안 2, 최우선)

### 2.1 기하: v9 문서의 center-distance 행렬은 기각, v8 segment 투영 유지
v9 예시 코드의 `distance_matrix`(노드 중심 간 거리)는 긴 세그먼트 중앙부 침투를 놓친다. v8의 `_compute_segment_penetration_loss`(점→세그먼트 법선 투영, 부호 있는 projection)를 유지하고, **projection을 표면 gap으로 변환**한다 (3모델 만장일치):

```python
# projection: inner 노드가 outer 세그먼트 법선 방향으로 떨어진 부호 있는 거리 (v8 그대로)
# t_seg: outer 세그먼트의 두께 (t_final[세그먼트 시작 노드], detach 금지!)
# t_pt : inner 점의 두께 (detach 금지!)
CLEARANCE = 0.5   # mm — 상수 안전 간극 (t→T_MIN이어도 충돌 항이 꺼지지 않게 하는 floor)

gap = projection - (t_seg + t_pt) / 2.0 - CLEARANCE
violation = torch.relu(-gap) * valid_mask.float()

# 선형 sum → 제곱 mean ("스프링" 소프트컨택: 절벽 gradient·핑퐁 진동·zero-grad 맹인 모드 제거)
loss_pair = torch.mean(violation ** 2)     # v8의 torch.sum(선형) 대체
```

- **t는 detach하지 않는다.** 두께 증가 → gap 감소 → 충돌 페널티 증가의 gradient 경로가 v9 제안 2의 핵심 메커니즘("두께가 커지면 좌표를 밀어내거나 스스로 두께를 양보"). Mp 미달은 §5의 비대칭 phys loss가, 과도한 박육화는 mass loss가 견제하므로 "두께 축소 편법" 우려(검찰 측)는 균형 손실 구조가 흡수한다.
- **초기 침투 폭발 방지**: 초기 형상에서 gap<0인 쌍이 있으면 첫 epoch부터 제곱 손실이 폭발할 수 있음 → `violation = torch.clamp(violation, max=1.0)` 하드 클립을 둔다 (OpenAI 안).

### 2.2 pair_margins(§3.2 자동 산정) 폐지
두께가 gap에 직접 들어가므로 파트쌍별 고정 margin 딕셔너리(`compute_pair_margins`)는 **삭제**한다. 상수는 CLEARANCE 하나만 남는다. 단, 초기 형상에서 이미 `gap < 0`인 고정쌍(예: Part1–Part4, 초기 이격 0.6mm < (1.6+1.4)/2+0.5)이 있으면 CLEARANCE를 쌍별로 `min(0.5, 초기 gap 여유)`로 낮춰 "출발부터 위반" 상태를 피한다 — v8 §3.2의 취지를 CLEARANCE에만 국소 적용.

### 2.3 Ghost Gate α — 완전 삭제
- v8에서 α가 필요했던 이유는 margin이 thickness-blind였기 때문. v10은 t=0.1mm 파트의 margin 기여가 0.05mm에 불과해 얇은 파트의 밀어내기가 자연 감쇠한다 → α는 수학적 중복 (Gemini critic 98, defense 승소).
- `alpha`, `GHOST_THRESHOLD`, `GHOST_STEEPNESS` 및 `alpha_o * alpha_i` 가중 코드 제거. 모니터링용 alpha 히스토리도 delta_t/두께 히스토리로 대체.

### 2.4 검사 대상 확장: 인접쌍 → 전체 쌍
고정 순서 `[0,2,1,3,4]`의 인접쌍만 검사하면 비인접 파트(예: Patch2↔Outer Hat) 간 대변형 침투가 blind zone. 파트 5개 = 10쌍뿐이므로 **모든 unordered 쌍 (i<j)** 검사로 확장한다 (계산 부담 미미). `parts_order_in_sections`는 시각화 용도로만 유지.

## 3. [§4.3] Mesh Order Loss — 2D 엣지 방향 보존 (제안 3, v8 붕괴 직접 대응)

v9의 1D `sign(dx_base)` 방식은 수직에 가까운 세그먼트(dx≈0)에서 부호가 불안정해 노이즈 gradient를 만든다 (양 모델 공통 지적). **초기 엣지 단위벡터와의 내적 투영**으로 일반화한다:

```python
def compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr, eps=0.5):
    """파트 내부 엣지의 방향 반전(criss-crossing/뒤집힘) 방지 — v8 붕괴 근본 대책"""
    src, dst = edge_index
    mask = (src < dst) & (edge_attr[:, 3] == 0.0)      # 구조 엣지만
    e0 = base_coords[dst[mask]] - base_coords[src[mask]]        # 초기 엣지 벡터
    e0_hat = e0 / (e0.norm(dim=1, keepdim=True) + 1e-8)         # 초기 방향 단위벡터
    e_new = new_coords[dst[mask]] - new_coords[src[mask]]       # 현재 엣지 벡터
    proj = (e_new * e0_hat).sum(dim=1)                          # 초기 방향 성분 길이
    violation = torch.relu(eps - proj)                          # eps=0.5mm 최소 투영 길이
    return torch.mean(violation ** 2)
```

- dx만 보는 방식과 달리 어떤 기울기의 세그먼트에서도 동작하며, 뒤집힘(proj<0)과 과도 압축(proj<eps)을 동시에 차단.
- 가중치 `w_order = 1.0`, 커리큘럼 무관하게 **epoch 0부터 상시 활성** (collision과 동급의 하드 제약 성격).

## 4. [§4.4] 두께 2단계 학습 — freeze + 그룹별 국소 리셋 + 램프 (제안 A 수정)

하드 스위치 단독(모멘텀 불연속)도, 순수 sigmoid 램프 단독(전환기 리바운드)도 결함이 있어 통합안 채택 (Defense 라운드 양측 수렴):

```python
# Stage 1 (epoch 0 ~ 120): 두께 동결 — forward에서 delta_t_part에 gate 0.0 곱함
#   (thickness_decoder 파라미터는 존재하되 gradient가 0 → 좌표(형상)만으로 Mp 학습)
#   Stage 1 동안 w_mass = 0 (dt=0이면 mass 게이트가 무의미하게 켜져 면적을 과잉 구속함)
# Stage 2 진입 (epoch 121):
#   1) thickness_decoder 파라미터 그룹만 AdamW state 초기화 (좌표 헤드 모멘텀은 보존)
#      for p in thickness_group['params']: optimizer.state.pop(p, None)
#   2) 두께 그룹 lr = 0.3 × base lr
#   3) gate를 즉시 1.0이 아니라 15 epoch sigmoid 램프로 0→1 상승
gate = torch.sigmoid(torch.tensor(0.6 * (epoch - 128.0)))  # epoch 121~135 램프
delta_t_part = delta_t_part * gate
```

- optimizer 생성 시 param_groups를 `[coord+backbone, thickness_decoder]` 2개로 분리해 둘 것.
- Stage 경계(120)는 기존 curriculum_ratio(0.2, 0.7)와 정합: Stage A(collision/order 안정화) → phys ramp 구간 내에서 형상 수렴 → 두께 해제.

## 5. [§4.5] 비대칭 Huber형 phys loss (제안 B 수정)

v9의 `5×err²` 원안은 w_phys=10 하에서 gradient 스케일 급증으로 기각. v8의 `sqrt(|err|)`도 err→0에서 gradient 발산(수렴 직전 진동 원인)으로 문제. **비대칭 Huber형**으로 둘 다 해결:

```python
def asymmetric_huber_phys(pred_mp, target_mp, delta=0.05, under_w=2.0):
    """err<delta: 제곱(수렴 안정), err>delta: 선형(폭발 방지). 미달(undershoot)엔 under_w배."""
    err = (pred_mp - target_mp) / target_mp
    abs_err = err.abs()
    huber = torch.where(abs_err <= delta,
                        0.5 * abs_err ** 2 / delta,          # quadratic 구간
                        abs_err - 0.5 * delta)               # linear 구간 (C1 연속)
    w = torch.where(err < 0, under_w, 1.0)                   # 미달 2배 (5배는 과함)
    return w * huber
```

- undershoot 가중은 5×가 아닌 **2.0×** (Critic 라운드 절충: 1.5~2.5 사이 권고. T_MAX 2.5 제한으로 undershoot가 구조적으로 쉬워진 상태에서 5×는 두께 인플레이션 압력을 부활시켜 제안 1·2와 상충).
- `l_phys_total = sqrt(...)` 래핑은 제거 (Huber가 그 역할을 대체).
- w_phys=10 유지, grad clip 5.0 유지.

## 6. 통합 손실 구성 (v10 최종)

```python
weights = {
    'w_phys':      10.0,   # 비대칭 Huber (§5)
    'w_collision':  5.0,   # surface-gap 제곱 소프트컨택, α 없음, 전쌍 검사 (§2)
    'w_order':      1.0,   # 2D mesh order loss, 상시 활성 (§3) ← 신규
    'w_mass':       2.0,   # Stage 1에서는 0, Stage 2부터 기존 sigmoid 게이트 (§4)
    'w_smooth':     0.5,
    'w_anchor':     0.02,
    'w_sat':        0.01,
}
# max_epochs 300, AdamW lr 1e-3 (두께 그룹 Stage 2에서 0.3×), grad clip 5.0
# feasibility 기준: Mp err < 2% AND l_collision < 0.05 (유지) + mesh order violation == 0 추가 권장
```

## 7. 구현 순서 및 검증 게이트

1. §2 collision v5 (α 삭제, gap 공식, 제곱, 전쌍) — **단독으로 먼저 적용해 1회 학습** (효과 분리 확인)
2. §3 mesh order loss 추가 — v8 붕괴 재현 여부 확인 (형상 유지되면 성공)
3. §1 T_MAX 2.5 + 재보정, §5 Huber phys — 동시 적용
4. §4 2단계 학습 — 마지막에 적용 (앞 단계들이 안정되면 불필요할 수도 있음; Stage 없이 수렴하면 생략 가능)
5. 각 단계 후 기존 `verify_thickness_gradient` 게이트 통과 확인 + **신규**: 학습 전 초기 형상에서 `gap` 분포 출력(모든 쌍), gap<0 쌍은 CLEARANCE 국소 완화(§2.2) 적용 확인.

---

<details>
<summary>숙의 과정 (synod-20260710-103702-cee330)</summary>

### 모델 기여
- **Claude (Validator/Judge)**: v9 예시 코드(center-distance)가 v8 실제 구현(segment 투영)보다 기하적으로 후퇴함을 발견 → surface-gap 통합안 제시. OpenAI 검찰 측의 조작된 실험 로그 인용을 식별해 Trust Score 감점.
- **Gemini (Architect/Defense, conf 95→98→95)**: 제곱 소프트컨택의 수치 안정성 논증, α 삭제 변론 승소, 1D dx → 2D 내적 일반화 제안.
- **OpenAI (Explorer/Prosecutor, conf 93→79→83)**: T_MAX 축소 시 bias/scale 재보정 필요성, 전쌍 검사 확장, 초기 침투 클립, 5× 비대칭 손실의 상호작용 위험(제안 1·2와 상충) 발견, 그룹별 optimizer 리셋 구현안.

### 해결된 쟁점
1. α 게이트: 삭제 vs detach 유지 → **삭제** (두께가 gap에 내재화되어 중복; 검찰 측 증거 신뢰도 결격)
2. 비대칭 손실: 5×err² vs 1.5×sqrt → **비대칭 Huber (under 2.0×)**
3. 2단계 학습: 하드 vs 램프 → **freeze + 그룹별 국소 리셋 + 15epoch 램프 통합**
4. 충돌 기하: center-distance vs segment 투영 → **segment 투영 유지 + 두께 반영 gap**

### 신뢰 점수 (T = min(C×R×I/S, 2.0))
- Claude 1.3 (Good) | Gemini 1.6 (High) | OpenAI 1.2 (Good, Defense 라운드 0.5로 하락)

</details>
