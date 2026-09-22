# command_v15.md — uni_section_v10 → v15 개선 지시서 (ALDA 전환 + Mass Loss 비대칭화, 좁은 범위)

> 출처: /synod idea 세션 (Claude Judge + Gemini flash/high Architect + OpenAI gpt4o Explorer, Solver 라운드 실 병렬 교차검증, conf 95(can_exit=true) / 75(can_exit=false) — Critic 라운드 없이 Judge가 문서 근거로 쟁점 해소)
> 기반: `docs/review/review_v10.md`(a) 목록의 **A-1, A-2 두 항목만** 채택. 나머지(EWMA 프루닝, softplus
> collision, Cascade 그래프수술, entropy 정규화, Laplacian/TV, 로깅 패널 등)는 **명시적으로 이번 범위 밖**.
> 베이스: `uni_section_v10.py` (v11~v14의 프루닝/게이트/Cascade 기능은 가져오지 않음 — v10 순수 구조 위에
> ALDA만 얹는다)

## 0. 범위 선언

| review_v10.md 항목 | 이번 v15에 포함? |
|---|---|
| A-1 ALDA 전환 (area objective, Mp·collision 하드 제약) | **포함** |
| A-2 Mass loss 비대칭화 | **포함** |
| A-3~A-11 (게이트 이원화, EWMA 프루닝, softplus collision, 옵티마이저 전체재구성, entropy 정규화, Laplacian/TV, t_clamp, Y-Snap, 로깅) | **제외** — v13/v14의 나머지 기능은 이후 버전에서 별도 검토 |
| collision v5(비-detach, 기하) 자체 형태 | **그대로 유지** — softplus 전환 없음. l_collision 값을 ALDA 제약항의 g_col 계산에 그대로 재사용 |
| mesh order loss, 2단계 두께 게이트, 커리큘럼(s_phys/s_smooth) | **그대로 유지** |

> **Solver 라운드 쟁점 및 해소**: Gemini(conf 95)는 collision도 Mp와 동일하게 ALDA 하드 제약으로 전환할 것을
> 제안했고, OpenAI(conf 75, can_exit=false)는 "collision까지 하드 제약화하면 비대칭·불안정 위험"을 우려해
> 가중합 유지를 제안했다. **Judge 판정**: `review_v10.md`의 A-1 원문이 "Mp·**collision**을 하드 제약(적응형
> μ, ρ)으로 재정식화"라고 명시하므로, 사용자가 선택한 범위는 이미 collision을 포함한다. 따라서 **collision도
> ALDA 하드 제약(g_col)으로 전환**하는 Gemini안을 채택하되, OpenAI가 지적한 리스크(§5 R-1)는 완화책과 함께
> 명시적으로 남긴다.

---

## 1. 함수 변경/신설

### 1.1 `compute_mass_loss` 수정 (A-2)

```python
def compute_mass_loss(new_coords, t, edge_index, edge_attr, target_area):
    """
    [command_v15.md A-2] 비대칭 mass loss.
    area <= target_area: gradient 0 (경량화 방향 자유)
    area >  target_area: relu(area/target - 1)^2 (초과만 처벌)
    v10의 대칭 ((area-target)/target)^2 을 대체. 반환 시그니처는 v10과 동일하게 유지
    (area, l_mass) — 단, 이 l_mass는 §1.3에서 ALDA 도입 시 fallback/모니터링 용도로만 남고
    실제 학습 loss에는 compute_alda_loss()의 결과가 사용된다(§2).
    """
    src, dst = edge_index
    edge_type = edge_attr[:, 3]
    mask = (src < dst) & torch.isclose(edge_type, torch.zeros_like(edge_type))
    src, dst = src[mask], dst[mask]

    seg_len = torch.norm(new_coords[src] - new_coords[dst], dim=1)
    t_src = t[src].squeeze(-1)
    area = torch.sum(seg_len * t_src)

    l_mass = torch.relu(area / (target_area + 1e-12) - 1.0) ** 2
    return area, l_mass
```

### 1.2 `compute_alda_loss` 신설 (A-1)

SECTION 1(Loss Functions) 내, `compute_mass_loss` 바로 아래에 추가한다.

```python
def compute_alda_loss(area, target_area, pred_mp_total, target_mp_total,
                       l_collision, alda_state):
    """
    [command_v15.md A-1] Augmented Lagrangian Dual-Ascent.
    Objective: area/target_area (스케일 정규화, 무차원)
    Constraint 1 (Mp):        g_Mp  = |pred_mp - target_mp| / target_mp - tau_Mp
    Constraint 2 (Collision): g_col = l_collision - tau_col   (l_collision은 v10 collision v5 값 그대로)
    L = f + (rho_Mp/2)*relu(g_Mp + mu_Mp/rho_Mp)^2 + (rho_col/2)*relu(g_col + mu_col/rho_col)^2

    Returns: (L_alda, g_Mp_value:float, g_col_value:float)
      g_Mp_value/g_col_value는 detach된 float — run_training의 10epoch 승수 갱신 루프에서 사용.
    """
    f = area / (target_area + 1e-12)

    g_Mp = (pred_mp_total - target_mp_total).abs() / (target_mp_total.abs() + 1e-12) - alda_state['tau_Mp']
    g_col = l_collision - alda_state['tau_col']

    term_Mp = (alda_state['rho_Mp'] / 2.0) * torch.relu(
        g_Mp + alda_state['mu_Mp'] / alda_state['rho_Mp']) ** 2
    term_col = (alda_state['rho_col'] / 2.0) * torch.relu(
        g_col + alda_state['mu_col'] / alda_state['rho_col']) ** 2

    L_alda = f + term_Mp + term_col
    return L_alda, g_Mp.detach().item(), g_col.detach().item()
```

- `pred_mp_total`/`target_mp_total`은 `train_step` 내에서 이미 계산되는 `pred_mp_sections`(section별 Mp 합)와
  `target_mps` 합을 사용한다(§2 참조). asymmetric_huber_phys는 **그대로 유지**하되, 이제 "phys 항"이 아니라
  ALDA의 g_Mp 계산 입력으로도 쓰인다 — 즉 `l_phys_total`(비대칭 Huber)은 여전히 loss에 남고, g_Mp는 별도로
  상대오차(비율)만 사용한다(§1.3에서 병행 방식 명시).

### 1.3 `weights` dict / 가중합과의 관계 — **부분 대체, 완전 폐기 아님**

- `w_mass`, `w_collision`은 **더 이상 직접 가중치로 곱해지지 않는다.** 대신 ALDA의 `L_alda` 항이
  area(objective)와 collision-constraint를 함께 대체한다.
- `w_phys`(비대칭 Huber phys loss)는 **유지한다.** 이유: ALDA의 g_Mp는 "제약 위반량"만 반영하는 페널티이고,
  phys loss(Huber)는 Mp 수렴의 매끄러운 gradient 경로를 제공하는 별도 역할이다. 두 항을 병행하는 것이
  Gemini/OpenAI 모두 이견 없는 안전한 선택(제약만으로는 초기 수렴이 느릴 수 있음).
- `w_order`, `w_smooth`, `w_anchor`, `w_sat`은 **v10 그대로 유지**.
- 최종 loss 조립(`train_step` 내부):
  ```python
  L_alda, g_Mp_val, g_col_val = compute_alda_loss(
      area, target_area, pred_mp_total, target_mp_total, l_collision, alda_state)

  loss = (weights['w_phys'] * l_phys_total * s_phys      # 유지
          + L_alda                                        # area objective + Mp/collision 제약 (신규, w_mass/w_collision 대체)
          + weights['w_order']  * l_order                 # 유지
          + weights['w_smooth'] * l_smooth * s_smooth      # 유지
          + weights['w_anchor'] * l_anchor                 # 유지
          + weights['w_sat']    * l_sat)                   # 유지
  ```
- **`mass_gate` 로직은 폐기한다.** v10에서 mass_gate = thickness_gate × sigmoid(Mp오차 기반)로 mass 항을
  게이팅했으나, ALDA는 g_Mp 제약 위반 시 자동으로 페널티가 커지는 구조라 별도 게이트가 불필요(Gemini 판단
  채택). 단, §2.1의 Stage1 상호작용 리스크(R-2)를 반드시 함께 적용할 것.

---

## 2. `train_step` / `run_training` 통합

### 2.1 ALDA 상태 저장 위치

`run_training` 시작부, `weights` 초기화 근처에 `alda_state` dict를 만들어 **매 epoch `train_step`에
인자로 전달**한다(클래스화하지 않음 — v10의 기존 함수형 스타일과 일관성 유지, 상태는 `run_training` 스코프의
plain dict로 충분).

```python
alda_state = {
    'mu_Mp': 1.0, 'mu_col': 1.0,
    'rho_Mp': 200.0, 'rho_col': 200.0,
    'rho_Mp0': 200.0, 'rho_col0': 200.0,
    'tau_Mp': 0.01,   # 1%
    'tau_col': 0.02,
    'max_mu': 1e5,
    'rho_min': 200.0, 'rho_max': 2000.0,
    'update_every': 10,
    'slack_threshold': 0.015,  # 위반 1.5% 미만이면 rho 갱신 skip
}
```

`train_step(model, data, optimizer, target_mps, target_area, epoch, max_epochs, weights, curriculum, curriculum_ratio, collision_spec, alda_state)`
— 시그니처에 `alda_state`를 마지막 인자로 추가(기존 v10 인자는 순서 유지). 반환 `info` dict에 `alda_g_Mp`,
`alda_g_col`, `alda_mu_Mp`, `alda_mu_col`, `alda_rho_Mp`, `alda_rho_col`을 추가해 history에 기록한다
(A-11 로깅 패널은 범위 밖이지만, 이 수치 자체는 §4 검증 Gate에 필수이므로 콘솔 출력/history에는 남긴다).

### 2.2 10-epoch 승수 갱신 루프 — `run_training`의 epoch 루프 내부, `train_step` 호출 **직후**

```python
for epoch in range(max_epochs):
    info = train_step(model, data, optimizer, target_mps, target_area,
                       epoch, max_epochs, weights, curriculum,
                       curriculum_ratio, collision_spec, alda_state)
    ...  # 기존 history append, feasibility 판정 등 유지

    if epoch > 0 and epoch % alda_state['update_every'] == 0:
        g_Mp_val = info['alda_g_Mp']
        g_col_val = info['alda_g_col']

        alda_state['mu_Mp'] = min(alda_state['max_mu'],
                                   max(0.0, alda_state['mu_Mp'] + alda_state['rho_Mp'] * g_Mp_val))
        alda_state['mu_col'] = min(alda_state['max_mu'],
                                    max(0.0, alda_state['mu_col'] + alda_state['rho_col'] * g_col_val))

        # Constraint Slack: 위반이 1.5% 미만이면 rho는 갱신하지 않음(진동 방지)
        if g_Mp_val > alda_state['slack_threshold']:
            # norm_area_grad / norm_gMp_grad 비율 계산 (아래 §2.3 참조)
            alda_state['rho_Mp'] = clip(alda_state['rho_Mp0'] * ratio_Mp,
                                         alda_state['rho_min'], alda_state['rho_max'])
        if g_col_val > alda_state['slack_threshold']:
            alda_state['rho_col'] = clip(alda_state['rho_col0'] * ratio_col,
                                          alda_state['rho_min'], alda_state['rho_max'])

        print(f"[ALDA] epoch {epoch}: mu_Mp={alda_state['mu_Mp']:.2e} mu_col={alda_state['mu_col']:.2e} "
              f"rho_Mp={alda_state['rho_Mp']:.1f} rho_col={alda_state['rho_col']:.1f} "
              f"g_Mp={g_Mp_val:+.4f} g_col={g_col_val:+.4f}")
```

### 2.3 그래디언트 노름 비율(`ratio_Mp`, `ratio_col`) 계산 — 구현 주의

Gemini 원안은 `torch.autograd.grad(f, model.parameters(), ...)`로 objective/constraint 각각의 파라미터
그래디언트 노름을 별도로 계산할 것을 제안했다. v10의 `train_step`은 이미 `loss.backward()`를 1회만 호출하는
구조이므로, **10epoch마다의 rho 갱신 시점에만** 별도의 forward+backward를 1회 추가로 수행해 `f`, `g_Mp`,
`g_col` 각각에 대한 `model.parameters()` 그래디언트 노름을 구한다(`retain_graph=True`로 그래프 재사용,
`allow_unused=True`로 일부 파라미터가 특정 항에 기여하지 않는 경우 처리). 이 추가 backward는 optimizer.step()
을 호출하지 않는 **진단 전용 패스**이며, 10epoch에 1번만 실행되므로 학습 속도에 미치는 영향은 미미하다.

```python
ratio_Mp = (norm_f / (norm_g_Mp + 1e-9)).clamp(max=1.0).item()
ratio_col = (norm_f / (norm_g_col + 1e-9)).clamp(max=1.0).item()
```

---

## 3. 초기값·스케줄 표

| 파라미터 | 초기값 | 범위/제약 | 갱신 주기 |
|---|---|---|---|
| `mu_Mp` | 1.0 | [0, 1e5] | 10 epoch |
| `mu_col` | 1.0 | [0, 1e5] | 10 epoch |
| `rho_Mp` | 200.0 | [200, 2000] | 10 epoch (g_Mp>1.5%일 때만) |
| `rho_col` | 200.0 | [200, 2000] | 10 epoch (g_col>1.5%일 때만) |
| `tau_Mp` | 0.01 (1%) | 고정 | - |
| `tau_col` | 0.02 | 고정 | - |
| grad-clip norm | 10.0 | 전역(`clip_grad_norm_`), v10의 5.0에서 상향 | 매 step |
| Constraint Slack | 1.5% | 위반이 이보다 작으면 rho 동결 | - |

> v10의 기존 `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)`을 **10.0으로 상향**해야 한다
> (v13 §1.2 근거 — ALDA 승수 항의 gradient 스케일이 v10의 가중합보다 커질 수 있음).

---

## 4. 검증 기준 (Gate)

v15가 "성공"이라 판단하려면 300-epoch 풀런 기준 아래를 모두 만족해야 한다:

1. **질량 감소 전환**: 최종 area가 v10 baseline(1713mm², 초기 대비 +48%) 대비 **감소**, 이상적으로는
   초기 area(1157mm²) 이하 또는 근접.
2. **Mp/충돌 유지**: `Mp err < 1.5%` (tau_Mp=1%에 약간의 여유), `l_collision < 0.02` — v10 수준(0.03%/0.0159)
   에서 크게 후퇴하지 않을 것.
3. **승수 안정성**: `mu_Mp`, `mu_col`이 300epoch 내내 1e5 상한에 도달하지 않고 안정적으로 수렴(발산 없음).
4. **Sanity**: 학습 전체 구간에서 NaN/Inf 발생 0건.

사전 검증(풀런 이전 권장): 30epoch 미니런에서 `mu_Mp`/`mu_col`이 발산하지 않고, `g_Mp`/`g_col` 로그가
epoch10/20/30에서 출력되는지 확인.

---

## 5. 리스크 (이 좁은 범위에 특화)

| ID | 리스크 | 메커니즘 | 완화책 |
|----|--------|----------|--------|
| R-1 | Collision을 Mp와 동시에 하드 제약화 시 두 제약의 경쟁 | g_Mp, g_col이 서로 다른 스케일/빈도로 위반되면 rho_Mp/rho_col이 서로 다른 속도로 커져 한쪽이 다른 쪽을 억누를 수 있음(OpenAI 우려, R-1) | rho 상한을 동일하게 2000으로 공유 클립, 초기 30epoch은 rho 갱신 skip(warm-up)하고 Mp err/l_collision 로그로 관찰 후 갱신 시작 |
| R-2 | mass_gate 폐기 ↔ Stage1(두께 동결, epoch<128) 충돌 | ALDA의 area objective가 Stage1 동안에도 활성화되면, 두께가 사실상 게이트로 0에 가깝게 눌린 상태에서 area 항이 왜곡된 신호를 줄 수 있음 | ALDA 항 전체를 `thickness_gate_value(epoch)`와 동일한 스케줄로 warm-up: `L_alda_effective = L_alda * max(gate, 0.05)` 형태로 Stage1 동안 약화(완전 0은 금지 — 제약 자체는 항상 감시되어야 하므로 하한 0.05 유지) |
| R-3 | g_col 그래디언트가 무충돌 구간에서 0에 가까워 `ratio_col` 분모가 0에 근접 | collision v5는 위반이 없으면 loss가 0이라 grad_g_col norm이 매우 작아질 수 있음 | `norm_g_col + 1e-9` epsilon 외에, `ratio_col`을 `clamp(max=1.0)`로 상한을 둬 rho_col이 비정상적으로 커지지 않도록 방지(이미 §2.3에 반영) |
| R-4 | 10epoch마다 추가 backward pass(§2.3)가 `verify_thickness_gradient` 게이트나 기존 optimizer state에 부작용 | 진단용 backward가 우발적으로 optimizer.step()과 얽히면 이중 업데이트 위험 | 진단 backward는 반드시 `optimizer.zero_grad()` 이후 별도 블록에서 수행하고, 이 블록이 끝나면 다시 `optimizer.zero_grad()`로 정리 후 다음 epoch으로 진행 |
| R-5 | (b)형 리스크 — T_MAX=2.5mm 하드클램프(review_v10.md B-5)가 여전히 존재 | ALDA가 area를 강하게 누르면 두께가 T_MIN(0.1mm) 쪽으로 내려가려 하지만, 만약 특정 파트가 Mp 확보를 위해 오히려 두께를 늘려야 하는 상황이면 T_MAX 상한이 병목이 될 수 있음 | v15에서는 T_MAX/T_MIN을 v10 값 그대로 유지하고 관찰만 한다(수정은 범위 밖). 학습 종료 후 `t_per_part`가 T_MAX 근접(>2.3mm)인 파트가 있으면 후속 버전(v16+)에서 재평가 |

---

## 부록: 숙의 과정

- **Solver 라운드**: Gemini(flash, thinking high, conf 95, can_exit=true) — ALDA를 Mp+collision 동시 하드
  제약으로 정식화하는 완전한 함수/통합 방안 제시, mass_gate 완전 폐기 판단. OpenAI(gpt4o, conf 75,
  can_exit=false) — collision까지 하드 제약화하는 것에 대한 우려(비대칭·불안정 위험), mass_gate 유지 권고,
  부분 도입의 해석 리스크 지적.
- **쟁점 해소(Critic 라운드 생략, Judge 직접 판정)**: `review_v10.md` A-1 원문이 "Mp·collision을 하드
  제약으로 재정식화"라 명시하므로 사용자가 선택한 범위 자체에 collision 포함이 이미 확정되어 있음 — Gemini안
  채택. 단, OpenAI가 제기한 리스크는 폐기하지 않고 §5 R-1로 명문화해 완화책과 함께 남김. mass_gate 폐기에
  대한 OpenAI의 반론(R-2)도 Stage1 warm-up 완화책으로 반영.
- **미채택**: Gemini가 제안한 별도 `alda_state` 클래스화는 v10의 함수형 스타일과 맞지 않아 plain dict로
  단순화(Judge 판단).
