# command_v8.md — uni_section_v7 Mp 미수렴 원인 분석 및 v8 개선 방향

> 작성: /synod review 세션 `synod-20260710-094855-801db0`
> (Claude Validator + Gemini flash/high + OpenAI o3/medium — 3모델 실제 병렬 교차검증, 조기 합의)
> 최종 신뢰도: **95%** (Claude 95 / Gemini 98 / OpenAI 93, 전원 can_exit=true)

---

## 1. v7 결과 요약 (results/uni_section_v7.md)

| 항목 | 값 |
|------|-----|
| Target Mp | 47,421,470 N·mm |
| Final pred Mp | 11,182,746 N·mm (**오차 76.4%**) |
| Epoch 0 → 19 | MpErr 43.9% → 73.1%, Area 1241.6 → 591.2 mm² (**절반 이하로 붕괴**) |
| Epoch 59~250 | alpha mean/min **0.766 / 0.556에 완전 동결**, MpErr 74~81% 정체 |
| l_collision | 30~360 사이 발산적 진동, 끝까지 수렴 안 함 |
| Feasible epoch | **0회** (Mp err<2% AND collision<0.05 동시 만족 없음) |

핵심 증상 3가지: ① Mp가 목표로 올라가기는커녕 **내려가서** 정체, ② alpha(=두께의 함수)가 특정 값에 **정확히 동결**, ③ collision이 끝까지 진동.

---

## 2. 근본 원인 (3모델 만장일치)

### §2.1 [최우선·결정적] `ImplicitPNASolver.backward`가 두께 `t`의 gradient를 반환하지 않음

`uni_section_v7.py:125-156`:

```python
@staticmethod
def backward(ctx, grad_output):
    ...
    t_e  = t[u].squeeze(-1)          # t를 상수로 사용 (requires_grad 없음)
    ...
    (grad_coords,) = torch.autograd.grad(mp_direct, coords_g)
    return grad_coords * grad_output, None, None, None, None
    #                                 ^^^^ 두 번째 입력 t에 대한 grad = None
```

- `l_phys`(Mp 손실)는 **좌표에만** 역전파되고 **두께에는 gradient가 0**이다.
- 즉 "두께를 키워 Mp를 맞춰라"는 신호가 thickness_decoder에 **한 번도 도달한 적이 없다**.
- v7의 §2.1 수정(alpha 분리)·§2.3 soft clamp는 모두 두께 gradient "경로 위의" 문제를 고쳤지만, 경로의 **원천(backward)이 막혀 있어** 효과가 없었다.

**증거 사슬 (수치로 정확히 일치):**
1. 두께에 걸리는 유일한 gradient는 mass loss(area 축소 압력, `area*1e-6`)와 AdamW weight decay뿐 → delta_t가 -1.5(tanh 포화)로 붕괴.
2. delta_t=-1.5 포화 시 soft clamp 출력 t_new ≈ 0.545 mm → alpha = sigmoid(5×(0.545-0.5)) = **0.556** — 로그의 동결값과 정확히 일치.
3. tanh 포화 후 gradient ≈ 0 → epoch 59 이후 alpha가 소수점 셋째 자리까지 불변.
4. 두께 붕괴로 Area 절반 → Mp가 11M으로 추락, 회복 수단 없음. 커리큘럼이 s_phys를 1.0으로 올려도 **역전파되지 않는 손실의 가중치만 키운 것**이므로 무의미.

### §2.2 [구조적] 고정 노드 간극 0.6mm < collision margin 2mm — 기하학적으로 만족 불가능한 제약

- Part1(#03 Plate) 하부 플랜지: y=8.05 고정, Part4(#08 Patch2): y=7.45 고정 → 간극 **0.6 mm**.
- 두 노드 모두 fix_x=fix_y=1이라 절대 움직일 수 없는데 collision margin=2mm를 요구 → **1.4mm 침투가 수학적으로 영구 보장**.
- 만족 불가능한 제약이 s_collision=1.0(항상 최우선) + w_collision=5로 상시 강한 gradient를 뿜어내며, 움직일 수 있는 주변 노드들을 계속 흔든다 → epoch 50 이후 collision(30~360)·anchor(→33) 진동의 직접 원인.

### §2.3 [보조] mass loss의 무목표 축소 압력 + alpha detach

- `target_area=None`이면 `l_mass = area*1e-6` → "무조건 얇게"라는 단방향 압력. §2.1로 반대 신호(phys)가 차단된 상태에서 **유일하게 두께를 지배**.
- alpha가 collision loss에서 detach되어 있어 collision조차 두께를 회복시킬 경로가 없음(§2.1이 고쳐지기 전까지는 detach 유지가 안전하나, 이 조합이 "얇을수록 이득" 편향을 완성).

---

## 3. v8 수정 지시 사항

### §3.1 [최우선] `ImplicitPNASolver.backward`에 grad_t 복원

```python
@staticmethod
def backward(ctx, grad_output):
    coords, t, fy, y_pna_buf, edge_index = ctx.saved_tensors
    ...
    with torch.enable_grad():
        coords_g = coords.detach().requires_grad_(True)
        t_g      = t.detach().requires_grad_(True)      # ← 추가
        ...
        t_e  = t_g[u].squeeze(-1)                        # ← t_g 사용
        t_y_b = t_e * (dx_b / (L + 1e-12))               # y_top/y_bot/H도 t_g 경유
        ...
        mp_direct = torch.sum(Area_fy_b * (m_top_b + m_bot_b))

    grad_coords, grad_t = torch.autograd.grad(mp_direct, (coords_g, t_g))
    return grad_coords * grad_output, grad_t * grad_output, None, None, None
```

- IFT 논리는 동일(∂Mp/∂y_pna=0이므로 y_pna 고정 하 직접 미분만으로 충분) — t에 대해서도 그대로 성립.
- **검증 필수**: `torch.autograd.gradcheck` 또는 유한차분으로 ∂Mp/∂t 대조. `t += ε`시 Mp 증가 확인.

### §3.2 collision margin을 기하와 정합시키기

택 1 (권장 순):
1. **margin을 0.5mm로 축소** (0.6mm 고정 간극보다 작게), 또는
2. 파트쌍별 margin: (Part1, Part4) 쌍만 margin=0.4, 나머지 2.0 유지, 또는
3. Part1–Part4 고정 노드 접촉을 collision 검사에서 제외.

어느 쪽이든 "고정 노드만으로 위반이 확정되는 제약"이 없어야 collision이 0으로 수렴 가능해진다.

### §3.3 mass loss 재정식화

- `target_area`를 초기 단면적(≈1241 mm²)으로 지정하고 `|area - target|/target` 형태 사용, 또는
- Mp 오차가 5% 이내로 들어온 뒤에만 mass 항 활성화(gate).
- 목적: "무조건 얇게" 단방향 압력 제거. §3.1이 적용되면 phys가 반대 방향 gradient를 제공하므로 균형 회복.

### §3.4 두께 포화 방지 세이프가드

- delta_t가 ±1.4 이상(tanh 포화 근접)인 파트 수를 로그에 추가하고, 포화 시 경고.
- 필요 시 tanh 출력에 작은 leak(예: `0.95*tanh(x) + 0.05*x/3`) 또는 delta_t L2 정규화(약하게)로 포화 이탈 경로 확보.
- v7의 satParts 모니터링(alpha 기준)은 유지하되, **delta_t 포화 기준**이 더 조기 경보임.

### §3.5 커리큘럼 재점검 (§3.1 적용 후)

- §3.1로 phys→두께 경로가 열리면 v7 커리큘럼(collision=1.0 고정, phys 0.2→1.0 ramp) 자체는 유지 가능.
- 단 Stage A(epoch<50) 동안 두께가 mass/drift로 무너지지 않도록 §3.3 gate와 병행할 것.
- 학습 재실행 시 250 epoch 유지, feasibility 기준(2% / 0.05)도 유지 — §3.1~3.3이 맞다면 완화 불필요.

### §3.6 v8 반영 우선순위

| 순위 | 항목 | 기대 효과 |
|------|------|----------|
| 1 | §3.1 grad_t 복원 | Mp가 두께로 제어 가능해짐 — 미수렴의 직접 해소 |
| 2 | §3.2 margin 정합 | collision 진동 소멸, feasibility 판정 가능 |
| 3 | §3.3 mass 재정식화 | 두께 하한 붕괴 재발 방지 |
| 4 | §3.4 포화 세이프가드 | 조기 경보 |
| 5 | §3.5 커리큘럼 유지 | 변경 최소화 |

---

## 4. 예상 도달 가능성 점검 (v8 실행 전 확인 권장)

- Epoch 0 (t 초기값, alpha≈1)에서 pred Mp ≈ 26.6M (err 43.9%) — 목표 47.4M은 초기 대비 약 **1.78배**.
- 두께 상한 T_MAX=3.0mm는 초기 평균 두께(≈1.7mm)의 약 1.7~1.8배 → **두께만으로 목표에 아슬아슬하게 도달하거나 약간 부족**할 수 있음.
- 좌표(단면 높이 증가) 자유도가 함께 기여해야 하므로, w_anchor=0.05가 과도하게 좌표를 묶는지 v8 학습 로그에서 확인할 것.
- 필요 시 T_MAX 3.0→4.0 상향 또는 target Mp 도달 불가 시 명시적 경고 출력을 v8에 포함.

---

<details>
<summary>숙의 과정 (Synod 세션 기록: .omc/synod/synod-20260710-094855-801db0)</summary>

### 모델 기여
- **Claude (Validator)**: backward의 grad_t=None 최초 식별, alpha=0.556 ↔ t_new=0.545mm 수치 대조, Stage A에서 mass 항 비활성(s_mass=0)임에도 두께가 붕괴하는 메커니즘(backbone drift + weight decay) 보완 분석, 목표 도달 가능성(§4) 점검.
- **Gemini (Architect, conf 98)**: grad_t 복원 코드 청사진, margin<0.6mm 정합, mass gate 제안.
- **OpenAI (Explorer, conf 93)**: "커리큘럼은 역전파되지 않는 손실의 가중치만 키웠다"는 정식화, alpha detach의 편향 증폭 지적, t-포화 방지 prior 제안.

### 합의 방식
Round 1(Solver)에서 3모델이 독립적으로 동일한 근본 원인에 도달 (전원 conf≥93, can_exit=true) → 조기 합의로 Critic/Defense 라운드 생략.

### 신뢰 점수
- Claude 95 / Gemini 98 / OpenAI 93 → 최종 신뢰도 95%
</details>
