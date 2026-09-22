# command_v16.md — uni_section_v15 → v16 개선 지시서 (게이트 이원화 + EWMA/히스테리시스 프루닝 트리거)

> 출처: /synod idea 세션 (Claude Judge + Gemini flash/high Architect + OpenAI gpt4o Explorer, Solver 라운드
> 실 병렬 교차검증, conf 95(can_exit=true) / 85(can_exit=false) — Critic 라운드 없이 Judge가 문서 근거로
> log_alpha 초기값 쟁점 해소)
> 기반: `docs/review/review_v10.md`(a) 목록의 **A-3(게이트 이원화), A-4(EWMA+히스테리시스 프루닝 트리거)
> 두 항목만** 채택. 베이스는 **v10이 아니라 v15**(`uni_section_v15.py`, ALDA + 비대칭 mass loss 적용 완료).

## 0. 범위 선언

| review_v10.md 항목 | 이번 v16에 포함? |
|---|---|
| A-1 ALDA 전환, A-2 Mass loss 비대칭화 | 이미 v15에 적용됨(전제 조건, 재작업 없음) |
| A-3 게이트 이원화(z_gate/z_open) | **포함** |
| A-4 EWMA+히스테리시스 프루닝 트리거 | **포함** |
| A-5 Cascade 그래프수술(실제 노드/엣지 삭제) + 옵티마이저 전체재구성 | **제외** — 이번엔 "소프트 삭제"(z 고정+로그)만, 실제 그래프 구조 변경은 다음 버전(v17+) |
| A-6~A-11(softplus collision, entropy 정규화, Laplacian/TV, t_clamp, Y-Snap, 로깅) | **제외** — 이후 버전에서 별도 검토 |

> **핵심 설계 결정(Solver 라운드 만장일치, conf 95/85 모두 동일 결론)**: A-4의 트리거가 확정(`PENDING_DELETE`
> 20epoch 유지)되어도 **실제 노드/엣지를 그래프에서 삭제하지 않는다.** 대신 해당 파트의 `log_alpha`를 큰
> 음수(-10.0)로 고정하고 상태를 `DELETED`로 표시해, 이후 모든 forward pass에서 `z_gate=0.0`이 강제되도록
> 한다("Soft-Zeroing"). 이는 A-5(Cascade 그래프수술 + 옵티마이저 전체재구성)를 이번 범위에서 제외하기 위한
> 의도적 경계이며, 두께/면적/충돌 관점에서는 물리적으로 완전 삭제와 동일한 효과를 내면서도 텐서 shape
> 불일치나 옵티마이저 state 오염 위험이 전혀 없다.

---

## 1. 파트 존재 게이트 z 도입 (A-3)

### 1.1 파라미터 등록 — `CGDN.__init__` 내부에 추가

```python
class CGDN(nn.Module):
    ...
    def __init__(self, ...):
        super().__init__()
        ...
        self.num_parts = 5  # #00 Outer Hat, #03 Inner Plate, #06 Inner Hat, #07 Patch1, #08 Patch2
        # [command_v16.md] init_log_alpha=2.0 (command_v12.md §1.3 근거: z≈0.96, "유연"하게 시작
        # — 3.0(z≈0.99, 거의 고착)보다 완만해 학습 초반부터 프루닝 방향으로 움직일 여지를 남김)
        self.log_alpha = nn.Parameter(torch.full((self.num_parts,), 2.0))
```

> **Solver 라운드 쟁점 해소**: Gemini는 `log_alpha` 초기값 3.0(z≈0.99)을 제안했고 OpenAI는 "작은 음수값"을
> 제안했으나 그 근거("초기 게이팅 없음을 위해")가 내부적으로 모순됨(작은 음수 log_alpha는 오히려 z를 0에
> 가깝게 만들어 즉시 게이팅되는 효과를 냄, 의도와 반대). **Judge 판정**: 원 설계 문서인
> `command_v12.md` §1.3의 "init_log_alpha: 2.5(z≈1 고착) → 2.0(z≈0.96, 유연) 권장" 원문을 근거로 **2.0**을
> 채택한다.

### 1.2 게이트 이원화 함수 — SECTION 1(Loss Functions) 근처에 신규 추가

```python
def compute_gates(log_alpha, training=True, temperature=0.5, gamma=-0.1, zeta=1.1):
    """
    [command_v16.md A-3] HardConcrete relaxation, 게이트 이원화.
    z_gate: 결정론적 — Mp/면적/collision 등 물리 계산에 사용 (variance 없음)
    z_open: 확률적(학습 시)/z_gate와 동일(평가 시) — 희소화 loss에만 사용
    """
    s_det = torch.sigmoid(log_alpha)
    z_gate = torch.clamp(s_det * (zeta - gamma) + gamma, 0.0, 1.0)

    if training:
        u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
        logistic_noise = torch.log(u) - torch.log(1.0 - u)
        s_stoch = torch.sigmoid((log_alpha + logistic_noise) / temperature)
        z_open = torch.clamp(s_stoch * (zeta - gamma) + gamma, 0.0, 1.0)
    else:
        z_open = z_gate
    return z_gate, z_open
```

**검증 기준(§5)의 "forward pass가 analytic gate와 diff<1e-6"은 `training=False`(평가 모드) 조건에서만
성립함을 명시한다** — 학습 모드에서는 z_open이 의도적으로 확률적이므로 diff가 존재하는 것이 정상이다.

### 1.3 forward pass 통합 지점

| 계산 | 수정 위치 | 방법 |
|---|---|---|
| 두께 `t_final` | `CGDN.forward()`, `delta_t_part` 계산 이후 | `t_final = z_gate_per_node * t_final` (파트별 z_gate를 노드 단위로 broadcast) |
| Area | `compute_mass_loss()` | `area = sum(seg_len * t_src)` — t_src가 이미 z_gate로 게이팅된 t_final을 참조하므로 **area는 자동으로 게이팅됨**(별도 곱셈 불필요, t_final 경로 하나로 일원화하는 것이 Gemini/OpenAI 공통 권장 — 이중 게이팅 방지) |
| Collision | `compute_collision_loss_v5()` | `violation = torch.relu(-gap).clamp(max=1.0) * z_gate[seg_part] * z_gate[pt_part] * valid.float()` — 사라지는 부재의 충돌 항 무력화 |
| DELETED 파트 강제 | `train_step()` 시작부 | `pruning_state['state'][i]=='DELETED'`인 파트는 `log_alpha[i].data.fill_(-10.0)`로 고정, 이후 z_gate가 자동으로 0에 수렴(§4.1) |

> **주의(Gemini/OpenAI 공통 지적)**: t_final에 z_gate를 곱하는 지점을 하나로 통일해야 한다. area·collision
> 양쪽에서 각각 별도로 z_gate를 곱하면(t_final에도 곱하고 area 계산에서 또 곱하는 식) z²만큼 과도하게
> 감쇠되는 이중 게이팅 버그가 생긴다. t_final 한 곳에서만 게이팅하고, area/Mp/collision은 이미 게이팅된
> t_final을 통해 자연히 전파되게 한다(collision만 예외 — violation 자체가 z_i*z_j를 명시적으로 필요로 하므로
> §1.3 표의 collision 행은 t_final 게이팅과 별개로 추가 필요).

---

## 2. ALDA 통합 및 보호 파트 (§2, A-3 연장)

### 2.1 희소화 항 추가 — `compute_alda_loss` 호출부(`train_step`)에서 조립

```python
CANDIDATE_PARTS = [2, 3, 4]   # #06 Inner Hat, #07 Patch1, #08 Patch2
S_PROTECT = [0, 1]            # #00 Outer Hat, #03 Inner Plate

l_sparse = z_open[CANDIDATE_PARTS].mean()   # [command_v16.md §2] 후보 파트만 희소화

loss = (contrib_phys + contrib_smooth + L_alda_effective
        + contrib_order + contrib_anchor + contrib_sat
        + weights['w_sparse'] * l_sparse)   # 신규 가중치 w_sparse
```

- `compute_alda_loss`의 `area` 인자는 §1.3에 따라 이미 z_gate로 게이팅된 `area`를 그대로 사용한다(함수
  시그니처 변경 없음).
- **보호 파트 강제(S_protect)**: `train_step` 시작부에서 `log_alpha[S_PROTECT]`는 옵티마이저 그래디언트를
  받지 않도록 별도 파라미터 그룹에서 제외하거나, 매 스텝 `z_gate[S_PROTECT] = 1.0`, `z_open[S_PROTECT] = 1.0`
  으로 강제 오버라이드한다(`compute_gates` 반환 직후 덮어쓰기 권장 — 구현 단순성).
- **Inner Hat(#06) 특례**: command_v13.md §2.1 근거를 계승해, Stage3(프루닝 활성 구간, epoch≥STAGE2_EPOCH=128)
  진입 후 40epoch 동안은 `t_min_floor=0.3mm`로 완전삭제를 유예한다:
  ```python
  if part_id == 2 and (epoch - STAGE2_EPOCH) < 40:
      t_final[part2_mask] = torch.clamp(t_final[part2_mask], min=0.3)
  ```

### 2.2 신규 가중치

`weights` dict에 `w_sparse` 추가(v13 λ_s 스케줄 참고, 고정값으로 단순화):
```python
weights['w_sparse'] = 0.05
```

---

## 3. EWMA + 히스테리시스 상태기계 (A-4)

### 3.1 상태 저장 위치 — `run_training`에서 초기화, `alda_state`와 별개의 `pruning_state` dict

```python
def make_pruning_state(num_parts=5):
    return {
        'ewma_z': torch.ones(num_parts),
        'state': ['ALIVE'] * num_parts,          # 'ALIVE' | 'PENDING_DELETE' | 'DELETED'
        'pending_duration': torch.zeros(num_parts),
        'ewma_alpha': 0.15,
        'thresh_low': 0.08,
        'thresh_high': 0.15,
        'confirm_epochs': 20,
        's_protect': [0, 1],
        'candidate_parts': [2, 3, 4],
    }
```

### 3.2 매 epoch 갱신 함수 — `run_training`의 epoch 루프, `train_step` 호출 직후

```python
@torch.no_grad()
def update_pruning_state(pruning_state, z_gate_per_part):
    for i in pruning_state['candidate_parts']:
        if pruning_state['state'][i] == 'DELETED':
            continue

        a = pruning_state['ewma_alpha']
        pruning_state['ewma_z'][i] = a * z_gate_per_part[i].item() + (1 - a) * pruning_state['ewma_z'][i]
        ez = pruning_state['ewma_z'][i]

        if pruning_state['state'][i] == 'ALIVE' and ez < pruning_state['thresh_low']:
            pruning_state['state'][i] = 'PENDING_DELETE'
            pruning_state['pending_duration'][i] = 0
        elif pruning_state['state'][i] == 'PENDING_DELETE':
            if ez > pruning_state['thresh_high']:
                pruning_state['state'][i] = 'ALIVE'
                pruning_state['pending_duration'][i] = 0
            else:
                pruning_state['pending_duration'][i] += 1
                if pruning_state['pending_duration'][i] >= pruning_state['confirm_epochs']:
                    pruning_state['state'][i] = 'DELETED'
                    print(f"[Pruning] part {i} DELETED at epoch (soft-zero, log_alpha frozen to -10.0)")
```

- `S_protect`({0,1})는 `candidate_parts`에서 애초에 제외되어 있으므로 이 루프에 진입하지 않는다(§2.1의
  강제 오버라이드와 이중으로 안전).
- `run_training`의 epoch 루프에서 `train_step` 호출 후: `update_pruning_state(pruning_state, z_gate.detach())`
  → `state[i]=='DELETED'`가 새로 확정된 파트에 대해 `model.log_alpha.data[i] = -10.0` 적용.

### 3.3 스케줄: Stage3(프루닝 활성) 진입 시점

v15는 `STAGE2_EPOCH=128`(두께 언락) 단일 스케줄만 갖고 있다. v16은 이를 3단계로 재해석하되 **기존
`thickness_gate_value`/`STAGE2_EPOCH` 자체는 수정하지 않는다**(범위 밖 코드 건드리지 않기):

```
Stage 1 (epoch < STAGE2_EPOCH=128):  z_gate=1 강제 고정(모든 파트), 프루닝 로직 비활성
Stage 2 (epoch >= 128):              게이트 학습 활성화, EWMA/히스테리시스 트리거 가동
```

- Stage1 동안 `log_alpha`는 옵티마이저 업데이트에서 제외(파라미터 그룹에 넣지 않거나 lr=0)하고 `z_gate=1.0`
  으로 강제 — v15의 두께 동결 구간과 정합성을 맞춰, "두께도 얼려놓고 존재 여부도 얼려놓는" 일관된 Stage1을
  유지한다(v11이 이 정합성 없이 stochastic z를 Stage1부터 물리에 흘려보내 발산했던 실패를 재발 방지).
- Stage2부터 `pending_duration` 카운트가 시작되므로, 최초 삭제 확정은 최소 epoch 128+20=148 이후에나
  발생한다.

---

## 4. 검증 기준 (Gate)

1. **게이트 일관성**: `model.eval()` 모드에서 `z_gate`와 `z_open`의 차이가 1e-6 미만(§1.2 명시대로 평가
   모드에서만 성립 확인).
2. **히스테리시스 단위테스트**: `ewma_z`를 0.05~0.12 사이에서 인위적으로 진동시키는 시뮬레이션에서 `DELETED`
   상태로 전이되지 않아야 하고, 0.08 미만을 20epoch 연속 유지하는 시뮬레이션에서는 정확히 `DELETED`로
   전이되어야 함.
3. **실제 프루닝 발동**: 300epoch 풀런에서 `candidate_parts`({2,3,4}) 중 최소 1개 파트가 `DELETED` 상태에
   도달(v11/v12가 실패했던 지점 — "트리거가 실제로 발동하는가").
4. **Mp/충돌 유지**: 프루닝 발동 이후에도 `Mp err < 1.5%`, `l_collision < 0.02` 유지(ALDA 제약이 계속
   작동함을 확인).
5. **area 추가 감소**: v15 결과 대비 최종 area가 더 감소(프루닝이 ALDA 단독보다 추가적인 경량화 효과를
   내는지 확인).

---

## 5. 리스크

| ID | 리스크 | 메커니즘 | 완화책 |
|----|--------|----------|--------|
| R-1 | v11 재발: stochastic z가 물리 계산에 유입되어 발산 | z_open을 물리 계산에 실수로 사용 | §1.3에서 t_final/area/collision은 **z_gate만** 사용하도록 명시적으로 고정, 코드 리뷰 체크리스트 항목화 |
| R-2 | v12 재발: 트리거 미발동 | 희소화 압력(w_sparse)이 너무 약하거나 Stage3 진입이 늦음 | Stage2 시작을 v15의 STAGE2_EPOCH=128과 동기화해 별도 지연 없이 즉시 가동, w_sparse=0.05는 프루닝 미발동 시 0.1로 상향 재시도(사전 검증 단계에서 확인) |
| R-3 | v13 이전 단일임계 EWMA 재발: 삭제→복원 진동 | 순수 임계값만 사용 | §3.2 히스테리시스(0.08/0.15 이중 임계) 필수 적용, 단일임계 재도입 금지(review_v10.md C-2) |
| R-4 | **ALDA area 왜곡**(Solver 라운드 공통 지적, v15와의 신규 상호작용) | 파트가 게이팅되어 area가 급락하면 `f = area_gated/target_area`가 인위적으로 작아져, ALDA가 "이미 목표 달성"으로 오판하고 생존 파트의 추가 두께 최적화 압력을 조기에 낮출 위험 | target_area는 **초기 스냅샷(고정값)을 그대로 유지**하고 재계산하지 않는다(이미 v15 구조가 이렇게 되어 있음, 변경 금지) — f가 낮아지는 것 자체는 "목표를 초과 달성했다"는 정확한 신호이므로 왜곡이 아니라 의도된 동작임을 확인. 다만 mu_Mp/mu_col이 이 상황에서 비정상적으로 낮게 유지되는지(=Mp 제약이 느슨해지는지)는 검증 §4-4에서 별도 관찰 |
| R-5 | 이중 게이팅 버그 | area/collision 각각에서 z_gate를 중복으로 곱함 | §1.3 표에서 t_final 단일 게이팅 지점 원칙 명시, collision만 명시적 예외로 문서화 |
| R-6 | S_protect 파트가 그래디언트를 통해 log_alpha가 흔들려 z_gate가 실수로 1 미만이 되는 경우 | §2.1의 "매 스텝 오버라이드"를 빠뜨리면 발생 | 오버라이드 코드를 `compute_gates` 호출 직후 단일 지점에 강제 배치, 단위테스트로 S_protect 파트의 z_gate==1.0 항상 확인 |

---

## 부록: 숙의 과정

- **Solver 라운드**: Gemini(flash — pro는 이번에도 rate-limit되어 폴백, thinking high, conf 95,
  can_exit=true) — Soft-Zeroing 아키텍처, 게이트 이원화 구현, ALDA 통합, EWMA/히스테리시스 상태기계를
  구체적 코드로 제시. OpenAI(gpt4o, conf 85, can_exit=false) — 대체로 동일한 아키텍처에 수렴했으나
  log_alpha 초기값 제안이 근거와 모순되어 신뢰도를 낮춤, ALDA area 왜곡 리스크(R-4)를 독립적으로 지적.
- **쟁점 해소(Critic 라운드 생략, Judge 직접 판정)**: log_alpha 초기값은 `command_v12.md` §1.3 원문
  ("2.5(고착)→2.0(유연) 권장")을 근거로 2.0 채택 — Gemini의 3.0(과 OpenAI의 내부 모순된 제안) 대신 원 설계
  문서 값을 우선.
- **공통 합의**: 그래프 수술(A-5) 제외, soft-zeroing 채택, t_final 단일 게이팅 지점, S_protect 강제 오버라이드,
  Inner Hat(#06) t_min_floor 유예.
