# command_v13.md — uni_section_v12 → v13 개선 지시서 (Constrained Dual-Ascent + 안정화된 프루닝)

> 출처: /synod design 세션 (Claude Judge + Gemini flash/high Architect + OpenAI o3/high Explorer, Solver+Critic 2라운드 실 병렬 교차검증, conf 95→92 / 88→85, 조기합의 미달로 Critic 라운드까지 진행)
> 기반: `docs/review/review_v12.md` (v8~v12 리뷰 결론)
> 목표: v12가 달성한 **Mp/collision 안정 수렴**을 유지한 채, 지금까지 4개 버전 모두 실패한 **실제 질량 감소**를
>       제약 기반(Augmented Lagrangian Dual-Ascent) 최적화로 강제하고, 미발동 상태였던 **프루닝 트리거를 실제로 작동**시킨다.

## 0. 판정 요약표

| review_v12.md 권장사항 | 판정 | 비고 |
|---|---|---|
| ①Mp/질량을 단일 가중합에서 분리 → Dual-Ascent | **채택(수정)** | 승수 고정 대신 **적응형 ρ** 필수(Critic 합의) — 고정 ρ는 Mp err 작아질 때/수술 직후 발산 |
| ②프루닝 트리거 재설계(EWMA) | **채택(수정)** | 순수 EWMA<0.08 단일 임계는 진동 시 오탐지 다발 → **히스테리시스 이중 임계** 필수 |
| ③z(1-z) entropy 정규화 | **채택(조건부)** | Mp err와 결합(coupling) 없이 단독 사용 시 조기 삭제로 하중경로 붕괴 위험 |
| ④softplus collision gap | **채택(수정)** | β 무제한 anneal 금지 — **β≤60 상한** 필수(원안 β→760은 collision gradient 소멸) |
| ⑤Cascade 옵티마이저 state 위생 | **채택(강화)** | 삭제 파트 키만 아니라 **step counter 포함 전체 리셋** 필수(Adam bias-correction 오정렬) |
| ⑥경량화-매끄러움 균형(Laplacian/TV) | **채택(수정)** | 프루닝 단계 on/off 이분법 대신 **항시 저가중 + 프루닝 구간 스케일업** |
| ⑦엣지케이스 가드(t_clamp, jitter) | **채택** | 이견 없음 |

> **핵심 결론(양 모델 수렴, Critic 라운드에서 구체화):** Dual-Ascent 전환 자체는 필수이나, review_v12.md 원안의
> 단순 고정 승수/고정 β/부분 optimizer 리셋으로는 **새로운 발산·NaN 실패 모드**를 만든다. v13은 review_v12.md의
> 7개 방향을 유지하되, 아래 §1~§7에 명시된 **적응형 스케줄과 안전장치**를 반드시 동반해야 한다.

---

## 1. [최우선] Constrained Dual-Ascent 손실 재정식화

v8~v12 공통 실패 원인: Mp gradient가 항상 질량 항을 압도(area +30~48%). 원인은 질량엔 Augmented-Lagrangian
승수가 있어도(μ_m) Mp 쪽엔 동등한 승수가 없는 **비대칭 구조**. v13은 목적함수를 전환한다.

### 1.1 목적함수 전환: "면적 최소화, Mp/충돌은 제약"
```python
# 목적(최소화): area   (Mp/collision은 더 이상 가중합 손실 항이 아니라 하드 제약)
f = sum(z_i * area_i for i in parts)

g_Mp  = abs(Mp - Mp_target) / Mp_target - tau_Mp      # tau_Mp = 0.01 (1%)
g_col = softplus_gap_violation(coords, t_detached)     # §4 참조, - tau_col
```

### 1.2 Augmented Lagrangian, 단 승수는 **적응형**
```
L = f
  + (rho_Mp/2)  * max(0, g_Mp  + mu_Mp/rho_Mp)^2
  + (rho_col/2) * max(0, g_col + mu_col/rho_col)^2
  + gamma_H * H(z)          # §3 entropy 정규화
  + beta_TV * TV(t)         # §6 두께 total-variation
```
- **초기값**: `mu_Mp=1.0, mu_col=1.0, rho_Mp0=200.0, rho_col0=200.0` (review_v12.md 원안 ρ=10은 Critic 검증 결과 근접-수렴 구간에서 과소, 초기 발산 구간에서 과대 — 200으로 재보정)
- **승수 업데이트**: 10 epoch마다
  ```
  mu_Mp  = max(0, mu_Mp  + rho_Mp  * g_Mp)
  mu_col = max(0, mu_col + rho_col * g_col)
  ```
- **적응형 ρ (Critic 필수 수정 — Explorer 반증 사례 반영)**: 고정 ρ는 그래프 수술 직후처럼 잔차(residual)가
  급변할 때(‖r‖: 0.03→0.9 같은 점프) 승수를 폭증시켜 다음 forward에서 NaN을 유발한다.
  ```
  rho_Mp  = clip(rho_Mp0  * min(1, ||g_area_grad|| / (||g_Mp_grad|| + 1e-9)), 200, 2000)
  rho_col = clip(rho_col0 * min(1, ||g_area_grad|| / (||g_col_grad||+ 1e-9)), 200, 2000)
  ```
  - **상한 하드캡**: `mu_Mp ≤ 1e5`, gradient clip norm=10 (Explorer 지적 — 무제한 승수는 NaN으로 직행)
  - **Constraint Slack**: 위반이 1.5% 미만이면 ρ를 더 키우지 않음(진동 방지)

### 1.3 승수 업데이트 주기와 EWMA 트리거의 시간축 분리
승수를 매 epoch 업데이트하면 outer/inner 루프 구분이 사라져 §2의 EWMA(window≈10)가 비정상(non-stationary)
신호를 보고 오발동한다. **승수 업데이트(10 epoch)와 EWMA 트리거 판정 주기를 동기화**하고, 승수 업데이트
직후 1~2 epoch은 EWMA 판정을 일시 정지(freeze)한다.

---

## 2. [프루닝 트리거] EWMA + 히스테리시스

v12의 실패 원인: "20epoch 연속 z<0.1"이 너무 엄격해 300epoch 내내 미발동(Patch2 z=0.56에서 정체).
단, review_v12.md 원안의 순수 EWMA(<0.08) 단일 임계도 Critic 검증 결과 **오탐지 후 삭제→복원 루프**를
유발함이 확인되어 히스테리시스로 보강한다.

### 2.1 스케줄
```
Stage 1 (epoch 0-30):    형상 warmup, 게이트 z=1 고정
Stage 2 (epoch 31-100):  Dual-Ascent 활성, 두께 언락, z 연속값 학습(트리거 없음)
Stage 3 (epoch 100-300): 프루닝 활성 — EWMA 트리거 + entropy 정규화 동시 가동
```
- Stage3 시작을 v12의 epoch150 → **epoch100**으로 앞당김(review_v12.md 권고, 프루닝 압력 시간 확보)
- `S_protect = {0, 1}` (Outer Hat, Inner Plate)만 보호. **Inner Hat(#06)을 프루닝 후보로 재개방**
  - ⚠️ Critic 경고: #06 즉시 완전삭제 허용 시 3개 후보(#06,#07,#08)가 동시 경쟁하며 Mp 급락 → 잔여 패치
    두께가 T_MAX(4mm)까지 폭주 위험. **완화책**: #06은 Stage3 진입 후 40 epoch은 `t_min_floor=0.3mm`로
    하한 고정(게이트 z를 곱하되 t_raw 하한을 0.3mm로 clamp)한 뒤에만 완전삭제(z→0) 허용.

### 2.2 EWMA + 히스테리시스 상태기계
```python
ewma_z[i] = 0.15 * z_cont[i] + 0.85 * ewma_z[i]   # alpha=0.15, window~10epoch

# 삭제 판정
if state[i] == 'ALIVE' and ewma_z[i] < 0.08:
    state[i] = 'PENDING_DELETE'   # 즉시 삭제하지 않고 후보 등록
# 복원 판정 (오탐지 방지)
if state[i] == 'PENDING_DELETE' and ewma_z[i] > 0.15:
    state[i] = 'ALIVE'            # 히스테리시스 — 재상승 시 취소
# 확정 삭제: PENDING_DELETE 상태가 20 epoch 유지되면 Cascade 발동
if state[i] == 'PENDING_DELETE' and pending_duration[i] >= 20:
    trigger_cascade(i)
```
- `pending_duration`은 `PENDING_DELETE` 진입 시 0으로 리셋, `ALIVE` 복귀 시 함께 리셋.
- 승수 업데이트 직후(§1.3) 1~2 epoch은 `ewma_z` 갱신을 skip(freeze)하여 다중-메커니즘 간섭 방지.

---

## 3. [z 딜레마 해소] Entropy 정규화 — Mp 오차와 결합

review_v12.md 원안(단순 z(1-z))을 그대로 쓰면 Critic이 지적한 대로 **Mp가 아직 불만족인데도 entropy가
게이트를 조기에 0/1로 밀어붙여 필수 하중경로를 삭제**할 위험이 있다. Mp err와 결합한다.

```python
H(z) = -mean(z*log(z+1e-6) + (1-z)*log(1-z+1e-6))   # z in (0.2, 0.8) 구간에서만 유의미한 값

alpha_ent = ALPHA_ENT_MAX * sigmoid(10 * (0.03 - mp_rel_err))   # Mp err<3%일 때만 강하게 활성
# ALPHA_ENT_MAX = 0.002
```
- entropy 항과 §2 EWMA 희소화(λ_s) 램프가 동시에 강하게 걸리면 서로 상쇄되어 z≈0.5에 **재고착**할 수 있음(Critic
  신규 실패모드). **완화책**: λ_s 지수 램프는 `alpha_ent`가 자기 최댓값의 절반 아래로 내려간 epoch부터만 가동.
  ```
  lambda_s(t) = 0.05 * 1.1^(t-100)   if t>=100 AND alpha_ent(t) < ALPHA_ENT_MAX/2   else 0
  ```

---

## 4. [Collision] Softplus Gap — β 상한 필수

review_v12.md 원안의 softplus(β anneal) collision proxy는 그대로 두면 β가 학습 후반 지나치게 커져
(원안 스케줄대로면 epoch300에 β≈760) 접촉 근방 gradient가 소멸해 **침투를 감지 못하는 채로 방치**된다(Critic
정량 반증: d<-5mm에서 미분값 1e-4 이하).

```python
t_detached = t_final.detach()          # v10 방식(비-detach)과 달리 gap 부호반전 미분불연속 회피
gap = compute_gap(coords, t_detached)
BETA_MAX = 60.0                        # 원안 상한 없음 → 60으로 고정 상한(Critic 필수 수정)
beta_t = min(BETA_MAX, beta0 * 1.5**(epoch // 40))
violation = F.softplus(beta_t * (gap_safe - gap)) / beta_t   # log1p(exp(beta*x))/beta, 수치안정형
l_collision = mean(violation ** 2)
```
- **collision 전용 Lagrange 승수 추가**: §1의 `mu_col`이 이 항에 직접 결합됨. review_v12.md/1차 설계에는
  collision 전용 승수가 없었다는 Explorer 지적을 반영해 §1.2 수식에 이미 포함시켰다(재확인 필요).
- **후처리 안전장치**: 매 50epoch마다 실제 gap<0(침투)이 관측되면 `beta0 *= 1.2`로 국소 보정(Critic 제안).

---

## 5. [Cascade 옵티마이저 위생] 전체 리셋(모멘트 + step counter)

v12 잠재 버그: part_id 재색인 후 옵티마이저 state가 재사용되어 삭제 파트의 모멘텀이 새 파트에 오염.
review_v12.md는 "삭제 파트 키만 제거"를 제안했으나, Critic 검증 결과 **Adam의 step counter(`t`)가 그대로면
살아남은 파라미터도 bias-correction이 어긋나 일시적으로 6배 과대 스텝**이 발생함(Explorer 유도).

```python
def rebuild_optimizer_on_cascade(model, old_optimizer_lr):
    # 삭제 파트 키만 골라내는 부분 리셋 금지 — 전체 재구성
    surviving_params = [p for p in model.parameters() if p.requires_grad]
    new_optimizer = torch.optim.AdamW(surviving_params, lr=old_optimizer_lr, weight_decay=1e-4)
    return new_optimizer   # state={}, step=0 부터 재시작 (1 epoch 성능 손실 감수, 안전 우선)
```
- warm-start로 이식하는 것은 **가중치(state_dict)뿐**이며 옵티마이저 모멘텀/step은 절대 이식하지 않는다(v12
  `transfer_weights`가 이미 이 원칙을 따르므로 유지, optimizer 쪽만 강화).

---

## 6. [형상 매끄러움] Laplacian/TV — 항시-저가중 + 구간 스케일업

review_v12.md 원안의 "프루닝 단계에서만 on"은 Critic이 지적한 대로, 게이트가 §2 히스테리시스로 복원(ALIVE
복귀)될 때 매끄러움 항이 꺼져있던 구간 동안 두께가 들쭉날쭉해진 상태로 남는 불연속을 만든다.

```python
w_laplacian = 0.1                                  # 항상 켜짐(기존 온/오프 이분법 폐기)
w_laplacian_prune = 0.5                             # Stage3(프루닝 구간)에는 5배 스케일업
l_smooth_total = w_laplacian * laplacian(coords) \
               + (w_laplacian_prune if stage==3 else 0) * laplacian(coords) \
               + w_TV * total_variation(t_final)
```

---

## 7. [엣지케이스 가드]

- **두께 하한**: `t_clamp = 0.15mm` — PNA bisection에 전달되는 두께가 0에 근접할 때 발생하는 non-smooth
  gradient 방지. 단, `z_gate < 0.02`인 노드는 PNA 호출 자체를 skip하고 `Mp_contrib=0`을 해석적으로 대입(완전
  삭제 후보에는 clamp가 오히려 완전삭제를 방해하므로 분기 처리, Critic 지적 반영).
- **Y-Snap jitter**: 노드 좌표 일치(coincidence) 시 Jacobian singular 방지를 위해 `epsilon=1e-4mm` 랜덤
  지터 추가.
- **NaN 가드**: `mu_Mp` 상한 `1e5`, 전역 gradient clip norm=10 (Explorer 지적, §1.2에 이미 포함, 여기서도
  구현 체크리스트로 재확인).

---

## 8. 구현 순서 (의존성 기준 4-Phase 로드맵)

Phase 간 의존성이 있으므로 반드시 순서대로 구현·검증한다. 각 Phase는 독립적으로 실행 가능한 검증 기준을 가진다.

```
Phase 1: ALDA 손실 + Softplus collision  (§1, §4)
   └─ 의존성 없음. v12.py의 가중합 손실을 §1.2 수식으로 교체하는 것이 우선 작업.
   └─ 검증: mu_Mp, mu_col이 위반 시 증가하는지 로그 확인. Mp err<1% 도달 후 area가 감소로 전환되는지 확인.

Phase 2: 프루닝 트리거 + entropy 정규화  (§2, §3)
   └─ Phase 1의 mu_Mp/g_Mp가 있어야 §3의 alpha_ent 결합이 의미를 가짐 → Phase 1 이후 착수.
   └─ 검증: ewma_z[Inner Hat]가 완만하게 감소하는지, H(z)가 게이트 양극화(0 또는 1 근접)에 따라 감소하는지 확인.
   └─ 20-epoch 미니런(v12 체크포인트에서 재개)으로 사전 검증: Mp_err≤5%, collision 감소 추세, mu_Mp<1e4,
      EWMA(z) 안정, 아직 삭제 없음 — 이 조건 확인 후에만 300epoch 풀런 진행.

Phase 3: Cascade 옵티마이저 전체 리셋  (§5)
   └─ 실제 프루닝 이벤트(Phase 2 트리거)가 발생해야 테스트 가능 → Phase 2 이후.
   └─ 검증: 수술 직후 optimizer.state_dict() 키가 생존 파라미터와 정확히 일치, step=0부터 재시작 확인.
   └─ 수동 테스트: #06을 강제로 삭제하고 50epoch 재개 → Mp가 회복되며 mu_Mp가 30epoch 내 하강 추세로 전환되는지 확인.

Phase 4: 엣지케이스 가드 + 매끄러움 균형  (§6, §7)
   └─ 나머지 항목과 독립적이나, 전체 파이프라인이 안정화된 후 회귀 검증하는 것이 효율적 → 마지막.
   └─ 검증: PNA bisection NaN/미수렴 경고 0건. 침투 스트레스 테스트(노드 +0.3mm 랜덤 변위 후 15epoch 내 gap>0 복귀).
```

---

## 9. 리스크 매트릭스 (Critic 라운드에서 도출된 신규 상호작용 실패 모드)

| 리스크 | 상호작용 메커니즘 | 완화책 |
|---|---|---|
| 승수 발산 → NaN | ρ가 빠르게 스케일되면 하드 페널티화되어 v10 수준으로 mass gradient 죽임 | `mu_Mp≤1e5`, grad clip=10, Constraint Slack(위반<1.5%면 ρ 동결) |
| entropy vs sparsity 충돌로 z≈0.5 재고착 | α_ent와 λ_s가 동시에 강하게 걸리면 서로 상쇄 | λ_s 램프는 α_ent가 최댓값의 절반 아래로 떨어진 후에만 가동(§3) |
| EWMA 조기 삭제(premature snap) | Stage3 진입 직후(epoch100-110) 확률적 변동으로 핵심 부재(Inner Hat) 오삭제 | Stage3 진입 후 10epoch은 EWMA 트리거 판정 자체를 freeze |
| softplus β 무한 anneal → 침투 방치 | β 과대 시 접촉 근방 gradient 소멸 | β≤60 하드캡 + 50epoch마다 실침투 감지 시 국소 보정(§4) |
| Adam bias-correction 오정렬 | 부분 optimizer 리셋 시 step counter 불일치로 순간 6배 과대 스텝 | 부분 리셋 금지, 전체 재구성(§5) |
| Inner Hat 동시경쟁 붕괴 | #06,#07,#08 동시 프루닝 후보 개방 시 Mp 급락 → 잔여 패치 두께 폭주 | #06은 40epoch 최소두께 고정(t_min_floor=0.3mm) 후에만 완전삭제 허용(§2.1) |
| 제조 공차 이하 미세 침투 | β 상한/오차허용 내에서도 0.05mm 수준 침투가 tolerance 통과 가능 | 50epoch마다 후처리 기하 점검, gap<0 발견 시 β 국소 보정 |

---

## 10. 사전 검증 체크리스트 (300-epoch 풀런 이전)

Explorer 제안 기반, Phase 2 착수 전 반드시 통과해야 하는 사전 검증:

1. **20-epoch 미니런** (v12 체크포인트에서 재개): Mp_err≤5%, l_collision 감소 추세, mu_Mp<1e4, EWMA(z) 안정, 삭제 없음
2. **수동 삭제 복구 테스트**: Inner Hat(#06) 강제 삭제 후 50epoch 재개 → Mp 회복, mu_Mp가 30epoch 내 하강 전환
3. **충돌 스트레스 테스트**: 노드 좌표 +0.3mm 랜덤 변위 → 15epoch 내 모든 gap>0 복귀 확인

세 항목 모두 통과한 뒤에만 300-epoch 풀 트레이닝(Cascade 포함)을 실행한다.

---

## 부록: 숙의 과정

- **Solver 라운드**: Gemini(pro→rate-limit→flash 폴백, thinking high) conf 95/can_exit=true — ALDA 공식화, 4-Phase 로드맵 제시. OpenAI(o3, reasoning high) conf 88/can_exit=false(재프롬프트 후 XML 포맷 확보) — 7개 상호작용 리스크와 3개 신규 조합 실패모드 도출.
- **Critic 라운드**: 양측 모두 can_exit=false 유지(92/85)이나 근본적 이견 없이 구현 세부(적응형 ρ, β 상한, optimizer 전체 리셋, entropy-sparsity 커플링, Inner Hat 단계적 삭제)에 대해 수렴 — Defense(법정) 라운드 없이 Judge(Claude) 종합으로 진행.
- **미해결로 남긴 사항**: GNN 메시지패싱을 통한 국소 제약 위반의 공간적 확산(비국소 진동) 문제는 양측 모두 "누락된 고려사항"으로만 지적했을 뿐 구체적 해법 합의는 없음 — v13 구현 중 실측 후 필요시 v14 검토 과제로 이월.
