# command_v1_14.md — 파트 간 겹침 해소 v1.14 작업지시서

작성일: 2026-08-15 (`/synod design` 세션, Gemini flash conf95(Solver)/98(Critic) + OpenAI o3
conf83(Solver)/86(Critic) — Solver 라운드에서 핵심 구조(Top-K 방향 풀링 + clearance 커리큘럼)에
즉시 수렴, Critic 라운드에서 "ALDA dual-ascent 복구를 필수로 켤지 vs 기본 OFF 실험 플래그로 둘지"
쟁점을 Gemini가 자신의 1라운드 입장(필수)을 스스로 정정하며 OpenAI 안(기본 OFF)으로 수렴,
`collision_spec` 자료구조 버그를 양 모델 공통 발견·정정. Defense 라운드는 생략)

근거 문서: `docs/idea/idea_v1_14.md`(구현 아이디어), `docs/review/review_v1_13.md`(근본 원인)
대상 파일: `AI_design_v1_13.py`, `uni-section/code/uni_section_v21.py`

---

## §0 개요 및 신규 발견 사실

`idea_v1_14.md`가 "필수"로 지정한 두 아이디어(direction-레벨 Top-K 풀링, clearance 커리큘럼)를
그대로 구현하되, **이번 design 세션에서 Claude가 학습 루프 코드를 직접 대조해 새로 발견한 사실**을
반영한다:

`l_collision`(`compute_collision_loss_v5()`의 출력)은 loss에 **직접 가중합되지 않는다**. 오직
`compute_alda_loss()`의 ALDA(Augmented Lagrangian) 제약항 `term_col`을 통해서만 반영된다
(`AI_design_v1_13.py` L1939-1967):

```python
l_collision = usv18.compute_collision_loss_v5(..., top_k_fraction=PHYS_TOPK_FRACTION)
L_alda, g_Mp_val, g_col_val = usv18.compute_alda_loss(
    area, target_area, pred_mp_total, target_mp_total, l_collision, alda_state)
L_alda_effective = L_alda * max(gate, 0.05)
# ...
loss = (contrib_phys + contrib_smooth + L_alda_effective + ... )
```

그리고 `alda_state = usv18.make_alda_state()`(`AI_design_v1_13.py` L2051)가 반환하는
`mu_col`(초기 1.0)/`rho_col`(초기 200.0)은 **학습 700epoch 내내 단 한 번도 갱신되지 않는다** —
`uni_section_v21.compute_rho_adaptive()`는 정의만 되어 있을 뿐 `AI_design_v1_13.py` 어디에서도
호출되지 않는다(전체 파일 grep으로 확인). 반면 물리(Mp) 손실은 `contrib_phys = weights['w_phys'] *
l_phys_total * s_phys`로 **ALDA와 별개로 직접** loss에 더해진다. 즉 collision은 정적 `rho_col=200`
고정 제약 하나로만 물리 손실과 경쟁해야 하는 구조적으로 불리한 위치에 있다.

---

## §1 Direction-레벨 Top-K 풀링 (idea_v1_14.md #1, 필수)

**대상**: `uni-section/code/uni_section_v21.py` L654 `compute_collision_loss_v5()`

**변경**: 기존 within-direction Top-K(30%, `top_k_fraction`)는 그대로 두고, direction들 "사이"의
집계 방식을 `total_loss / n_dirs`(153개 전체 단순 평균)에서 Top-K 평균으로 교체한다.

```python
def compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids, collision_spec, z_gate=None,
                               top_k_fraction=0.3, top_k_dir_fraction=0.25, min_dirs_to_keep=5):
    """
    [v1.14 §1] direction-레벨 집계를 단순 평균에서 Top-K 평균으로 교체 — 153개 direction 중
    극소수만 위반해도 나머지 정상 direction에 의해 l_collision이 희석되던 문제(review_v1_13.md #1)
    해소. compute_mesh_order_loss()(v1.8)/compute_smoothness_lse_v3()(v1.10)와 동일한 패턴.
    top_k_dir_fraction/min_dirs_to_keep은 uni_section_v21.py 상수 블록(신규 함수 정의보다 앞)에
    배치할 것 — 기본 인자값은 def 시점에 즉시 평가되므로 순서를 지켜야 한다(v1.10 §배치 주의 참고).
    """
    dir_losses = []
    for (sec_int, a, b), directions in collision_spec.items():
        ...  # 기존 순회 로직 그대로 유지, loss_dir 계산까지 동일
        for d in directions:
            ...
            dir_losses.append(loss_dir)   # 기존 total_loss += loss_dir; n_dirs += 1 대신 리스트 수집

    if not dir_losses:
        return torch.tensor(0.0, device=new_coords.device)
    dir_losses_t = torch.stack(dir_losses)
    k = max(min_dirs_to_keep, int(dir_losses_t.numel() * top_k_dir_fraction))
    k = min(k, dir_losses_t.numel())
    top_k_losses, _ = torch.topk(dir_losses_t, k=k, largest=True)
    return top_k_losses.mean()
```

**호출부 수정**: `AI_design_v1_13.py` L1939-1941
```python
l_collision = usv18.compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids,
                                               collision_spec, z_gate=z_gate_part_avg,
                                               top_k_fraction=PHYS_TOPK_FRACTION,
                                               top_k_dir_fraction=COL_TOPK_DIR_FRACTION)  # 신규
```
`COL_TOPK_DIR_FRACTION = 0.25`를 `AI_design_v1_13.py`의 v1.14 로컬 오버라이드 상수 블록에 추가
(기존 관례: 버전별 `# [v1.N] 로컬 오버라이드 상수` 섹션에 근거 주석과 함께).

**근거**: `g_col = l_collision - tau_col(0.02)`이고 `term_col = (rho_col/2)*relu(g_col+mu_col/rho_col)^2`
이므로, `l_collision`이 희석되어 0.01 미만이면 `g_col<0`이 되어 그래디언트가 **완전히 0인 dead
zone**에 빠진다(Gemini Solver 라운드 수식 유도). Top-K로 `l_collision`을 위반 direction들의 실제
스케일(예: 0.1~0.15 수준)로 끌어올리면 `rho_col=200`이 고정이더라도 dead zone을 벗어나
`∂term_col/∂l_collision = rho_col * max(g_col+mu_col/rho_col, 0)`이 0이 아닌 유의미한 값(예: 200 ×
0.135 ≈ 27)을 갖게 된다.

**주의(Critic 라운드 공통 지적)**: `top_k_dir_fraction`이 너무 작으면 학습 초반 다발적 위반이
무시될 수 있고, `min_dirs_to_keep=5`가 실제 위반 direction 수보다 훨씬 크면 정상 direction까지
포함돼 희석 효과가 되살아난다. 스모크 테스트 후 재보정 필요(값 자체는 실험값으로 명시).

---

## §2 Clearance Epoch 커리큘럼 (idea_v1_14.md #4 계열, 필수·저비용)

**대상**: `AI_design_v1_13.py` 학습 루프(`train_step_multi()` 또는 그 호출부인
`run_training_multi()`의 epoch 루프)

**변경**: `build_collision_spec()`이 학습 시작 시 1회 고정한 `clearance`(각 direction dict의
`'clearance'` 필드, 초기값 ≈ -0.05mm)를, epoch가 진행됨에 따라 목표치(0.5mm)까지 선형/코사인으로
완만히 강화한다.

```python
def clearance_curriculum_value(epoch, target_epoch=ASWC_STAGE2_END,
                                start_clearance=None, target_clearance=0.5):
    """
    [v1.14 §2] build_collision_spec()이 학습 시작 시 고정한 near-zero-slack clearance를
    epoch 진행에 따라 target_clearance(mm)까지 선형 보간. start_clearance는 None이면 각
    direction의 원래(초기 계산된) clearance 값을 그대로 시작점으로 사용한다(전역 상수로
    덮어쓰지 않음 — direction마다 초기 slack이 다르므로 개별 시작값 보존 필요, Critic 라운드 지적).
    target_epoch(기본 ASWC_STAGE2_END=400) 이후에는 target_clearance로 고정.
    """
    t = min(epoch / target_epoch, 1.0)
    if start_clearance is None:
        return None  # 호출부에서 direction별 원본값 사용
    return (1 - t) * start_clearance + t * target_clearance
```

**적용 위치 — 자료구조 정정 필수(Critic 라운드 공통 발견)**: `collision_spec`은
`{(sec_int, a, b): [direction_dict, ...]}` 형태의 **중첩 dict-of-list-of-dict**이며, 최상위 키가
튜플이다. Solver 라운드 두 모델의 코드 스케치(`collision_spec['clearance'] = ...`,
`for spec in collision_spec: spec['clearance'] = ...`) 모두 이 구조와 맞지 않아 각각 `KeyError`/
`TypeError`를 유발한다. **올바른 업데이트 패턴**:

```python
# epoch 루프 내, train_step_multi() 호출 직전
progress = min(epoch / ASWC_STAGE2_END, 1.0)
for key, dir_list in collision_spec.items():
    for d in dir_list:
        d['clearance'] = (1 - progress) * d['_init_clearance'] + progress * CLEARANCE_TARGET_MM
```

`build_collision_spec()` 호출 직후(1회) 각 direction dict에 `_init_clearance` 필드를 원본
`clearance` 값으로 백업해 두어야 한다(커리큘럼 갱신이 누적 덮어쓰기가 되지 않도록):
```python
collision_spec = usv18.build_collision_spec(x[:, :2], x[:, 6:7], part_ids, section_ids)
for dir_list in collision_spec.values():
    for d in dir_list:
        d['_init_clearance'] = d['clearance']
```

`CLEARANCE_TARGET_MM = 0.5`를 v1.14 로컬 상수로 추가.

**근거**: `review_v1_13.md` #3(1회성 고정 clearance ≈ -0.05mm)에 대한 직접 대응. 학습 초반에는
초기 형상의 near-zero slack을 그대로 인정해 불필요한 조기 페널티를 피하고, 후반으로 갈수록 설계
목표 마진(0.5mm)을 강제해 최종 형상의 실질 여유를 확보한다.

---

## §3 ALDA Dual-Ascent 복구 — 기본 OFF 실험 플래그 (신규 발견 대응, Critic 라운드 합의)

**Solver 라운드 이견**: Gemini는 "핵심 필수 작업"으로 상시 활성화를 제안했으나, **Critic
라운드에서 Gemini 자신이 입장을 정정**했다 — 이번 패치가 이미 Top-K 방향 풀링이라는 검증되지
않은 새 손실 역학을 도입하는데, 여기에 한 번도 실제 구동된 적 없는 `rho_col` 최대 10배 에스컬레이션
(200→2000)을 동시에 상시 켜면 두 개의 새 비선형 메커니즘이 상호작용해 예측 불가능한 발산/그래디언트
폭주 위험이 있다. OpenAI의 "기본 OFF, 명시적 opt-in, 롤백 경로 보장" 안을 두 모델 모두 최종 채택.

**대상**: `AI_design_v1_13.py` `train_step_multi()` epoch 루프 끝부분

**변경**:
```python
ENABLE_DYNAMIC_ALDA = False   # [v1.14 §3] 기본 OFF. True로 바꿔도 mu_col/rho_col 외 어떤 기존
                              # 동작도 변경되지 않는다 — False 유지 시 v1.13과 완전히 동일(collision
                              # 관련 ALDA 경로에 한해 — Top-K/커리큘럼은 §1·§2가 이미 별도 변경).

if ENABLE_DYNAMIC_ALDA and epoch % alda_state['update_every'] == 0 \
        and epoch >= alda_state['rho_freeze_epochs']:
    if g_col_val > alda_state['slack_threshold']:
        alda_state['mu_col'] = min(
            alda_state['max_mu'],
            alda_state['mu_col'] + alda_state['rho_col'] * g_col_val)
        alda_state['rho_col'] = min(alda_state['rho_col'] * 1.2, alda_state['rho_max'])
    # Mp 쪽(mu_Mp/rho_Mp)은 이번 패치 범위 밖 — 손대지 않는다.
```

로그에 `alda_state['rho_col']`, `alda_state['mu_col']`을 매 `update_every` epoch마다 출력해
에스컬레이션 여부를 추적 가능하게 한다.

**근거**: `alda_state`가 애초에 `update_every`/`rho_min`/`rho_max`/`slack_threshold` 필드를 갖고
있는 것으로 보아 dual-ascent 갱신이 설계 의도였을 가능성이 높으나, 실제로는 `AI_design_v1_13.py`
어디에서도 호출되지 않는 죽은 경로였다. §1의 Top-K 개선만으로 `l_collision`이 dead zone을
벗어나는지 먼저 확인한 뒤, 그래도 `g_col_val`이 `slack_threshold(0.015)`를 지속적으로 초과하면
이 플래그를 켜서 추가 에스컬레이션을 시도한다.

**롤백 기준**: `ENABLE_DYNAMIC_ALDA=False`로 되돌리면 즉시 v1.13과 동일한 ALDA 동작으로 복귀(단,
§1/§2 변경사항은 별개로 유지됨에 유의 — "v1.13과 완전히 동일"은 ALDA dual-ascent 경로에 한정된
표현임, Critic 라운드에서 OpenAI 자신의 표현을 재정정).

---

## §4 경계 접합부 clearance 마진 보상 (idea_v1_14.md #2, 선택)

**대상**: `uni-section/code/uni_section_v21.py` `build_collision_spec()`

**변경**: 두 파트 중 하나라도 `fix_x`/`fix_y`로 고정된 노드가 관여하는 direction에 한해, clearance
계산 시 마진을 추가 상향(+0.2mm)한다.

```python
BOUND_NODE_EXTRA_MARGIN_MM = 0.2   # [v1.14 §4] 선택 적용, §1·§2 결과 확인 후 필요 시 도입
```

**적용 조건**: §1(Top-K)과 §2(커리큘럼) 적용 후 재학습 결과에서도 특정 접합부(fix 노드 관여)에
겹침이 남을 경우에만 추가. §2의 clearance curriculum과 마진이 이중으로 누적되지 않도록 "curriculum
결과값에 마진을 더하는" 방식으로 결합(곱셈이 아니라 덧셈).

---

## §5 Out of Scope (명시적 제외)

- `compute_collision_penalty_unclamped()`의 추가 Huber 재설계/`W_COL_AUX` 피드백 스케일링
  (idea_v1_14.md #5) — 이번 패치는 미포함.
- 경계 노드 "조인트 쌍 공동 이동(rigid-body coupled displacement)" — idea_v1_14.md에서 이미 기각된
  "경계 노드 자유 이동" 아이디어의 제한된 변형. 별도 `/synod design` 세션에서 재검토.
- Mp 쪽 ALDA(`mu_Mp`/`rho_Mp`) dual-ascent 활성화 — collision 쪽만 이번 범위.
- 메시 해상도/토폴로지 변경.
- 기존 체크포인트(.pt) 마이그레이션(`alda_state`에 새 필드는 없으므로 즉시 문제는 아니나, §3
  플래그를 켠 채로 저장된 `rho_col`/`mu_col` 값을 다른 세션에서 재사용할 경우 별도 검토 필요 —
  OpenAI Critic 라운드 지적, 이번 v1.14 필수 항목은 아님).

---

## §6 검증 계획

1. **`l_collision` 스케일 변화**: §1 적용 직후 학습 로그에서 `l_collision`이 기존 대비(review_v1_13.md
   측정 기준 대략 0.01~0.03 수준 추정) 유의미하게 커졌는지(위반 direction들의 실제 스케일로
   회복됐는지) 확인.
2. **`g_col_val` 추이**: `tau_col(0.02)` 대비 `g_col_val`이 0 이하로 수렴하는 경향을 보이는지.
   §3을 켰다면 `alda_state['rho_col']`/`['mu_col']`이 `update_every`마다 상승하고 `g_col_val`이
   `slack_threshold(0.015)` 아래로 수렴하는지.
3. **`contrib_phys` 대비 collision 경로 스케일 대조**: `contrib_phys`, `L_alda_effective`,
   `contrib_col_aux`를 함께 로그로 남겨 물리 손실과 collision 관련 항들의 상대적 크기 비율을
   epoch별로 비교(절대 목표 비율은 미확정 — 실측 후 판단).
4. **최종 형상 물리 검증**: `reports/v1_14/final_section_v1_14.csv`를 review_v1_13.md와 동일한
   방법(노드±t/2 밴드 겹침 전수 검사)으로 재분석해 Part0-1/1-2/1-4 겹침이 해소됐는지 확인.
5. **부작용 확인**: 두께가 여전히 전 파트 T_MAX(2.3mm)로 균일 saturate되는지, Mp 달성률(target
   miss 비율)이 v1.13 대비 악화되지 않았는지.
6. **롤백 기준**: 위반 collision 지표(`l_collision`/`g_col_val`)가 v1.13 대비 20% 이상 악화되거나
   Mp 오차가 1.5배 이상 확대되면 §1의 `top_k_dir_fraction`/§2의 커리큘럼 종료 epoch을 재조정하고,
   §3(플래그 켠 경우)은 즉시 `ENABLE_DYNAMIC_ALDA=False`로 되돌린다.

---

<details>
<summary>숙의 과정</summary>

### 모델 기여
- **Gemini (Architect, Solver conf95→Critic conf98):** ALDA 제약식의 편미분을 직접 유도해 Top-K
  풀링이 "dead zone 탈출"이라는 구체적 수학적 효과를 갖는다는 것을 증명. Critic 라운드에서는
  자신의 1라운드 입장(dual-ascent 상시 활성화)을 스스로 재검토해 안전한 기본 OFF 안으로 정정.
- **OpenAI (Explorer, Solver conf83→Critic conf86):** dual-ascent를 처음부터 기본 OFF 실험
  플래그+명시적 롤백 경로로 설계해 리스크를 최소화했고, Critic 라운드에서 `collision_spec` 자료구조
  버그를 Gemini와 동시에 독립 발견. 체크포인트 마이그레이션·CI 회귀 테스트 등 이번 문서 범위를
  넘는 추가 고려사항도 제시(§5에 일부 반영).

### 해결된 주요 쟁점
1. "ALDA dual-ascent 복구를 필수로 켤 것인가, 실험 플래그로 둘 것인가?" → 기본 OFF 실험 플래그로
   수렴(Gemini가 Critic 라운드에서 자기 정정, OpenAI 원안 채택).
2. "clearance 커리큘럼 코드가 실제로 동작하는가?" → 두 모델의 원안 모두 `collision_spec` 자료구조
   (튜플 키 → dict 리스트)를 잘못 다뤄 런타임 에러를 유발함을 Critic 라운드에서 공통 발견,
   올바른 중첩 루프 패턴으로 정정(§2).

### 신뢰 점수
- Gemini: Solver 95, Critic 98 (High trust)
- OpenAI: Solver 83, Critic 86 (High trust)
- Claude(조율자): `AI_design_v1_13.py` 전체 grep으로 `alda_state` 미변경 사실 및 `compute_rho_adaptive()`
  미호출 사실을 직접 확인해 Solver 프롬프트에 선반영, `train_step_multi`/`compute_alda_loss`/
  `make_alda_state` 정확한 라인 번호·시그니처 대조

</details>
