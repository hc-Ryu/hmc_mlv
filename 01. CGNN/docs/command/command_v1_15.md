# command_v1_15.md — 두께 연동 플랜지 오프셋 레이어 구현 작업지시서

작성일: 2026-08-17 (`/synod design` 세션, Gemini flash conf98(Solver)/98(Critic) + OpenAI o3
conf92(Solver, format enforcement 1회 적용)/88(Critic). Solver 라운드에서 Stage-1 처리 방침이 즉시
수렴했고, Critic 라운드에서 4개 쟁점 중 3개가 공통 수렴(테이퍼 구현·AND 조건·인덱스 공간 버그),
1개(`CLEARANCE_TARGET_MM`)는 Judge가 판정. Defense 라운드 생략)

근거: `docs/idea/idea_v1_15.md`, `docs/review/review_v1_13.md`, v1.14 실행 결과
대상: **신규 파일** `AI_design_v1_15.py`, `uni-section/code/uni_section_v23.py`
(프로젝트 관례: 버전마다 새 파일 생성, 기존 파일 수정 금지)

---

## §0 배경 — 왜 이 작업이 필요한가

v1.14 결과의 잔여 겹침(Plate-Inner 0.4119mm, Outer-Plate 0.0620mm)은 **설계 결함이 아니라 모델링
아티팩트**임이 실측으로 확증됐다(`idea_v1_15.md` §0):

- 초기 형상에서 일부 고정 노드쌍은 중심선 거리가 `(t_a+t_b)/2`와 1e-6mm 이내로 일치 = **표면 접촉**
  상태로 설계된 용접 플랜지다.
- 두께 증가분으로 예측한 겹침(+0.0620 / +0.4119mm)이 실측과 **소수점 4자리까지 일치**한다.
- 양쪽 노드가 BC 고정이라 좌표 자유도가 0이므로, collision loss는 **기하학적으로 만족 불가능한
  제약**을 강요해 왔고 그 결과 두께가 부당하게 깎였다(Mp 달성률 78.8% 정체).

**해결 방침**: 두께 변화에 연동해 접합 쌍을 절반씩 반대 방향으로 밀어내는 **미분 가능한 오프셋
레이어**를 `forward()` 안에 도입하고, 해당 쌍은 collision loss에서 제외한다.

---

## §1 신규 파일 생성

| 신규 파일 | 원본 | 변경 범위 |
|---|---|---|
| `uni-section/code/uni_section_v23.py` | `uni_section_v22.py` 사본 | `build_collision_spec()`, `compute_collision_loss_v5()` 2개 함수만 수정(§7) |
| `AI_design_v1_15.py` | `AI_design_v1_14.py` 사본 | import 교체, 상수 블록, 오프셋 레이어, Stage-1 처리, 진단/assert |

각 파일 헤더 docstring 최상단에 v1.15/v23 변경 이력을 기존 형식대로 추가한다.

---

## §2 신규 상수 블록

`AI_design_v1_15.py`의 v1.10 상수 블록 뒤에 배치한다.
**★ 배치 주의**: 아래 상수들은 신규 함수의 기본 인자값으로 참조되므로 반드시 해당 함수 정의보다
앞에 있어야 한다(기본 인자값은 def 시점에 즉시 평가됨 — v1.10에서 확인된 관례).

```python
# ══════════════════════════════════════════════════════════════════
# [v1.15] 로컬 오버라이드 상수 — 두께 연동 플랜지 오프셋
# 근거: docs/idea/idea_v1_15.md, docs/command/command_v1_15.md
# ══════════════════════════════════════════════════════════════════
FLANGE_X_TOL_MM      = 1.0   # 접합 쌍 후보 탐색 시 x 좌표 허용 오차
FLANGE_GAP0_MAX_MM   = 0.5   # 이 값 이하의 초기 표면 간극만 "접합 쌍"으로 인정.
                              # ★ 실측 근거(초기 형상 fixed-fixed 408쌍의 gap0 분포):
                              #   gap0<=0.5mm 147쌍 / 0.5<gap0<=2.0 160쌍 / gap0>2.0 101쌍.
                              #   Outer-Plate는 max 33.17mm까지 존재해(명백히 비접합) 임계가 필수다.
                              #   0.5mm는 v1.14에서 실제 겹친 쌍(Plate-Inner gap0<0.412,
                              #   Outer-Plate gap0<0.062)을 모두 포함한다. §10에서 재보정.
FLANGE_TAPER_N       = 2      # 플랜지 노드에서 자유 영역 쪽으로 오프셋을 감쇄시킬 이웃 개수
FLANGE_TAPER_WEIGHTS = (1.0, 2.0 / 3.0, 1.0 / 3.0)   # [본인, +-1, +-2] 가중치
FLANGE_ASSERT_TOL_MM = 0.01   # export 후 간섭 하드 assert 허용 오차
```

---

## §3 접합 쌍 사전 계산 — `identify_mating_flange_pairs()`

**대상**: `AI_design_v1_15.py` 신규 함수. `build_collision_spec()`과 동일하게 **학습 시작 시 1회**
호출하고 결과를 모델 버퍼로 등록한다.

**탐지 규칙** (idea_v1_15.md ④ "초기 표면 간극 보존"):
1. 서로 다른 파트 `a`, `b`의 같은 floor 노드 중, **양쪽 모두 `fix=1`** 인 것만 후보.
2. `|x_i - x_j| < FLANGE_X_TOL_MM` 인 것 중 `|y_i - y_j|` 최소인 상대를 짝으로 선택.
3. `dist0 = ||p_i - p_j||`, `gap0 = dist0 - (t_a,init + t_b,init)/2`
4. `gap0 <= FLANGE_GAP0_MAX_MM` 인 쌍만 채택(그 이상은 접합부가 아니라 단순히 멀리 있는 노드).
5. 단위 법선 `n_ij = (p_j - p_i) / dist0`.

**캐싱할 버퍼** (`register_buffer`로 등록, 학습 중 불변):

```python
# P = 채택된 접합 쌍 개수
'flange_idx_a'   : (P,)   long   — 전역 노드 인덱스
'flange_idx_b'   : (P,)   long
'flange_normal'  : (P, 2) float  — i -> j 단위 법선
'flange_gap0'    : (P,)   float  — 진단용(오프셋 계산에는 불필요, 검증에 사용)
'flange_part_a'  : (P,)   long   — hard 삭제 게이팅용
'flange_part_b'  : (P,)   long
'flange_sec'     : (P,)   long
# 테이퍼용 (§6)
'taper_idx_a'    : (P, 1+2*FLANGE_TAPER_N) long   — [i, i-1, i+1, i-2, i+2] (범위 밖은 i로 클램프)
'taper_w'        : (1+2*FLANGE_TAPER_N,)   float  — [1.0, 2/3, 2/3, 1/3, 1/3]
'taper_idx_b'    : (P, 1+2*FLANGE_TAPER_N) long
```

**이웃 인덱스 산정**: `build_bpillar_from_csv()`가 `for floor: for part: for point` 순으로 노드를
쌓으므로 **같은 (part, floor) 그룹 안에서 노드 인덱스가 연속이고 point_idx 순서**다. 따라서 k번째
이웃은 단순히 인덱스 ±k이며 **그래프 순회가 불필요하다**. 그룹 경계를 넘는 ±k는 자기 자신(i)으로
클램프하고 해당 가중치를 0으로 만든다(경계에서 오프셋이 이웃 파트로 새는 것을 방지).

---

## §4 미분 가능 오프셋 레이어 — `CGDN17.forward()`

**대상**: `AI_design_v1_15.py` `CGDN17.forward()`.

**★ 삽입 위치**: `t_raw` 계산(v1.14 기준 L847) **직후**, `return`(L859) **직전**.
좌표는 L796에서 이미 확정되지만 `t_raw`는 L847에서야 존재하므로, 두께 구동 오프셋은 이 구간에만
삽입할 수 있다.

```python
        t_raw = t_min + (t_max - t_min) * torch.sigmoid(t_initial_logit + delta_t_part)

        # ── [v1.15 §4] 두께 연동 플랜지 오프셋 ──────────────────────────────
        # 초기 표면 간극 gap0 = dist0 - (t_a0+t_b0)/2 를 학습 내내 보존한다:
        #   dist 가 (Δt_a + Δt_b)/2 만큼 벌어지면 gap 은 gap0 로 유지된다.
        # 구동 두께는 t_raw(제조 두께)다 — t_final(=t_raw*z_gate)을 쓰면 z_gate 가 0~1의
        # 연속 완화값(존재 '확률')이므로 "두께가 절반인 판재" 같은 비물리 기하가 만들어진다.
        # 삭제된 파트는 두께를 흐리는 대신 pair 자체를 끈다(아래 pair_alive).
        flange_offset = torch.zeros_like(new_coords)
        if self.flange_idx_a.numel() > 0:
            t_init_node = x[:, 6]                                   # (N,)
            dt = (t_raw.squeeze(-1) - t_init_node)                  # (N,) Δt (파트 내 균일)
            dt_a = dt[self.flange_idx_a]                            # (P,)
            dt_b = dt[self.flange_idx_b]

            # hard 삭제된 파트가 낀 pair 는 비활성화(연속 파트는 항상 1.0)
            pair_alive = (z_gate[self.flange_sec, self.flange_part_a]
                          * z_gate[self.flange_sec, self.flange_part_b])
            pair_alive = (pair_alive >= BINARIZE_THRESHOLD).float()  # (P,) 0/1, grad 없음

            # i 는 -n 방향, j 는 +n 방향으로 각자 Δt/2 만큼
            move_a = (-0.5 * dt_a * pair_alive).unsqueeze(-1) * self.flange_normal   # (P,2)
            move_b = ( 0.5 * dt_b * pair_alive).unsqueeze(-1) * self.flange_normal

            # §6 테이퍼: 본인 + 이웃 2개까지 가중 분배. index_add_ 는 누적이므로
            # 한 노드가 여러 pair/이웃에 걸쳐도 자동으로 합산된다(실측: 663노드 중 153개가
            # 2개 이상 pair 에 관여).
            for k in range(self.taper_w.numel()):
                w = self.taper_w[k]
                flange_offset.index_add_(0, self.taper_idx_a[:, k], w * move_a)
                flange_offset.index_add_(0, self.taper_idx_b[:, k], w * move_b)

        new_coords = new_coords + flange_offset
        # ────────────────────────────────────────────────────────────────────

        if gate_active:
            ...
        return new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open, flange_offset
```

**반환값 1개 추가**(`flange_offset`)에 주의 — 모든 호출부(`train_step_multi()`,
`report_design_comparison()`, `export_final_section_csv()`, `run_static_dry_run()`,
`compute_rho_adaptive()` 등)의 언패킹을 함께 수정해야 한다. 누락 시 즉시 ValueError로 드러난다.

**금지 사항**: `t_raw`에 `detach()`를 걸지 말 것. 두께→좌표→Mp 그래디언트 경로가 이 설계의 존재
이유다(Solver 라운드에서 OpenAI가 안정화용으로 제안했으나 채택하지 않음 — §10에서 불안정이
실측되면 최후 폴백으로만 검토).

---

## §5 Stage-1 freeze 처리 — **오프셋은 동결 대상이 아니다**

**문제**: `train_step_multi()` L2138-2139는 Stage 1(epoch<100, "좌표 동결·두께만 학습") 동안
`new_coords`를 base 좌표로 **덮어쓴다**. 이 구간에도 두께는 계속 커지므로, 오프셋까지 함께 지워지면
**Stage 1 내내 간섭이 발생하고 Stage 2 진입 시 좌표가 튀는 불연속**이 생긴다.

**결정(양 모델 즉시 수렴)**: 동결 대상은 **GNN이 예측한 변위(`delta_coords`)뿐**이며, 두께의
결정론적 함수인 오프셋은 Stage 1에도 반영한다.

```python
# AI_design_v1_15.py train_step_multi() — 기존 L2138-2139 교체
if epoch < ASWC_STAGE1_END:
    # [v1.15 §5] GNN 변위만 동결하고 두께 연동 오프셋은 보존한다.
    # (기존 v1.14: new_coords = base_coords.detach() + 0.0*(...) — 오프셋까지 지워졌다)
    new_coords = base_coords.detach() + flange_offset
```

`delta_coords`는 언패킹 이후 `train_step_multi()` 안에서 **사용되지 않는다**(v1.14 기준 L2132의
언패킹이 유일한 등장). 따라서 이 방식이 다른 손실에 부작용을 주지 않음이 코드로 확인됐다.

---

## §6 테이퍼링 — **dense matmul 금지, 사전 인덱스 버퍼 사용**

플랜지 노드만 최대 0.206mm 밀고 인접 자유 노드를 그대로 두면 폴리라인에 단차가 생긴다. 이
프로젝트는 v1.7~v1.10 **네 버전에 걸쳐 바로 그 "국소 꺾임/스파이크"와 싸웠으므로**(
`compute_smoothness_lse_v3`, boundary continuity, Top-K smoothness) 재발은 용납할 수 없다.

**구현**: §4 코드의 `taper_idx_*` / `taper_w` 버퍼 + `index_add_` 누적(양 Critic 공통 판정).

**기각된 대안**: Solver 라운드에서 Gemini가 제안한
`torch.matmul(taper_matrix, offset_vector)`(taper_matrix가 노드수×노드수) 방식은
**1581×1581 ≈ 2.5M 원소 행렬을 매 forward 곱하는 비효율**이며, `matmul`은 누적이 아니라 **대체**라
플랜지 노드 자신의 오프셋을 보존하려면 항등 성분을 따로 넣어야 하는 취약한 형태다. 채택하지 않는다.

---

## §7 collision loss에서 fixed-fixed 접합 쌍 제외

**대상**: `uni_section_v23.py`의 `build_collision_spec()`(마스크 생성) +
`compute_collision_loss_v5()`(마스크 적용).

### §7.1 조건은 AND — OR은 결함

`fix_pair_mask`는 **양쪽 노드가 모두 고정(AND)** 일 때만 True다.
Solver 라운드에서 OpenAI가 쓴 `(i 고정 OR j 고정)`은 **결함**이며 양 Critic이 공통 확인했다 —
한쪽만 고정이면 나머지 자유 노드를 움직여 간섭을 **실제로 해결할 수 있으므로** 마스킹하면 안 된다.
전체 1581개 중 697개가 고정이라 OR을 쓰면 유효한 충돌 회피의 상당 부분이 조용히 무력화된다.

### §7.2 인덱스 공간 — 전역 노드 id를 로컬 마스크에 쓰면 안 된다(결함)

`_signed_projections(c_seg, c_pt)`는 `new_coords[sec_mask & (part_ids == a)]` 같은
**(섹션, 파트) 부분집합**으로 호출되므로 반환 행렬 `(n_pts, n_segs)`의 인덱스는 **로컬(0..n-1)** 이다.
전역 노드 인덱스(0..1580)를 이 로컬 길이 마스크에 그대로 쓰면 IndexError이거나 엉뚱한 곳을
마스킹한다(양 Critic 공통 확인).

**해결**: `build_collision_spec()`이 이미 `(sec, a, b)` 그룹을 순회하므로, **그 시점에 로컬 인덱스로
변환한 불리언 마스크를 direction dict에 함께 저장**한다.

```python
# uni_section_v23.build_collision_spec() — directions.append 시 함께 저장
# fix_mask_global: (N,) bool  (호출부에서 x[:,2].bool() 전달)
# mask_pt[p] = True  <=> pt_part 의 p 번째 노드가 고정
# mask_seg[s] = True <=> seg_part 의 s 번째 세그먼트 양 끝이 모두 고정
directions.append({
    'seg_part': roles[0], 'pt_part': roles[1],
    'sigma': sigma, 'clearance': clearance,
    'ff_mask': ff_mask,   # (n_pts, n_segs) bool — AND 조건으로 미리 조립
})
```

### §7.3 적용은 `valid`에서 제외 — `proj` 오염 금지

```python
# compute_collision_loss_v5() 내부
proj, valid = _signed_projections(c_seg, c_pt)
if d.get('ff_mask') is not None:
    valid = valid & (~d['ff_mask'])      # [v1.15 §7] 접합 쌍 항목을 아예 무효화
if valid.sum() == 0:
    continue
```

**기각된 대안**: `proj.masked_fill(mask, CLEARANCE+1.0)`처럼 큰 값을 심는 방식은 해당 항목을 여전히
`valid`로 남겨 `relu`/`topk` 경로에 들어가게 하고, 그래디언트 스파이크를 유발할 수 있다(양 Critic
공통 지적). `valid`에서 빼는 편이 안전하고 연산도 준다.

---

## §8 clearance 커리큘럼에서 접합 쌍 제외 (이중 규제 방지)

v1.14 §2의 `apply_clearance_curriculum()`은 모든 direction의 clearance를
`CLEARANCE_TARGET_MM(0.5)`까지 조인다. 그런데 오프셋 레이어는 같은 접합 쌍의
`dist - (t_a+t_b)/2`를 **`gap0`으로 유지**하려 한다. 두 메커니즘이 같은 양을 서로 다른 목표값으로
끌어당기는 **이중 규제**가 되므로 반드시 분리해야 한다(양 Critic 공통 지적).

**조치**: `ff_mask`가 있는 direction은 커리큘럼 갱신 대상에서 제외한다(원본 `_init_clearance` 유지).

**`CLEARANCE_TARGET_MM` 값 — Judge 판정: 0.5 유지**
- Gemini Critic: 0.2로 하향 / OpenAI Critic: 0.5 유지 + 접합 쌍만 면제.
- **판정**: 접합 쌍을 면제하면 v1.14에서 관측된 과도한 두께 하방 압력의 원인 자체가 제거되므로,
  일반 쌍에 적용되는 전역 목표까지 낮출 근거가 없다. 국소 문제를 전역 완화로 해결하는 것은
  이 프로젝트가 review_v1_13.md에서 이미 경계한 패턴이다. **0.5 유지 후 §10에서 실측 재보정.**
- 참고: OpenAI Critic이 근거로 든 "0.5mm는 제조 GD&T에서 유래"라는 서술은 **이 저장소에 그런 문서가
  없다**(`clearance_default=0.5`는 코드 기본값일 뿐). 결론은 채택하되 그 근거는 채택하지 않는다.

---

## §9 이중 안전장치

collision loss에서 접합 쌍을 빼면, 오프셋 레이어에 버그가 있어도 **아무도 눈치채지 못한다**.

1. **진단 전용 비마스킹 관통 지표**: 매 epoch, 마스킹 없이 접합 쌍의
   `penetration = relu((t_a+t_b)/2 + gap0 - dist)`를 `detach()` 상태로 계산해
   `l_flange_pen_max` / `l_flange_pen_mean`으로 로깅한다. **역전파에는 절대 넣지 않는다.**
2. **export 후 하드 assert**: `export_final_section_csv()` 직후, 저장된 CSV를 다시 읽어
   노드 ±t/2 밴드 겹침을 전수 검사하고 `FLANGE_ASSERT_TOL_MM(0.01)` 초과 시 `AssertionError`로
   중단한다(review_v1_13.md에서 쓴 검사와 동일 방식).

---

## §10 검증 계획

1. **단위 검증(학습 없이)**
   - 접합 쌍 탐지 결과가 실측과 일치하는가: `gap0<=0.5mm` 기준 **147쌍** 근처인가,
     Outer-Plate/Plate-Inner의 정확 접촉 쌍(각 6개, floor 16)이 포함되는가.
   - 두께를 인위적으로 `Δt=+0.2mm` 넣었을 때 접합 쌍 거리가 정확히 0.2mm 늘어나는가(오차 1e-5mm).
   - 이웃 노드가 `[1, 2/3, 1/3]` 비율로 따라 움직이는가(테이퍼).
   - `index_add_` 경로에 autograd 오류가 없고 `t_raw`로 그래디언트가 흐르는가.
2. **학습 로그**
   - `l_flange_pen_max`가 **Stage 1 시작부터** ~0 을 유지하는가(§5가 제대로 동작하는 증거).
   - `l_smooth_lse`가 v1.14 대비 악화되지 않는가(§6 테이퍼 효과 — 악화 시 `FLANGE_TAPER_N` 상향).
   - `l_col`/`g_col`이 접합 쌍 제외 후에도 비접합 쌍의 위반을 정상 포착하는가.
3. **최종 결과**
   - `final_section_v1_15.csv` 겹침 = **0** (§9-2 assert 통과).
   - Mp 달성률이 78.8%에서 개선되는가(두께 하방 압력 해소 효과 — 개선 없으면 원인 재분석).
   - 두께가 다시 전 파트 T_MAX로 saturate되지 않는가.
4. **재보정 대상**: `FLANGE_GAP0_MAX_MM`, `FLANGE_TAPER_N`, `CLEARANCE_TARGET_MM`은 모두 시작값이며
   위 실측 후 조정한다.

---

## §11 Out of Scope

- `fix_x`/`fix_y` 마스크 자체의 완화(자유 이동) — `idea_v1_14.md`에서 기각, 여전히 유효.
- `ENABLE_DYNAMIC_ALDA` 활성화 — v1.14 §3 플래그, 본 작업과 독립.
- GNN 아키텍처·손실 가중치·커리큘럼 스케줄 재설계.
- 법선 벡터 동적 갱신 — 1차는 정적 법선으로 시작(`idea_v1_15.md` ⑦). §10에서 각도 편차 측정 후 판단.
- 다중 pair 노드(153개)에 대한 1D 체인 최소제곱 분배 — 1차는 `index_add_` 합산으로 시작하고,
  §9-1 진단 지표의 잔여 위반이 유의미할 때만 후속 버전에서 격상(`idea_v1_15.md` ①).

---

<details>
<summary>숙의 과정</summary>

### 모델 기여
- **Gemini (Architect, Solver conf98 → Critic conf98):** Stage-1에서 GNN 변위만 동결하고 오프셋은
  보존해야 한다는 핵심 결정을 정식화. Critic 라운드에서 자신의 dense matmul 테이퍼 안을 스스로
  기각하고 `index_add_` 방식을 채택.
- **OpenAI (Explorer, Solver conf92 → Critic conf88):** `taper_idx`/`taper_w` 버퍼 기반의 효율적
  테이퍼 구현을 제시(채택). 다만 Solver 라운드에서 필수 XML 블록을 누락해 format enforcement를
  1회 적용했고, `fix_pair_mask`를 OR로 쓴 것과 전역/로컬 인덱스 혼용은 Critic 라운드에서 자신이
  결함으로 인정했다.
- **Claude (Judge/Validator):** 삽입 지점(L847/L859)·Stage-1 덮어쓰기(L2138-2139)·`delta_coords`
  미사용 사실을 코드로 확인해 Solver 프롬프트에 선반영했고, 아래 2건을 실측으로 확정.

### Judge 판정 사항
1. **`FLANGE_GAP0_MAX_MM` 근거 확보** — 두 모델 모두 접합 쌍 판정 임계를 구체화하지 못했다.
   Claude가 초기 형상의 fixed-fixed 408쌍 `gap0` 분포를 실측(≤0.5mm 147쌍 / 0.5~2.0 160쌍 /
   >2.0 101쌍, Outer-Plate는 최대 33.17mm)해 **0.5mm를 데이터 기반 초기값**으로 확정했다.
   임계가 없으면 33mm 떨어진 노드까지 접합 쌍으로 묶이는 심각한 오작동이 발생한다.
2. **`CLEARANCE_TARGET_MM`은 0.5 유지** — 두 Critic이 대립했으나, 접합 쌍 면제(§8)로 근본 압력이
   제거되므로 전역 목표까지 낮출 근거가 없다고 판정. 단 OpenAI가 든 "GD&T 유래" 근거는 저장소에
   존재하지 않아 기각하고, 결론만 채택했다.
3. **`delta_coords` 미사용 확인** — §5 방식이 다른 손실에 부작용이 없음을 코드로 검증(v1.14 기준
   L2132 언패킹이 유일한 등장).

### 신뢰 점수
- Gemini: Solver 98, Critic 98 (High trust)
- OpenAI: Solver 92, Critic 88 (format enforcement 1회, 근거 미검증 서술로 Credibility 일부 감점)
- Claude(조율자): 코드 사실 3건·데이터 분포 1건 실측으로 쟁점 확정

</details>
