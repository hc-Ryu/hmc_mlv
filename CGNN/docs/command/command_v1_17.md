# command_v1_17.md — 플랜지 노드 완전 고정 + Stage4 직전 1회성 강체 이동 작업지시서

작성일: 2026-08-21 (`/synod design` 세션, fairy_tail 프로젝트, Gemini flash/thinking-high
(pro는 rate limit 3회 재시도 실패 후 flash로 폴백, Solver conf95/can_exit=true) + OpenAI
o3/reasoning-high (Solver conf66/can_exit=false — Explorer 페르소나로 6개 실패 시나리오 제시).
Defense 라운드 대신 Claude가 실제 코드(`AI_design_v1_15_6.py`, `uni_section_v25.py`)를 직접
재확인해 두 모델의 주장을 교차검증하는 Judge 라운드로 대체 — Gemini 코드 초안 2건과 OpenAI
지적 1건이 실제 코드와 불일치함을 확인해 정정했다(§5, §7).

근거: `docs/idea/idea_v1_17.md`(신뢰도 85%), `docs/command/command_v1_16.md`(선행 경쟁안,
폐기됨, §4 우선순위 정책·Inner-Patch2 예외 처리 계승), `AI_design_v1_15_6.py`(대상 코드베이스)
대상: **신규 파일** `AI_design_v1_17.py` (프로젝트 관례: 버전마다 새 파일 생성, 기존 파일
수정 금지)

---

## §0 배경 — 왜 이 작업이 필요한가

`idea_v1_17.md`가 승인한 아키텍처는 v1.16(파트 단위 Y-shift를 `forward()`에 미분 가능하게
통합, `command_v1_16.md`)보다 더 근본적인 대안이다: 학습 중에는 플랜지 노드를 아예 움직이지
않고(오프셋/테이퍼 메커니즘 완전 제거), 학습이 끝난 뒤 Stage4 진입 직전 딱 한 번 numpy로
강체 이동을 계산해 적용한다. Stage4가 "존재 확정 후 형상·두께 재수렴"을 위해 이미 존재하는
독립 refinement 구간이므로, 강체 이동으로 바뀐 형상을 자연스럽게 흡수·재수렴시킬 수 있다는
것이 핵심 근거다(v1.16이 실패한 export 시점 후처리와 달리 train/export 불일치가 없다).

**Judge 코드 재확인 결과 반영 사항**: 두 모델 모두 실제 좌표/옵티마이저 구현을 정확히
알지 못한 채 일부 세부 사항을 추정했다. 본 작업지시서는 다음 3가지를 실제 코드로 정정한다:
1. 좌표는 `model.node_coords[P,N,2]` 같은 구조화 텐서가 아니라 **평평한 `(N,2)` 텐서**이며
   `part_ids_local`(x[:,4])·`section_ids_local`(x[:,5])로 그룹핑된다(Gemini 초안 코드는
   존재하지 않는 API를 가정했다 — §3에서 정정).
2. Stage4 진입 시 옵티마이저는 이미 **모멘텀을 명시적으로 이전**한다(L3233-3251,
   `old_state_by_param` → 새 `AdamW`에 재주입, 실패 시에만 모멘텀 없이 폴백) — "LR만 감쇠하고
   모멘텀은 그대로 유지된다"는 OpenAI의 주장은 인용한 라인 번호(L3190-3198)가 실제 코드와
   다르다. 다만 "모멘텀이 보존된다"는 결론 자체(전이 방식이든 유지든)는 맞고, 이것이 강체
   이동 직후 첫 backward에서 문제가 될 수 있다는 우려의 본질은 유효하다(§7에서 반영).
3. 이동량은 **파트당 스칼라 1개가 아니라 (floor, part)당 스칼라**다 — `FLANGE_PAIR_SPEC`이
   17개 floor 각각에 독립 적용되므로(idea_v1_17.md §6 3단계 "floor별로"), 전역 파트 단위
   평균으로 뭉개면 OpenAI가 지적한 "밴딩/bowing" 문제(§8)가 그대로 발생한다. Gemini 초안
   코드는 이 floor 차원을 누락했다 — §4에서 정정한 (17, num_parts) 형태로 교체한다.

---

## §1 파일 계획

- **신규 파일**: `AI_design_v1_17.py` = `AI_design_v1_15_6.py`의 사본.
- **삭제 대상**: L1310-1349 전체(`flange_offset`, `move_a`/`move_b`, 테이퍼 `index_add` 루프).
  `identify_mating_flange_pairs()`(L759-868) 자체는 **유지**한다 — 오프셋 계산에는 더 이상
  쓰이지 않지만, 반환하는 `flange_idx_a/b`, `flange_normal`, `flange_gap0` 버퍼가 Stage4
  직전 강체 이동 계산과 §6 진단 로그에 그대로 필요하다.
- **의존성**: `uni_section_v25.py`의 `build_collision_spec()`/`compute_collision_loss_v5()`
  (변경 없음, §5).

---

## §2 forward() 정리 — L1310-1349 제거

`AI_design_v1_15_6.py`의 `flange_offset` 계산 블록(§4에서 확인된 정확한 위치)을 완전히
삭제한다. `identify_mating_flange_pairs()` 호출과 `_flange_ready` 플래그, `register_buffer`
등록은 그대로 둔다(§1 참고). `fix_y_mask`에 이미 등록된 플랜지 노드는 y 고정 대상이므로,
이 블록 제거만으로 "학습 중 100% 고정"이 자동으로 달성된다(v1.15 이전 상태로 복귀) — 별도의
새 마스킹 로직은 필요 없다.

---

## §3 신규 상수 및 버퍼

```python
# ══════════════════════════════════════════════════════════════════
# [v1.17] Stage4 직전 1회성 강체 Y-shift
# 근거: docs/idea/idea_v1_17.md, docs/command/command_v1_17.md
# ══════════════════════════════════════════════════════════════════
RIGID_SHIFT_ANCHOR_ORDER   = ['Outer', 'Plate']   # Outer 우선, 죽으면 Plate로 폴백(§4.3)
RIGID_SHIFT_CLIP_FRAC_MM   = 6.0   # |Δy_part| 절대 상한(mm) — OpenAI §6 "12mm 겹침" 시나리오
                                    # 방어. idea_v1_17.md는 구체적 상한을 명시하지 않았으므로
                                    # v1.16의 상대적 clip(RIGID_SHIFT_CLIP_FRAC=0.10*bbox) 대신
                                    # 절대값 상한을 신규 채택 — 근거는 §8.
STAGE4_RIGID_SHIFT_ABORT_FRAC = 0.5   # max(|Δy|) > 0.5 * RIGID_SHIFT_CLIP_FRAC_MM 초과 시
                                       # 강체 이동을 적용하지 않고 경고 후 중단(§9 fail-fast)
```

**좌표/데이터 레이아웃(실제 코드 기준, Gemini 초안 정정)**: 강체 이동 계산 시점(Stage4 진입
직전)의 좌표는 `final_coords`(L3172, `info['new_coords'].detach().clone()`, shape `(N,2)`
평평한 텐서), 두께는 `t_final`(동일 shape), 파트/섹션 인덱스는 `part_ids_local`/
`section_ids_local`(x[:,4]/x[:,5], 각 shape `(N,)`)이다. `[P, N, 2]`처럼 파트별로 구조화된
텐서는 존재하지 않는다 — 모든 계산은 이 평평한 텐서와 `torch.nonzero`/boolean 마스킹으로
수행한다(§4).

---

## §4 Stage4 직전 강체 이동 — 정확한 삽입 위치와 알고리즘

삽입 위치: `AI_design_v1_15_6.py` L3184-3202의 Stage4 진입 판정(`enter_stage4`,
`stage4_start_epoch = last_epoch + 1`) **직후**, 옵티마이저 재빌드(L3241) **이전**.
이 시점에 이미 `final_coords`, `final_z_gate`(L3172-3174)가 계산되어 있다.

### §4.1 (floor, part)당 Δy — 우선순위 계층 전파

`FLANGE_PAIR_SPEC`(L742-756, part-id 매핑 `0=Outer,1=Plate,2=Inner,3=Patch1,4=Patch2`)의
각 쌍은 이미 특정 floor에 귀속되어 있다(`identify_mating_flange_pairs()`가 반환하는
`flange_sec`, `flange_part_a`, `flange_part_b`). 강체 이동은 **floor마다 독립적으로**
계산한다(§0의 정정 사항 3) — `dy_part[sec, part]` shape `(17, 5)`.

```python
import numpy as np

def compute_rigid_shift(final_coords, t_final, flange_pairs, pruning_state,
                         z_gate_final, num_sections=NUM_SECTIONS):
    """[v1.17 §4] Stage4 진입 직전 1회 실행(no_grad, numpy). floor x part 단위 Δy를 계산한다.
    반환: dy_part (17, 5) numpy array."""
    idx_a, idx_b = flange_pairs['flange_idx_a'], flange_pairs['flange_idx_b']
    normal = flange_pairs['flange_normal']              # (P, 2), 단위 법선 i->j
    sec_of = flange_pairs['flange_sec']                  # (P,)
    part_a, part_b = flange_pairs['flange_part_a'], flange_pairs['flange_part_b']

    y = final_coords[:, 1].cpu().numpy()
    t = t_final.squeeze(-1).cpu().numpy()
    n_y = normal[:, 1].cpu().numpy()                     # 법선의 y성분(§4.2 방향 처리)

    dy_part = np.zeros((num_sections, 5))

    # §4.3: 앵커 폴백 — floor별로 Outer/Plate 생존 여부 확인
    alive_state = (pruning_state['state'] != STATE_DELETED).numpy()   # (17, n_cand)
    zc = z_gate_final.cpu().numpy()                                    # (17, 5)

    def part_alive(sec, part_id):
        if part_id in CONTINUOUS_PARTS:      # Outer 등 연속 파트는 항상 생존
            return True
        cand_col = CANDIDATE_PARTS.index(part_id) if part_id in CANDIDATE_PARTS else None
        if cand_col is None:
            return True
        return bool(alive_state[sec, cand_col]) and (zc[sec, part_id] >= BINARIZE_THRESHOLD)

    for sec in range(num_sections):
        outer_ok = part_alive(sec, 0)
        plate_ok = part_alive(sec, 1)
        # §4.3 [OpenAI 지적 #1 반영] Outer가 해당 floor에서 죽어 있으면 Plate를 앵커로 승격.
        # 둘 다 죽었으면 이 floor는 이동시키지 않는다(Δy=0) — 죽은 파트끼리 좌표를 맞출
        # 물리적 근거가 없고, 어차피 export되지 않는다.
        if outer_ok:
            anchor = 0
        elif plate_ok:
            anchor = 1
            dy_part[sec, 1] = 0.0   # Plate 자신이 앵커면 Δy_Plate=0으로 재정의
        else:
            continue   # 둘 다 삭제된 floor — 이동 없음

        pair_mask = (sec_of == sec)

        def mean_shift(parent_part, child_part):
            m = pair_mask & (
                ((part_a == parent_part) & (part_b == child_part)) |
                ((part_a == child_part) & (part_b == parent_part))
            )
            if not m.any():
                return 0.0
            ia, ib = idx_a[m], idx_b[m]
            # a가 parent인지 child인지에 따라 부호를 맞춘다
            a_is_parent = (part_a[m] == parent_part)
            y_parent = np.where(a_is_parent.numpy(), y[ia], y[ib])
            y_child  = np.where(a_is_parent.numpy(), y[ib], y[ia])
            t_parent = np.where(a_is_parent.numpy(), t[ia], t[ib])
            t_child  = np.where(a_is_parent.numpy(), t[ib], t[ia])
            nrm = np.where(a_is_parent.numpy(), n_y[m.numpy()], -n_y[m.numpy()])
            y_target = y_parent + nrm * (t_parent / 2.0 + t_child / 2.0)
            return float(np.mean(y_target - y_child))

        if anchor == 0:
            dy_part[sec, 1] = mean_shift(0, 1)                       # Plate <- Outer
        # §4.4 Inner/Patch1: Plate(이미 이동한 값 기준)로부터
        dy_part[sec, 2] = dy_part[sec, 1] + mean_shift(1, 2)         # Inner  <- Plate
        dy_part[sec, 3] = dy_part[sec, 1] + mean_shift(1, 3)         # Patch1 <- Plate
        # §4.5 Inner-Patch2 샌드위치 예외(v1.16 command §3.3 계승) — Inner 경유만 사용,
        # Plate<->Patch2 직접 관계는 재조정하지 않고 잔차를 collision loss(§5)로 흡수한다.
        dy_part[sec, 4] = dy_part[sec, 2] + mean_shift(2, 4)         # Patch2 <- Inner

    # §9 안전장치: 절대값 클램프
    dy_part = np.clip(dy_part, -RIGID_SHIFT_CLIP_FRAC_MM, RIGID_SHIFT_CLIP_FRAC_MM)
    return dy_part
```

### §4.2 방향(Y-only)

idea 문서 §3의 결정을 그대로 따른다 — 전역 Y축 이동만 수행한다(웹 구조의 X-민감성 때문에
2D 이동을 배제). `flange_normal`의 y성분(`n_y`)을 부호 판단에만 사용하고 실제 이동은 항상
Y축으로만 가한다. v1.16과 달리 `proj_y(flange_normal)`을 새로 구현할 필요는 없다 — 대다수
플랜지가 이미 거의 수직이며, 이 안은 애초에 1회성 근사이므로 v1.16 수준의 정밀한 각도 보정은
과설계다(idea 문서에 없는 요구사항을 추가하지 않는다).

### §4.3 앵커 폴백

`command_v1_16.md` §3.4의 정책(Outer 죽으면 Plate로, 둘 다 죽으면 이동 안 함)을 그대로
채승한다. **[OpenAI 지적 #1 반영]**: `z_gate`가 게이팅으로 Outer 두께를 0에 가깝게 만든
상태에서도 "노드가 존재"하는 것과 "파트가 사실상 삭제됨"은 구분해야 한다 — `part_alive()`는
`pruning_state`의 hard 삭제 상태와 `z_gate >= BINARIZE_THRESHOLD`를 AND로 결합해(L3217의
`hard_exists` 계산과 동일한 패턴) 이 구분을 명시적으로 처리한다.

### §4.4 Plate 4중 이웃 충돌

idea 문서 §4의 우선순위(Outer 앵커 → Plate → 자식들)를 그대로 구현한다. **[OpenAI 지적 #2
반영]**: Inner와 Patch2가 서로 다른 부호의 요구를 낼 수 있는 과결정 상황은 v1.16과
동일하게 재조정을 시도하지 않고 Inner 경유 경로로만 결정하며(§4.5), 남는 잔차는 §5의
collision loss가 흡수한다. 이는 idea 문서 §5.3이 이미 "정책적 완화, 완전 해결 아님"으로
명시한 잔여 리스크이며, 본 작업지시서가 새로 해결하려 시도하지 않는다(v1.16의 동일 결정과
일관성 유지).

---

## §5 강체 이동을 forward()에 반영하는 방법

`compute_rigid_shift()`는 **numpy, no_grad, 1회 실행**이다. 계산된 `dy_part (17,5)`를
학습 재개 후(Stage4 루프) 매 forward에서 좌표에 반영하려면, 텐서로 변환해 버퍼로 등록하고
`forward()`에 소규모 블록을 추가한다:

```python
# Stage4 진입 직전, compute_rigid_shift() 실행 직후
model.register_buffer('stage4_rigid_dy',
                       torch.from_numpy(dy_part).float().to(device), persistent=False)
model._stage4_rigid_active = True
```

`forward()` L1349(구 `flange_offset` 블록 삭제 위치) 자리에 다음을 추가한다:

```python
if getattr(self, '_stage4_rigid_active', False):
    # [v1.17 §5] Stage4 직전 1회 계산된 (floor,part) 강체 Y-shift를 매 forward 동일하게 적용.
    # fix_y_mask와 무관하게 적용한다 — 고정 노드도 파트 전체와 함께 강체 이동해야 하므로
    # (v1.16 command §4의 dy_node gather와 동일 원칙).
    dy_node = self.stage4_rigid_dy[section_ids_local, part_ids_local]   # (N,)
    new_coords = new_coords + torch.stack(
        [torch.zeros_like(dy_node), dy_node], dim=1)
```

이 블록은 `_stage4_rigid_active`가 켜진 뒤(Stage4 진입 이후) 항상 동일한 상수 오프셋을
더하므로, 그래디언트는 이 항을 그냥 통과한다(정지 상수 + 학습 가능 좌표). 메인 학습 루프
(Stage4 이전)에는 이 플래그가 꺼져 있으므로 어떤 영향도 없다.

---

## §6 Collision Loss 및 진단 신호

### §6.1 Collision Loss — 기존 `ff_mask` 그대로 유효

`uni_section_v25.py`의 `build_collision_spec()`/`compute_collision_loss_v5()`(L670-801)는
변경 불필요. `ff_mask`(양쪽 모두 고정인 쌍을 collision loss에서 제외)는 이미 
`FLANGE_PAIR_SPEC` 쌍을 정확히 커버하므로, 오프셋 제거 후에도 두께 성장이 collision loss로
억제되지 않는다(idea 문서 §5.2, 코드로 재확인 완료). Plate 전이 구간 노드 {4,5,6,25,26,27}은
`FLANGE_PAIR_SPEC`에 없으므로 `ff_mask` 대상이 아니다 — 이는 v1.15의 원래 의도된 동작이며
(전이 노드가 자유롭게 움직여 경계 불연속을 흡수해야 하므로), 본 작업으로 새로 생기는 문제가
아니다.

### §6.2 진단 신호 — `compute_flange_penetration_diag()`는 이 안에서 무의미해진다

**[OpenAI 지적 #4, 반드시 반영]**: `compute_flange_penetration_diag()`(L1071-1093)는
현재 "고정된 두 노드 사이의 부호 있는 거리"를 측정한다. 오프셋 메커니즘이 있던 v1.15에서는
이 값이 오프셋이 gap0를 얼마나 잘 보존하는지 보여주는 유의미한 신호였다. v1.17에서는 플랜지
노드가 **학습 내내 전혀 움직이지 않으므로** 이 값은 상수(초기 gap0)에 고정되고, 학습 로그의
`flange_pen`은 항상 ~0을 보고해 거짓 안전 신호가 된다.

신규 진단 함수 `compute_flange_gap_margin_diag()`를 추가한다 — 노드 이동이 아니라
**두께 성장이 초기 표면 간극을 얼마나 잠식했는지**를 직접 계산한다(no_grad, 학습에 영향 없음):

```python
def compute_flange_gap_margin_diag(model, t_final, flange_pairs):
    """[v1.17 §6.2] 노드가 안 움직이므로 flange_pen 대신 두께 기반 여유값을 진단한다.
    margin = gap0 - (t_a/2 + t_b/2 - t_a0/2 - t_b0/2). margin < 0 이면 두께가 커져 표면이
    이미 겹치고 있다는 뜻 — Stage4 직전 강체 이동이 처리해야 할 양의 프록시다."""
    if not getattr(model, '_flange_ready', False):
        return {'l_gap_margin_min': 0.0, 'n_gap_violated': 0}
    ia, ib = model.flange_idx_a, model.flange_idx_b
    gap0 = model.flange_gap0
    t = t_final.squeeze(-1)
    t0 = model._flange_t0   # §6.3에서 신규 등록
    dt = (t[ia] - t0[ia]) + (t[ib] - t0[ib])
    margin = gap0 - dt / 2.0
    return {
        'l_gap_margin_min': float(margin.min()),
        'n_gap_violated': int((margin < 0).sum()),
    }
```

### §6.3 신규 버퍼: 초기 두께 스냅샷

`compute_flange_gap_margin_diag()`는 학습 시작 시점의 `t_final`을 기준선으로 필요로 한다.
`identify_mating_flange_pairs()` 호출 직후(학습 루프 진입 전) 1회 `self._flange_t0 =
t_final.detach().clone()`로 등록한다(`register_buffer`, `persistent=False`).

### §6.4 로그 통합

L3162 부근의 epoch 로그(`flange_pen=...`)에 `gap_margin_min`, `n_gap_violated`를 추가
출력한다. Stage4 진입 직전에는 §4의 `dy_part` 분포(최대/평균 절대값)도 함께 로그로 남긴다
(idea 문서 §6-5의 "강체 이동량 분포" 검증 요구사항).

---

## §7 Stage4 옵티마이저 처리

**[Judge 정정]** 실제 코드(L3233-3251)는 Stage4 진입 시 옵티마이저를 재빌드하되 기존
파라미터의 모멘텀 상태(`optimizer.state[p]`)를 명시적으로 이전한다(`old_state_by_param`,
실패 시에만 모멘텀 없이 폴백). 이는 v1.9 §2가 "게이트가 부드럽게 보간되는" 상황을 전제로
설계한 것이다 — 좌표 자체가 불연속으로 점프하는 상황(§5의 강체 이동)은 이 전제와 다르다.

**정책**: v1.9 §2의 기본 동작(모멘텀 이전)은 Stage4 진입 자체에 대해서는 그대로 둔다(다른
안정성 검증이 이미 이 정책에 의존하고 있으므로 범위 밖에서 변경하지 않는다). 다만 §5의
강체 이동을 적용한 직후에 한해, 이동된 파트(Plate/Inner/Patch1/Patch2)에 속한 노드의
좌표를 소스로 쓰는 파라미터 그룹(`main_params`)에 대해서만 옵티마이저 상태를 리셋한다:

```python
# §5 강체 이동 적용 직후, 옵티마이저 재빌드(L3241) 이전에 삽입
if np.abs(dy_part).max() > 1e-6:
    print(f"[v1.17] 강체 이동 적용됨(max|Δy|={np.abs(dy_part).max():.3f}mm) — "
          f"main_params 모멘텀을 이전하지 않고 새로 시작합니다.")
    old_state_by_param = {p: s for p, s in old_state_by_param.items() if p not in main_params}
```

`thick_params`(두께 디코더)는 좌표 점프와 직접 관련이 없으므로 모멘텀을 그대로 이전한다.
추가로 idea 문서에 없던 항목이지만 OpenAI가 지적한 위험(모멘텀 잔존 시 첫 backward 스파이크)
을 저비용으로 방어하기 위해, Stage4 시작 후 `STAGE4_GATE_WARMUP_EPOCHS`와 별개로 5 epoch
동안 `main_params`의 LR만 선형 웜업(0.1x → 1.0x)한다.

---

## §8 Floor별 이동량 — 균일 가정에 대한 명시

idea 문서 §6-3은 "floor별로" 이동량을 계산하라고 명시하므로, §4의 `dy_part (17,5)`는 이미
floor마다 독립이다. **[OpenAI 지적 #5 관련 명확화]**: 이는 "파트 전체에 단일 스칼라"가
아니라 "파트+floor 조합마다 스칼라"이므로, 긴 Plate를 따라 발생하는 bowing/밴딩 편차는
floor 단위로는 흡수된다. 다만 floor **내부**(같은 floor 안의 여러 노드)는 여전히 단일 스칼라
이동이며, 이는 idea 문서의 원래 설계(파트-floor당 하나의 강체 이동)이므로 본 작업지시서의
범위에서 추가로 세분화하지 않는다 — floor 간 편차가 크게 관측되면(§9 테스트 7) 후속 버전
과제로 남긴다.

---

## §9 검증 및 테스트 계획

idea 문서 §6-5의 우선순위(Mp 달성률 → 겹침 0건 → 이동량 분포)를 그대로 따르되, OpenAI가
제기한 실패 시나리오를 명시적 테스트 항목으로 추가한다.

1. **[최우선] Mp 달성률 회귀 테스트**: v1.15.5/v1.15.6 대비 악화되지 않는지 확인(두께 억제
   재발 여부의 직접 지표, idea 문서 §5의 핵심 리스크).
2. **겹침 0건 확인**: `check_part_overlap()`을 Stage4 종료 후 export 좌표에 대해 실행.
3. **강체 이동량 분포**: §6.4 로그의 `dy_part` 분포가 `RIGID_SHIFT_CLIP_FRAC_MM` 근처에
   몰리지 않는지(클램프가 상시 개입한다면 클램프 값 자체가 근본 문제를 가리는 것) 확인.
4. **[OpenAI #1] Outer 프루닝 시나리오**: 특정 floor에서 `z_gate[:,0]`을 인위적으로 0으로
   만든 상태로 Stage4 직전 스냅샷을 구성 → §4.3 앵커 폴백이 NaN/Inf 없이 Plate로 전환되는지,
   해당 floor의 `dy_part`가 나머지 정상 floor 대비 이상치로 튀지 않는지 확인.
5. **[OpenAI #6] Fail-fast 가드**: 인위적으로 큰 두께 변화를 주입해
   `max(|dy_part|) > STAGE4_RIGID_SHIFT_ABORT_FRAC * RIGID_SHIFT_CLIP_FRAC_MM`을 유발 →
   강체 이동이 적용되지 않고 경고 로그만 남기는지 확인(§3 상수).
6. **[OpenAI #3] 옵티마이저 리셋 효과**: §7의 `main_params` 모멘텀 리셋을 켠 경우와 끈 경우
   각각 Stage4 첫 10 epoch의 loss 곡선을 비교 — 리셋이 실제로 스파이크를 줄이는지 실측
   확인(리셋이 유의미한 개선을 보이지 않으면 §7 정책은 재검토 대상).
7. **Floor 간 이동량 편차 확인(§8)**: `dy_part`의 floor축 표준편차가 큰 파트가 있는지 확인 —
   크면 후속 버전에서 floor 간 스무딩 필요성의 근거 데이터로 기록.
8. **[OpenAI #4] 진단 신호 검증**: §6.2 `compute_flange_gap_margin_diag()`가 학습 중반부
   두께가 커짐에 따라 `n_gap_violated`가 0에서 증가하는 추세를 실제로 포착하는지 확인(기존
   `flange_pen`이 상수로 죽어있는 것과 대비해 유의미한 신호인지 실측 비교).

---

## §10 리스크 로그

| 항목 | 상태 | 비고 |
|---|---|---|
| 두께 억제 재발(collision loss가 오프셋 제거 후 두께 성장을 억제) | **낮음(코드 확인)** | §6.1 — `ff_mask`가 이미 FLANGE_PAIR_SPEC 쌍을 collision loss에서 제외 |
| Outer 프루닝 시 앵커 실패 | **정책으로 해결** | §4.3 — Plate 폴백, 둘 다 없으면 해당 floor 이동 생략 |
| Plate 4중 이웃 과결정 | **정책적 완화, 완전 해결 아님** | §4.4/§4.5 — v1.16과 동일하게 Inner 경유 + collision loss 잔차 흡수, idea 문서 원래 판단과 일관 |
| `compute_flange_penetration_diag()`가 오프셋 제거 후 상시 0을 보고(거짓 안전 신호) | **신규 진단으로 해결** | §6.2-6.4 — `compute_flange_gap_margin_diag()` 신규 추가 |
| 강체 이동 직후 옵티마이저 모멘텀이 좌표 점프와 충돌 | **부분 완화** | §7 — main_params만 모멘텀 리셋 + 5epoch LR 웜업, 실측 검증 필요(§9-6) |
| 이동량이 비정상적으로 커질 때 폭주 | **fail-fast 가드로 완화** | §3/§9-5 — 절대 클램프 + abort 조건 |
| Floor 간 bowing/밴딩 편차 | **floor 단위 이동으로 부분 완화** | §8 — 파트당 단일 스칼라가 아니라 (floor,part) 단위이므로 v1.16 대비 이미 개선, 완전 해결은 아님 |
| Gemini 초안의 `[P,N,2]` 텐서 가정, 옵티마이저 "완전 리셋" 제안 | **정정 완료** | §0/§5/§7 — 실제 코드 레이아웃과 기존 모멘텀-이전 정책에 맞게 재작성 |

---

## §11 신뢰도 및 숙의 과정

<details>
<summary>숙의 과정</summary>

- **Gemini (Architect, flash 폴백 — pro는 rate limit 3회 재시도 실패, conf95, can_exit=true)**:
  우선순위 계층 전파 수식과 강체 이동 삽입 위치를 구체적으로 제시했으나, 코드베이스의 실제
  좌표 텐서 레이아웃(`[P,N,2]`로 가정, 실제로는 평평한 `(N,2)`)과 Stage4 옵티마이저 정책
  (완전 리셋 제안, 실제로는 모멘텀 이전 정책이 이미 존재)을 정확히 파악하지 못했다 — Judge가
  §0/§5/§7에서 정정.
- **OpenAI (Explorer, o3, conf66, can_exit=false)**: 6개 구체적 실패 시나리오(Outer 프루닝
  시 앵커 배율 오류, Plate 4중 이웃 과결정 재확인, 옵티마이저 모멘텀 충돌, 진단 신호 무의미화,
  floor 간 bowing, 이동량 상한 부재)를 제시했다. 대부분 유효했으나 옵티마이저 관련 인용
  라인 번호(L3190-3198)가 실제 코드와 달랐다(실제는 L3233-3251의 모멘텀 이전 로직) — Judge가
  §7에서 정정하되 우려의 본질(모멘텀 보존이 좌표 점프와 충돌 가능)은 유효하다고 판단해 반영.
- **Claude (Judge)**: `AI_design_v1_15_6.py`(L742-868, L1233-1349, L3184-3251)와
  `uni_section_v25.py`(L670-801)를 직접 재확인해 두 모델의 주장을 교차검증했다. Gemini의
  좌표 텐서 가정과 옵티마이저 리셋 제안, OpenAI의 옵티마이저 코드 인용을 각각 정정하고,
  두 모델 모두 놓친 "floor별 이동량 필요성"(idea 문서 §6-3 명시)을 §4/§8에서 명시적으로
  반영했다.

**신뢰도**: Gemini 95(can_exit=true, 그러나 API 가정 오류로 실측 근거 약화) · OpenAI 66
(can_exit=false, 실패 시나리오는 대부분 유효하나 일부 코드 인용 오류) · Claude 코드 검증으로
두 모델의 결함을 구체적으로 정정.

**최종 신뢰도: 78%** (핵심 메커니즘·우선순위 정책·collision loss 상호작용은 실제 코드로
검증되어 확정 수준이나, §7의 옵티마이저 부분 리셋 정책과 §6.2의 신규 진단 함수는 idea 문서에
없던 신규 설계로 실제 구현·실행 전까지 가설 단계다. idea 문서 자체의 신뢰도(85%)보다 낮은
이유는, 본 작업지시서가 그 위에 새로 추가한 §5-§8의 구체적 구현 결정들이 아직 실측 검증을
거치지 않았기 때문이다.)

</details>
