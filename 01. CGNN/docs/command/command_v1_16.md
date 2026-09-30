# command_v1_16.md — 파트 단위 Y-shift로 테이퍼 완전 대체 작업지시서

작성일: 2026-08-21 (`/synod design` 세션, fairy_tail 프로젝트, Gemini flash/thinking-high
(Solver conf95→Critic conf100, 두 라운드 모두 can_exit=true) + OpenAI o3/reasoning-high
(Solver conf55/can_exit=false → Critic conf70/can_exit=true). Solver 라운드에서 Gemini가
낙관적 초안을 제시했고, OpenAI가 6개 구체적 결함을 지적(can_exit=false)해 Critic 라운드로
진행. Critic 라운드에서 Gemini가 6개 전부를 수용해 구체적 수정안을 제시했고, OpenAI가
이를 must-fix 2개(비플랜지 노드 이동, Outer 삭제 시 fallback)와 nice-to-have 4개로 재분류하며
수렴(두 모델 모두 can_exit=true). Defense 라운드 불필요 — 수렴 완료.)

근거: `docs/idea/idea_v1_16.md`(Idea 1, 신뢰도 88%), `docs/review/review_v1_16.md`(테이퍼
중첩 근본 원인 실측), `AI_design_v1_15_5.py` L1287-1329(대체 대상)
대상: **신규 파일** `AI_design_v1_16.py` (프로젝트 관례: 버전마다 새 파일 생성, 기존 파일
수정 금지)

---

## §0 배경 — 왜 이 작업이 필요한가

`review_v1_16.md`가 실측으로 확인한 버그는 "회전"이 아니라 **테이퍼(taper) 가중치 중첩**이다.
`identify_mating_flange_pairs()`가 만드는 노드별 오프셋에서, 서로 1칸 떨어진 등록 플랜지
노드들이 각자 자기 몫(가중치 1.0)에 더해 이웃의 테이퍼(±1칸=2/3, ±2칸=1/3)까지 겹쳐 받아
가운데 노드는 2.33배, 양끝은 2.0배를 받는 **노드별로 다른 배율**이 원인이다.

이전 시도(v1.16 하이브리드, 폐기됨)는 export 시점 후처리(`align_flange_pairs_rigid()`)로
접근했다가 겹침이 0건→139건으로 악화되어 실패했다. 이유: (a) 학습 중에는 버그가 있는 테이퍼
메커니즘이 그대로 작동해 형상을 왜곡시키고, (b) export 시점에만 사후 보정을 걸어 학습이 본
형상과 최종 export 형상이 달라져 Inner-Patch1 상대 위치 이동이라는 새 부작용을 낳았다.

**해결 방침**: 노드별 오프셋/테이퍼 로직을 완전히 제거하고, **파트 전체가 항상 동일한
Δy_part만큼만 이동하는 강체 Y-shift**로 교체한다. `forward()` 내부(학습 시점)에서 미분
가능한 닫힌 형태로 구현해, 노드 간 배율 차이라는 개념 자체를 구조적으로 없앤다.

---

## §1 파일 계획

- **신규 파일**: `AI_design_v1_16.py` = `AI_design_v1_15_5.py`의 사본.
- **삭제 대상**: 테이퍼 가중치 버퍼(`taper_idx_a`, `taper_idx_b`, `taper_w_a`, `taper_w_b`)와
  L1319-1323의 `index_add` 테이퍼 분배 루프.
- **교체 대상**: L1287-1329 전체(§4의 신규 블록으로 교체).
- **`identify_mating_flange_pairs()` 확장**: 기존 쌍별 버퍼(`flange_idx_a/b`, `flange_normal`)
  계산 시점에 §3의 신규 버퍼(`s_PC`, `flange_normal_y_proj`, part-id 매핑)를 함께 계산해
  `register_buffer`로 등록한다. Critic 라운드 합의: 이 값들은 **학습 중 1회만 계산하고
  불변으로 캐싱**해야 한다(재계산 시 CAD 원점 부호가 뒤집혀도 학습 중간에 달라지지 않도록
  결정론을 보장하기 위함).

---

## §2 신규 상수 블록

```python
# ══════════════════════════════════════════════════════════════════
# [v1.16] 로컬 오버라이드 상수 — 파트 단위 강체 Y-shift
# 근거: docs/idea/idea_v1_16.md, docs/command/command_v1_16.md
# ══════════════════════════════════════════════════════════════════
RIGID_Y_SHIFT_ENABLED   = True
GLOBAL_Y_AXIS           = torch.tensor([0.0, 1.0])
PROJ_Y_DEGENERATE_TOL   = 1e-3   # ||t_raw|| 가 이 값 미만이면 flange_normal이 global Y와
                                  # 평행 → 전역 Y축으로 폴백(0-division 방지)
RESIDUAL_COLLISION_MARGIN_MM = 0.05   # Inner-Patch2 잔차 허용 마진 (§5)
RESIDUAL_COLLISION_LAMBDA    = 1e3    # Inner-Patch2 잔차 손실 가중치 (§5)
RIGID_SHIFT_CLIP_FRAC   = 0.10   # |Δy_part| <= 0.10 * 파트 y-bbox 크기 (안전 클램프, §3.5)
```

**파트 ID 매핑**(기존 `CANDIDATE_PARTS`/`part_ids` 규약 그대로 사용):
`0=Outer, 1=Plate, 2=Inner, 3=Patch1, 4=Patch2`.

---

## §3 Δy_part 계산 — 닫힌 형태, 미분 가능

### §3.1 이동 방향: `proj_y(flange_normal)` (Critic 라운드 확정, Gemini 제안 채택)

Solver 라운드에서 Gemini는 전역 `(0,1)`을, OpenAI는 `flange_normal`의 y-투영을 각각
주장했다. Critic 라운드에서 Gemini가 OpenAI의 지적(≈5% 각도 플랜지 — 킬 스티프너, 30도
챔퍼 — 에서 전역 Y가 형상을 왜곡시킴)을 전면 수용해 다음으로 확정:

각 쌍(pair)의 실제 적용 방향 `t̂_pair`는 원래 `flange_normal`을 전역 Y축에 투영한 벡터다:

```
t_raw  = GLOBAL_Y_AXIS - (GLOBAL_Y_AXIS · flange_normal) * flange_normal
t̂_pair = t_raw / ||t_raw||₂          # 정상 케이스
if ||t_raw||₂ < PROJ_Y_DEGENERATE_TOL:
    t̂_pair = GLOBAL_Y_AXIS            # flange_normal이 global Y와 평행 → 전역 Y로 폴백
```

수직 플랜지(대다수, >95%)에서는 `flange_normal ⟂ Y`이므로 `t̂_pair`는 `GLOBAL_Y_AXIS`와
사실상 동일하게 수렴한다 — "Y-only"라는 아이디어 원안의 의도를 그대로 보존하면서 각도
플랜지에서만 방향을 보정한다. 이 값은 `identify_mating_flange_pairs()`에서 **1회 계산 후
버퍼로 캐싱**한다(파트가 파트 단위 스칼라 Δy_part 하나만 갖는다는 원칙은 그대로 유지 —
`t̂_pair`는 이동 방향일 뿐, "노드마다 다른 배율"을 재도입하는 것이 아니다).

### §3.2 부호(sign) 캐싱 — Critic 라운드 확정 (OpenAI 지적 #3, Gemini 수용)

CAD 원점이 파트마다 뒤집혀 있을 수 있으므로, 부호는 **`identify_mating_flange_pairs()`
실행 시점에 1회 계산해 버퍼로 저장**하고 이후 절대 재계산하지 않는다:

```python
s_PC = torch.sign(torch.dot(centroid_P0 - centroid_C0, GLOBAL_Y_AXIS))  # 초기 형상 기준
self.register_buffer('flange_s_PC', s_PC)  # shape (P,), 파트쌍마다 1개
```

### §3.3 파트별 Δy — 계층적 앵커 전파 (Problem B 최소위험 정책)

`Outer`를 앵커(`Δy_Outer = 0`)로 고정하고, `Plate → {Inner, Patch1}`, `Plate/Inner → Patch2`
순으로 전파한다. `Δt_X = t_raw_X - t_raw0_X`(파트 내 평균, §3.4).

```
Δy_Outer = 0
Δy_Plate  = Δy_Outer + s_(Plate←Outer) · 0.5·(Δt_Outer + Δt_Plate)
Δy_Inner  = Δy_Plate + s_(Inner←Plate) · 0.5·(Δt_Plate + Δt_Inner)
Δy_Patch1 = Δy_Plate + s_(Patch1←Plate) · 0.5·(Δt_Plate + Δt_Patch1)
```

**Inner-Patch2 샌드위치 예외** (목표 간극이 0이 아니라 `t_Plate`인 관계, 기존
`FLANGE_PAIR_SPEC`과 동일 취급): Patch2는 Plate/Inner 두 값으로부터 독립적으로 유도되는
2개의 요구가 상충할 수 있으므로(과결정), **재보정(re-shift)을 시도하지 않고** 다음 하나의
식으로만 결정한다(Inner 경유 경로를 우선):

```
Δy_Patch2 = Δy_Inner + s_(Patch2←Inner) · (0.5·(Δt_Inner + Δt_Patch2) + Δt_Plate)
```

이후 남는 잔차 `ε = (y_Patch2 - y_Inner) - (필요 간극)`는 §5에서 collision loss로 흡수한다
(좌표를 다시 건드리지 않음 — Critic 라운드에서 재조정 시 진동 위험이 지적되어 기각됨).

### §3.4 Outer 삭제 시 폴백 앵커 — Critic 라운드 확정 (OpenAI 지적 #2, must-fix, Gemini 수용)

`z_gate[sec, Outer] < BINARIZE_THRESHOLD`(파트 삭제)인 경우 앵커를 다음 순서로 대체한다:

```python
if not outer_alive:
    anchor = 'Plate' if plate_alive else first_alive_part_by_id_order()
    if anchor is None:
        Δy_part[:] = 0.0   # 모든 파트 삭제된 극단 케이스 — 학습 안정성 우선
```

이 로직은 매 forward마다 `z_gate`로부터 벡터화 연산으로 계산한다(if/else 분기가 아니라
`torch.where` 기반 마스킹으로 구현해 미분 그래프를 끊지 않는다).

### §3.5 안전 클램프 (방어적, Critic 라운드에서 "필수는 아니나 저비용이라 포함" 결론)

```python
bbox_y = part_bbox_y[part_id]  # 초기 형상 기준 파트별 y-bbox 크기, 버퍼로 사전 계산
Δy_part = torch.clamp(Δy_part, min=-RIGID_SHIFT_CLIP_FRAC * bbox_y,
                                 max= RIGID_SHIFT_CLIP_FRAC * bbox_y)
```

### §3.6 파트 내 균일 Δt (기존 `delta_t_part` 로직 재사용)

`Δt_X`는 파트 X에 속한 전체 노드(플랜지 노드 + 자유 노드 구분 없이)의 `t_raw - t_raw0` 평균이다.
기존 L1265-1269의 `group_mean`/`scatter_add_` 패턴을 파트 단위로 그대로 재사용할 수 있다
(이미 벡터화되어 있고 미분 가능).

---

## §4 `forward()` 통합 — L1287-1329 교체 블록

```python
        # ── [v1.16] 파트 단위 강체 Y-shift — 테이퍼 완전 대체 ────────────────
        # 근거: docs/idea/idea_v1_16.md, docs/command/command_v1_16.md
        # 노드별 오프셋(v1.15 테이퍼)을 제거하고, 파트마다 하나의 스칼라 Δy_part만
        # 계산해 그 파트의 모든 노드(플랜지+자유 구분 없음)에 동일하게 적용한다.
        rigid_offset = torch.zeros_like(new_coords)
        if getattr(self, '_flange_ready', False) and RIGID_Y_SHIFT_ENABLED:
            t_raw0 = t_min + (t_max - t_min) * torch.sigmoid(t_initial_logit)
            dt = (t_raw - t_raw0).squeeze(-1)                          # (N,) Δt

            # §3.6: 파트별 평균 Δt (기존 group_mean 패턴 재사용, part_ids_local 기준)
            dt_part = torch.zeros(self.num_parts, device=x.device, dtype=dt.dtype)
            part_count = torch.zeros(self.num_parts, device=x.device, dtype=dt.dtype)
            dt_part.scatter_add_(0, part_ids_local, dt)
            part_count.scatter_add_(0, part_ids_local, torch.ones_like(dt))
            dt_part = dt_part / part_count.clamp(min=1)

            # §3.4: Outer 삭제 시 앵커 폴백 (torch.where 기반, 미분 그래프 유지)
            outer_alive = (z_gate[:, 0] >= BINARIZE_THRESHOLD)         # (17,) 섹션별
            plate_alive = (z_gate[:, 1] >= BINARIZE_THRESHOLD)
            # anchor_part: 0=Outer 우선, 1=Plate 폴백, 이후 0 강제(모두 삭제 시 Δy=0)
            anchor_is_outer = outer_alive
            anchor_is_plate = (~outer_alive) & plate_alive

            # §3.3: 계층적 전파 (섹션별 벡터 연산)
            dy_part = torch.zeros(NUM_SECTIONS, self.num_parts, device=x.device)
            dy_outer_anchor = torch.zeros(NUM_SECTIONS, device=x.device)  # Δy_Outer=0 항상
            dy_plate = torch.where(
                anchor_is_outer,
                self.flange_s_PC[self.PLATE_OUTER_PAIR_IDX] * 0.5 * (dt_part[0] + dt_part[1]),
                torch.zeros(NUM_SECTIONS, device=x.device),  # Outer도 Plate도 없으면 0
            )
            dy_part[:, 0] = dy_outer_anchor
            dy_part[:, 1] = dy_plate
            dy_part[:, 2] = dy_plate + self.flange_s_PC[self.INNER_PLATE_PAIR_IDX] * 0.5 * (dt_part[1] + dt_part[2])
            dy_part[:, 3] = dy_plate + self.flange_s_PC[self.PATCH1_PLATE_PAIR_IDX] * 0.5 * (dt_part[1] + dt_part[3])
            # Inner-Patch2 샌드위치 예외 (§3.3): Inner 경유, 잔차는 §5 collision loss로 흡수
            dy_part[:, 4] = dy_part[:, 2] + self.flange_s_PC[self.PATCH2_INNER_PAIR_IDX] * (
                0.5 * (dt_part[2] + dt_part[4]) + dt_part[1])

            # §3.5: 안전 클램프
            dy_part = torch.clamp(dy_part, min=-RIGID_SHIFT_CLIP_FRAC * self.part_bbox_y,
                                             max= RIGID_SHIFT_CLIP_FRAC * self.part_bbox_y)

            # §3.1: 파트별 방향(t̂_pair)을 노드에 gather. 파트 자체는 단일 방향을 쓰므로
            # self.part_y_direction (num_parts, 2) 로 사전 캐싱된 값을 사용.
            dy_node = dy_part[section_ids_local, part_ids_local]           # (N,)
            rigid_offset = dy_node.unsqueeze(-1) * self.part_y_direction[part_ids_local]  # (N,2)

            new_coords = new_coords + rigid_offset

        # [v1.16] 반환 개수를 6개로 유지, 오프셋은 속성으로 전달(§5에서 사용)
        self.last_flange_offset = rigid_offset
        # ────────────────────────────────────────────────────────────────────

        return new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open
```

**사전 계산 필요 버퍼** (`identify_mating_flange_pairs()`에서 1회 계산, `register_buffer`):
- `flange_s_PC` — §3.2, shape `(P,)`.
- `part_y_direction` — §3.1, shape `(num_parts, 2)`. 파트마다 자신이 속한 대표 플랜지 쌍의
  `t̂_pair`를 사용(파트가 여러 쌍에 관여하면 Outer 방향 쪽 쌍을 대표값으로 채택 — 앵커 전파
  순서와 일관성 유지).
- `part_bbox_y` — 초기 형상 기준 파트별 y-bbox 크기, shape `(num_parts,)`.
- `PLATE_OUTER_PAIR_IDX` 등 4개 상수 — `flange_s_PC` 배열 내 해당 쌍의 인덱스.

---

## §5 Collision-Loss 상호작용

강체 이동이 명목상의 플랜지 간극을 해석적으로 처리하므로, `loss_collision`은 비-플랜지
경계 위반에만 집중한다. 단, **Inner-Patch2 잔차**(§3.3에서 재조정하지 않기로 한 부분)는
별도 항으로 명시적으로 주입한다:

```python
epsilon = (y_Patch2 - y_Inner) - required_gap_inner_patch2   # required_gap = t_Plate 기반
loss_residual_inner_patch2 = RESIDUAL_COLLISION_LAMBDA * torch.clamp(
    epsilon - RESIDUAL_COLLISION_MARGIN_MM, min=0.0).pow(2)
```

이 항을 기존 `compute_collision_loss_v5()` 반환값에 더해 총 손실에 합산한다. Critic 라운드
합의: 좌표를 다시 조정(재-shift)하지 말 것 — 이전 파트 정렬을 해치고 진동을 유발할 위험이
있음(OpenAI 지적, Gemini 동의).

---

## §6 검증 및 테스트 계획

두 모델의 Solver+Critic 라운드에서 공통 합의된 항목:

1. **제로-델타 회귀 테스트**: `initial_section_v4.csv`(Δt=0)로 실행. `rigid_offset`이
   정확히 0이고 어떤 파트도 움직이지 않는지 확인.
2. **강체 이동 검증**: `Plate`에 인위적으로 +2.0mm 두께 변화 주입. 확인 사항:
   - `Outer`의 Y-shift = 0.0 유지.
   - `Plate`에 속한 **모든** 노드(플랜지+자유 노드)가 float32 정밀도까지 동일한 Y-shift를
     갖는지(비-플랜지 노드 누락 여부가 must-fix였음 — §4의 `dy_node` gather가 파트 전체
     노드에 적용됨을 코드 리뷰로 재확인).
3. **그래디언트 흐름 테스트**: `loss.backward()`가 `t_logit`까지 0이 아닌 그래디언트를
   계산하는지 확인(`torch.autograd.gradcheck` 또는 수치 미분 비교).
4. **Outer 삭제 폴백 테스트**: `z_gate[:, 0]`을 강제로 0으로 설정한 상태에서 forward 실행 —
   NaN/Inf 없이 `Plate`가 대체 앵커로 동작하는지 확인(§3.4의 must-fix 항목).
5. **각도 플랜지 회귀 테스트**: 각도가 있는 플랜지 쌍(있다면)에 대해 `proj_y(flange_normal)`
   적용 전/후 겹침을 비교 — degenerate tolerance 분기(`PROJ_Y_DEGENERATE_TOL`)가 정상
   케이스를 오염시키지 않는지 확인.
6. **Outer-Plate류 패턴 소거 확인**: 모든 등록 플랜지 노드가 소수점 단위까지 동일 y를
   갖는지 확인 — "가운데 노드만 어긋나는" 패턴(원본 버그)이 완전히 사라졌는지가 최종 판정
   기준.
7. **대형 Δt 스트레스 테스트**: 모든 파트에 +4mm 두께 변화를 동시에 주입해 §3.5 안전
   클램프가 개입하는지, X축 방향 crowding으로 인한 자기 교차가 없는지 육안/수치 확인.

---

## §7 리스크 로그

| 항목 | 상태 | 비고 |
|---|---|---|
| 각도 플랜지에서 전역 Y가 형상 왜곡 | **해결** | §3.1 `proj_y(flange_normal)`로 대체(Critic 확정) |
| Outer 삭제 시 앵커 전파 실패 | **해결** | §3.4 폴백 체인(Plate→첫 생존 파트→Δy=0) |
| CAD 원점 부호 뒤집힘으로 인한 방향 오류 | **해결** | §3.2 `flange_s_PC` 초기 1회 계산 후 버퍼 캐싱, 재계산 금지 |
| 비-플랜지 노드가 이동에서 누락 | **해결** | §4 `dy_node` gather를 파트 전체 노드에 적용(플랜지/자유 구분 없음) |
| Inner-Patch2 과결정(재조정 불가) | **정책으로 흡수** | §3.3에서 재-shift 금지, §5 collision loss로 잔차 처리 — 완전 소거는 아님, QA가 회귀로 오인하지 않도록 §5에 명시 |
| 대형 Δt에서 X축 crowding으로 자기 교차 | **완화(미완전 해결)** | §3.5 클램프는 방어적 조치일 뿐, 근본적으로 X축 자유도는 다루지 않음 — §6 테스트 7에서 지속 모니터링 필요 |
| Plate의 4-이웃 과결정(idea 문서의 "문제 B") | **정책적 완화, 완전 해결 아님** | Outer 우선 앵커 정책으로 잔차를 최소화하지만, idea_v1_16.md §3의 판단대로 구조적으로 완전히 사라지지는 않음 |

**최종 신뢰도**: Gemini(Critic, conf100, can_exit=true) · OpenAI(Critic, conf70,
can_exit=true, must-fix 2건 모두 본 스펙에 반영 완료) — 두 모델 모두 추가 Defense 라운드
불필요 판정. Judge(Claude) 종합: 6개 쟁점 전부 구체적 수식/코드 수준으로 해소, 단 "문제 B"
(Plate 다중 이웃)와 "대형 Δt X축 crowding"은 완화이지 완전 해결이 아님을 명시적으로 유지
(idea 문서의 원래 판단과 일관).

**작업지시서 신뢰도: 92%** (구현 세부사항은 두 모델의 실질적 수렴으로 확정됐으나, 실제
`AI_design_v1_16.py` 작성 후 §6 테스트 4·5·7에서 예상치 못한 케이스가 나올 가능성 존재)
