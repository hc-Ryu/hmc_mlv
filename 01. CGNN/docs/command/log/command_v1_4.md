# command_v1_4.md — AI_design_v1_3.py → AI_design_v1_4.py: 관통·삭제부재 수정 지시서

작성일: 2026-08-08 (`/synod design` 세션, Gemini flash conf 98 + OpenAI o3 conf 93)
참조: `docs/idea/idea_v1_4.md`(설계 근거), `docs/review/review_v1_3.md`(실행 결과 리뷰),
`AI_design_v1_3.py`(기반 코드), `uni_section_v18.py`(**불변 — import만**)

> **앵커 규칙**: 본 문서의 위치 지정은 **함수/변수 이름 기준**이다. 라인 번호는 참고용이며
> 코드 변경으로 어긋날 수 있다.

> **불변 규칙**: `uni_section_v18.py`는 **한 줄도 수정하지 않는다.**

---

## §0. 목표 및 산출물

`AI_design_v1_3.py`를 복사해 **`AI_design_v1_4.py`** 를 만들고, 실행 결과 리뷰(`review_v1_3.md`)에서
확정된 두 결함을 고친다.

| 결함 | 근본 원인 | 대응 |
|---|---|---|
| **Inner가 Plate를 관통** | (a) `usv18.compute_alda_loss()`의 `f=area/target_area`가 relu 게이트 없이 상시 하강압 (b) `uni_section_v18.build_collision_spec()`이 초기 형상 1회로 clearance를 고정, `compute_collision_loss_v5()`가 관통 1mm 이상에서 `.clamp(max=1.0)`로 그래디언트 소멸 | §2(보조 무클램프 충돌 페널티), §3(면적 하한 가드레일) |
| **34개 게이트 중 0개 삭제** | 엔트로피 항이 대칭이라 "삭제 유도"가 아니라 "현재 위치 강화" 항으로 작동, `init_log_alpha=2.0`(z≈0.986)에서 시작 + `w_sparse=0.0`이라 0쪽으로 미는 힘이 전무 | §4(중립 초기화 + 지연·어닐링 엔트로피) |
| (감시 공백) §7 R1~R3이 "사망 이벤트"만 감지, "삭제 부재" 자체는 못 잡음 | — | §5(R4 신규) |

**적용 순서**: §2·§3(관통 방지)과 §4(삭제 유도)는 서로 독립적이므로 순서 무관하게 함께 적용 가능하다.
단 §5(R4)는 §4 적용 후에만 의미가 있으므로 마지막에 넣는다.

---

## §1. 신규 상수 블록

기존 v1.3 상수 블록(`W_SPARSE_V13` 등) 바로 뒤에 추가한다. **배치 위치는 v1.3과 동일한 원칙** —
import 직후 / `class CGDN17` 정의보다 앞.

```python
# ══════════════════════════════════════════════════════════════════
# [v1.4] 로컬 오버라이드 상수 — 관통 방지 + 삭제 유도
# 근거: docs/idea/idea_v1_4.md, docs/review/review_v1_3.md
# ══════════════════════════════════════════════════════════════════

# §2. 보조 무클램프 충돌 페널티
PENETRATION_FLOOR = 0.8     # mm. 이 깊이부터 무클램프 페널티가 발동(원본 손실이 아직 유효한 얕은
                             # 침투 구간은 원본에 맡기고, 원본이 죽는 1mm 근방부터 인계받는다)
W_COL_AUX = 5.0              # 보조 충돌 페널티 가중치. §7 1단계에서 실측 후 조정

# §3. 면적 하한 가드레일 — f(=area/target_area)의 상시 하강압을 상쇄
SAFETY_RATIO = 0.90          # 목표 면적의 90% 밑으로는 반대 방향 벌점 발동
W_AREA_FLOOR = 10.0           # W_MASS_V13과 대칭적인 크기로 시작(§2.6.3 근거와 동일 스케일)

# §4. 중립 초기화 + 지연·어닐링 엔트로피
INIT_LOG_ALPHA_V14 = 0.0     # usv18.CGDN 기본 동작(2.0) 대체. z_init ≈ 0.5 (entropy의 불안정 안장점)
ALPHA_ENT_START_EPOCH = None  # None = ASWC_STAGE2_END로 자동 설정(§4.2). 그 전까지 alpha_ent=0
ALPHA_ENT_ANNEAL_EPOCHS = 50  # START_EPOCH부터 이 기간에 걸쳐 0 → ALPHA_ENT_V13으로 선형 어닐링

# §5. R4 — 삭제-부재 감시 (진단 전용, raise 하지 않음)
R4_GRACE_EPOCHS = 200         # ASWC_STAGE2_END + 이 값까지 어떤 게이트도 0.5 밑으로 안 내려가면 경고
```

---

## §2. 보조 무클램프 충돌 페널티

### §2.1 왜 필요한가

`uni_section_v18.compute_collision_loss_v5()`(불변, `AI_design_v1_3.py`가 그대로 호출)는
`violation = relu(-gap).clamp(max=1.0)`로 위반량을 1.0mm에서 클램프한다. 관통이 1mm를 넘는 순간
이 항의 그래디언트가 정확히 0이 되어, 옵티마이저가 되돌아올 유인을 완전히 잃는다
(`review_v1_3.md` §[ERROR]2 — Gemini 단독 발견, conf 98).

이 항은 **대체가 아니라 보강**이다 — 원본 손실이 아직 유효한 0~0.8mm 구간은 원본에 맡기고, 0.8mm
이상부터는 클램프 없는 이차 페널티가 인계받아 관통 깊이에 비례해 계속 커지게 한다.

### §2.2 함수 시그니처 — `compute_collision_loss_v5`와 동일한 인자를 받는다

> **★ 설계 검증 (개정 — 초안의 위험 수정)**: 초안은 "`collision_spec`을 순회하며
> `usv18._signed_projections()`를 호출한다"고만 서술했는데, `collision_spec`은 `sigma`/`clearance`/
> 파트 역할(`seg_part`/`pt_part`) **메타데이터만** 담고 있고 좌표 자체는 담지 않는다(코드 확인
> 완료 — `build_collision_spec()`은 `torch.no_grad()` 블록 안에서 파트 쌍마다 `sigma`/`clearance`
> 스칼라만 계산해 저장한다). 따라서 좌표는 반드시 **매 스텝의 살아있는 `new_coords`/`t_final`**을
> 함수 인자로 받아야 하며, `collision_spec`에서 캐시된 좌표를 재사용해서는 안 된다(그렇게 하면
> autograd 그래프가 끊겨 그래디언트가 전혀 흐르지 않는 침묵 버그가 된다).

```python
def compute_collision_penalty_unclamped(new_coords, t_final, part_ids, section_ids,
                                         collision_spec, penetration_floor=PENETRATION_FLOOR):
    """[v1.4 §2] compute_collision_loss_v5()와 동일한 시그니처·순회 구조를 쓰되, 위반량에
    .clamp(max=1.0)을 적용하지 않는다. new_coords/t_final은 매 스텝 살아있는 텐서를 받아야 한다
    (collision_spec은 sigma/clearance 메타데이터만 담고 좌표는 담지 않으므로)."""
    total = torch.tensor(0.0, device=new_coords.device, requires_grad=True)
    n_dirs = 0
    for (sec_int, a, b), directions in collision_spec.items():
        sec_mask = (section_ids == sec_int)
        part_coords = {
            a: new_coords[sec_mask & (part_ids == a)],
            b: new_coords[sec_mask & (part_ids == b)],
        }
        part_t = {
            a: t_final[sec_mask & (part_ids == a)].mean(),
            b: t_final[sec_mask & (part_ids == b)].mean(),
        }
        for d in directions:
            c_seg = part_coords[d['seg_part']]
            c_pt  = part_coords[d['pt_part']]
            if c_seg.shape[0] < 2 or c_pt.shape[0] == 0:
                continue
            proj, valid = usv18._signed_projections(c_seg, c_pt)   # 불변 모듈의 private 헬퍼, import 가능
            if valid.sum() == 0:
                continue
            t_sum_half = (part_t[d['seg_part']] + part_t[d['pt_part']]) / 2.0
            gap = d['sigma'] * proj - t_sum_half - d['clearance']
            # ★ 핵심 차이: .clamp(max=1.0) 없음 — 관통이 깊을수록 계속 커진다
            violation = torch.relu(-gap - penetration_floor) * valid.float()
            loss_dir = (violation ** 2).sum() / valid.float().sum()
            total = total + loss_dir
            n_dirs += 1
    if n_dirs > 0:
        total = total / n_dirs
    return total
```

### §2.3 `train_step_multi()`에 편입

- **대상**: `l_collision` 계산 다음 줄(기존 `compute_collision_loss_v5` 호출부 바로 아래)

```python
# TO-BE — l_collision 계산 다음
l_collision = usv18.compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids,
                                               collision_spec, z_gate=z_gate_part_avg)
l_col_aux = compute_collision_penalty_unclamped(new_coords, t_final, part_ids, section_ids,
                                                collision_spec)   # [v1.4 §2] 무클램프 보강
```

- **loss 조립부**(다른 `contrib_*`와 동일 위치, stage 무관):

```python
contrib_col_aux = W_COL_AUX * l_col_aux

loss = (contrib_phys + contrib_smooth + L_alda_effective
        + contrib_order + contrib_anchor + contrib_sat
        + contrib_sparse + contrib_entropy + contrib_continuity
        + contrib_mass + contrib_col_aux)                # ← 추가
```

- **반환 dict에도 추가**(로그·리포트 추적용): `"l_col_aux": l_col_aux.item(), "contrib_col_aux": contrib_col_aux.item()`

### §2.4 검증

`grep -n "compute_collision_penalty_unclamped(" AI_design_v1_4.py` → 정의 1 + 호출 1 = **정확히 2곳**.

---

## §3. 면적 하한 가드레일

### §3.1 왜 필요한가

`usv18.compute_alda_loss()`의 `f = area/target_area`는 relu가 없는 **상시 활성 선형항**이다.
`l_mass`(v1.3, 목표 초과분만 벌점)가 0이 되어도 `f`는 멈추지 않아, review_v1_3.md가 확인한 대로
Stage4에서 area가 목표 밑인데도 8.5% 더 줄었다. 이를 상쇄하는 **반대 방향 벌점**을 추가한다.

### §3.2 `train_step_multi()`의 `l_mass` 계산 다음 줄에 추가

- **대상**: v1.3에서 이미 `l_mass = torch.relu(area / target_area - 1.0) ** 2` 바로 다음

```python
# TO-BE — l_mass 계산 다음 줄
l_mass = torch.relu(area / target_area - 1.0) ** 2
l_area_floor = torch.relu(1.0 - area / (SAFETY_RATIO * target_area + 1e-12)) ** 2   # [v1.4 §3]
```

`area`가 이미 스칼라(`usv18.compute_mass_loss()`가 엣지 합산으로 반환)이므로 `l_area_floor`도
추가 reduction 없이 스칼라다 — `l_mass`와 동일한 텐서 shape·reduction 방식을 그대로 따른다.

### §3.3 loss 조립부

```python
contrib_area_floor = W_AREA_FLOOR * l_area_floor

loss = (contrib_phys + contrib_smooth + L_alda_effective
        + contrib_order + contrib_anchor + contrib_sat
        + contrib_sparse + contrib_entropy + contrib_continuity
        + contrib_mass + contrib_col_aux + contrib_area_floor)   # ← 추가
```

반환 dict에도 `"l_area_floor": l_area_floor.item(), "contrib_area_floor": contrib_area_floor.item()`
추가.

### §3.4 `weights` 딕셔너리에 키 추가 (CLI 튜닝 가능하도록)

- **대상**: `run_training_multi()`의 `weights = {...}` 기본값

```python
weights = {'w_phys': 10.0, 'w_order': 1.0, 'w_smooth': 0.5, 'w_anchor': 0.02, 'w_sat': 0.01,
           'w_sparse': W_SPARSE_V13, 'w_mass': w_mass,
           'w_area_floor': w_area_floor}   # [v1.4] §3.4 — CLI로 조정 가능하게 노출
```

`train_step_multi()`의 `contrib_area_floor`는 `weights['w_area_floor']`를 참조하도록
(`W_AREA_FLOOR` 상수는 기본값으로만 사용). `run_training_multi()`에 `w_area_floor=W_AREA_FLOOR`
키워드 인자를 추가하고 `__main__`에 `--w-area-floor` CLI 인자를 추가한다(v1.3의 `--w-mass` 패턴과
동일).

---

## §4. 중립 초기화 + 지연·어닐링 엔트로피

### §4.1 초기화 버그 수정 — `init_log_alpha` 파라미터 재활용

> **★ 발견 (Gemini 단독, conf 98)**: `CGDN17.__init__`는 이미 `init_log_alpha` 인자를 받지만,
> `log_alpha_candidates`는 **리터럴 `2.0`**으로 하드코딩되어 있어 이 인자가 실제로는 아무 효과가
> 없다(부모 클래스가 삭제되는 `self.log_alpha`에만 영향을 줄 뿐). 새 전역 상수를 추가하는 대신
> **이 죽은 파라미터를 되살리는 것**이 더 깨끗한 수정이다.

- **대상**: `CGDN17.__init__`

- **AS-IS**

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    del self.log_alpha
    self.log_alpha_candidates = torch.nn.Parameter(
        torch.full((NUM_SECTIONS, len(CANDIDATE_PARTS)), 2.0)  # 원본 init_log_alpha=2.0과 동일 초기값
    )
```

- **TO-BE**

```python
def __init__(self, *args, init_log_alpha=2.0, **kwargs):
    super().__init__(*args, init_log_alpha=init_log_alpha, **kwargs)
    del self.log_alpha
    self.log_alpha_candidates = torch.nn.Parameter(
        torch.full((NUM_SECTIONS, len(CANDIDATE_PARTS)), float(init_log_alpha))  # [v1.4] §4.1 — 죽은 파라미터 재활용
    )
```

- **호출부 두 곳 모두 수정**: `run_training_multi()`의 `model = CGDN17(...)`와 `__main__`의
  dry-run 전용 `model = CGDN17(...)` 양쪽에서 `init_log_alpha=2.0` → `init_log_alpha=INIT_LOG_ALPHA_V14`
  로 변경. `grep -n "init_log_alpha=" AI_design_v1_4.py` → 정의부(`__init__`) 1곳 + 호출부 2곳
  = **총 3곳**, 호출부 2곳 모두 `INIT_LOG_ALPHA_V14`를 참조하는지 확인.

### §4.2 엔트로피 지연·어닐링 스케줄

- **대상**: `run_training_multi()` 상단(메인 루프 진입 전)에 헬퍼 함수 정의, 메인 루프의
  `train_step_multi()` 호출부에서 상수 대신 스케줄 값을 계산해 전달.

```python
# run_training_multi() 내부, 메인 루프 진입 전
alpha_ent_start = ALPHA_ENT_START_EPOCH if ALPHA_ENT_START_EPOCH is not None else ASWC_STAGE2_END

def alpha_ent_schedule(ep):
    """[v1.4 §4.2] 지연·어닐링. START 이전엔 0(순수 물리/면적 신호만), 이후 ANNEAL_EPOCHS에
    걸쳐 0 → ALPHA_ENT_V13으로 선형 상승."""
    if ep < alpha_ent_start:
        return 0.0
    t = min(1.0, (ep - alpha_ent_start) / ALPHA_ENT_ANNEAL_EPOCHS)
    return ALPHA_ENT_V13 * t
```

- **메인 루프 호출부 TO-BE** (v1.3의 `alpha_ent=ALPHA_ENT_V13` 리터럴을 대체):

```python
info = train_step_multi(model, data, optimizer, target_mps, target_area,
                         epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                         collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                         alpha_ent=alpha_ent_schedule(epoch), stage4_active=False)  # [v1.4] §4.2
```

**Stage 4 호출부는 변경하지 않는다** — 이미 `alpha_ent=alpha_ent_active`(0.0 고정)를 쓰고 있고,
Stage4는 게이트가 동결된 뒤이므로 스케줄과 무관하다.

### §4.3 로그에 스케줄 값 노출 (관찰용)

기존 epoch 로그 라인에 `alpha_ent_now` 항목을 추가해 지연·어닐링이 실제로 작동하는지 확인한다.

```python
print(f"Epoch {epoch:4d} || ... | alpha_ent={alpha_ent_schedule(epoch):.4f} | ...")
```

---

## §5. R4 — 삭제-부재 감시 규칙 (진단 전용, §7과 별개)

### §5.1 왜 필요한가

§7(v1.3)의 R1~R3은 전부 "사망 이벤트"(SUSPECT death, 연쇄 붕괴, 질량 역설 재발)를 트리거로
삼는다. **애초에 사망이 한 건도 없는** 이번 실패 모드는 세 규칙 중 어느 것도 감지하지 못한다.

### §5.2 구현 — `log_alpha_candidates < 0`이 곧 `z_gate < 0.5`와 정확히 동치

stretched sigmoid `z = clamp(sigmoid(la)*(ζ-γ)+γ, 0, 1)`, γ=-0.1, ζ=1.1에서
`sigmoid(la)=0.5 ⟺ la=0`이고 이때 `z = 0.5*1.2-0.1 = 0.5`다. 즉 **`z=0.5` 경계는 정확히
`la=0`과 일치**하므로, stretched sigmoid를 다시 계산할 필요 없이 `log_alpha_candidates`의 부호만
보면 된다(가장 저렴하고 정확한 방법).

- **대상**: `run_training_multi()` 메인 루프 진입 전 초기화, 루프 안 매 epoch 체크

```python
# 메인 루프 진입 전
any_gate_crossed_half = False   # [v1.4 §5] R4 상태 — Stage4까지 이어지므로 루프 밖에서 선언

# 메인 루프 안, epoch 로그 블록 근처
if not any_gate_crossed_half and bool((model.log_alpha_candidates < 0.0).any()):
    any_gate_crossed_half = True
    print(f"[R4] epoch {epoch}: 최초로 게이트가 0.5 미만으로 하강함 (정상 신호)")

if epoch == ASWC_STAGE2_END + R4_GRACE_EPOCHS and not any_gate_crossed_half:
    print(f"  !! [R4 WARNING] epoch {epoch} 시점까지 34개 게이트 중 단 하나도 0.5 밑으로 "
          f"내려가지 않았습니다 — 삭제 유도력이 여전히 부족할 수 있습니다 "
          f"(command_v1_4.md §4 파라미터 재검토 권장).")
```

`any_gate_crossed_half`는 **루프 밖에서 선언**하므로 Stage4 루프(§5.5, 별도 for문)에서도 값이
유지된다 — 별도로 전달할 필요 없다. R1~R3과 달리 **`raise`하지 않는다**(진단 전용,
학습을 막지 않음).

### §5.3 검증

`grep -n "any_gate_crossed_half" AI_design_v1_4.py` → 선언 1 + 갱신 1 + 경고 조건 1 = 최소 3곳.

---

## §6. Go/No-Go 판정 기준

| 항목 | 통과 조건 | 실패 시 조치 |
|---|---|---|
| §2 관통 방지 | 최종 3D 결과에서 Inner-Plate 관통 육안 확인 없음. 리포트에 `l_col_aux` 값이 학습 후반 0에 수렴 | `W_COL_AUX` 상향(5.0→10.0), `PENETRATION_FLOOR` 하향(0.8→0.5) |
| §3 면적 하한 | 최종 area가 `[SAFETY_RATIO×target, target]` 밴드 안에 안착(과도 초과달성 방지) | `W_AREA_FLOOR` 상향 또는 `SAFETY_RATIO` 상향(0.90→0.92) |
| §4 삭제 유도 | 34개 게이트 중 **최소 1개 이상**이 `z<0.5`로 하강(완전 삭제까지는 요구하지 않음 — "유도력 복원"이 이번 목표) | `ALPHA_ENT_START_EPOCH` 상향(늦게 켬) 또는 `INIT_LOG_ALPHA_V14`를 음수(-0.5)로 더 낮춤 |
| §5 R4 | R4 경고가 뜨지 않음(즉 §4가 실제로 작동) | R4가 뜨면 §4 파라미터 재검토, R1~R3(사망 관련)과는 무관하므로 별도 조치 |

**금지**: R1~R3(v1.3 롤백 규칙)의 임계값을 이번 수정을 정당화하려고 낮추지 않는다 — 사망 감시망과
삭제 유도력 문제는 서로 다른 축이다.

---

## 숙의 과정

<details>
<summary>Synod 세션 상세</summary>

**모델 기여**
- **Gemini (flash, conf 98, pro는 rate limit으로 폴백)**: `collision_spec`이 좌표를 캐싱하지 않고
  메타데이터만 담는다는 점을 근거로 "매 스텝 살아있는 좌표를 인자로 받아야 한다"는 요구사항을
  명확히 함(§2.2). `CGDN17.__init__`의 `init_log_alpha` 파라미터가 죽어있다는 것을 발견하고,
  새 전역 상수 대신 **기존 파라미터를 재활용**하는 더 깔끔한 수정을 제안(§4.1) — 이 세션의
  최고 기여.
- **OpenAI (o3, conf 93)**: 반환 dict·로그에 신규 항(`contrib_col_aux` 등)을 빠뜨리면 관찰이
  불가능해진다는 지적, `weights` 딕셔너리에 `w_area_floor`를 넣어야 CLI 튜닝이 가능하다는 지적
  (§3.4)을 반영했다. 다만 `tests/test_regression.py`, `model.log_alpha_temp`,
  `globals_assert()`, `logger.warning()`+csv 패턴 등 **실제 `AI_design_v1_3.py`에 존재하지 않는
  구조를 언급**해 이 부분은 채택하지 않았다(Claude가 grep으로 재검증 후 기각).
- **Claude (오케스트레이터)**: `z=0.5` 경계가 정확히 `log_alpha=0`과 일치한다는 수학적 사실을
  이용해 R4를 "sigmoid 재계산" 없이 부호 비교만으로 구현하도록 단순화(§5.2) — 두 모델 모두
  제안하지 않았던 최적화.

**신뢰도**: Gemini 98(폴백 모델임에도 실측 근거로 최고 기여), OpenAI 93(일부 환각 포함, 유효
지적과 분리해 채택) — 두 모델의 유효 지적이 서로 겹치지 않고 보완적이었으므로 최종 신뢰도를
**92%**로 판단한다.

</details>
