# command_v1_5.md — AI_design_v1_4.py → AI_design_v1_5.py: 충돌 페널티 폭주 수정 지시서

작성일: 2026-08-09 (`/synod design` 세션, Gemini flash conf 98/95 + OpenAI o3 conf 92/79)
참조: `docs/idea/idea_v1_5.md`(설계 근거, 신뢰도 89%), `docs/review/review_v1_4.md`(실행 결과 리뷰),
`AI_design_v1_4.py`(기반 코드), `uni_section_v18.py`(**불변 — import만**)

> **앵커 규칙**: 본 문서의 위치 지정은 **함수/변수 이름 기준**이다. 라인 번호는 참고용이며
> 코드 변경으로 어긋날 수 있다.

> **불변 규칙**: `uni_section_v18.py`는 한 줄도 수정하지 않는다. **§4(중립 초기화+지연
> 엔트로피)와 §7(R1~R3 롤백)의 상수·로직도 이번 버전에서 절대 수정하지 않는다** — idea_v1_5.md와
> 이번 design 세션 양쪽에서 확정된 제약이다.

> **정정 이력**: idea_v1_5.md는 "W_COL_AUX 웜업이 §4 지연 엔트로피 구간(epoch 50~150)과
> 동기화된다"고 가정했으나, 이번 design 세션에서 코드를 직접 읽어 확인한 결과 **실제 엔트로피
> 어닐링은 `ASWC_STAGE2_END=400`부터 시작**한다(idea_v1_5.md 작성 시점의 가정 오류). 실행 로그
> 상 실제 loss 폭증은 게이트 활성화(`GATE_ACTIVE_EPOCH=11`) 직후부터 시작해 `Epoch 90 ||
> loss=2.3267` → `Epoch 100 || loss=53.5881`로 10-epoch 샘플링 안에서 23배 폭증한다. 따라서
> 본 문서의 웜업 구간은 idea_v1_5.md의 원안(50~150)이 아니라 **실측 로그 기반으로 재설계된
> epoch 11~80** 을 사용한다.

---

## §0. 목표 및 산출물

`AI_design_v1_4.py`를 복사해 **`AI_design_v1_5.py`** 를 만들고, `review_v1_4.md`에서 확정된
크래시 원인(무클램프 제곱 충돌 페널티의 그래디언트 폭발)을 고친다.

| 결함 | 근본 원인 | 대응 |
|---|---|---|
| **epoch 510 `RuntimeError [R3]` 크래시** | `compute_collision_penalty_unclamped()`의 관통 페널티가 무클램프 제곱이라 관통 깊이의 제곱에 비례해 무한정 커짐. `W_COL_AUX=5.0`이 상시 곱해져 게이트가 요동치는 초반(epoch 11~100) 구간에 그래디언트 폭발 유발 | §2(Huber형 상한), §3(W_COL_AUX 웜업) |
| (관측 공백) `l_col_aux`가 반환 dict엔 있으나 콘솔에 미노출 | — | §4(로깅) |

**적용 순서**: §2(Huber 상한)를 먼저 적용해 무클램프 문제를 원천 차단한 뒤, §3(웜업)을 추가한다.
§4(로깅)는 독립적이므로 아무 때나 적용 가능. §2·§3 모두 적용 없이 §4만 넣어도 동작에는 문제
없음(디버깅 우선순위상 §2·§3 먼저 권장).

---

## §1. 신규 상수 블록

기존 v1.4 상수 블록(`PENETRATION_FLOOR`, `W_COL_AUX` 등, L163-177) 바로 뒤에 추가한다. 배치
원칙은 v1.3/v1.4와 동일 — import 직후 / `class CGDN17` 정의보다 앞.

```python
# ══════════════════════════════════════════════════════════════════
# [v1.5] 로컬 오버라이드 상수 — 충돌 페널티 상한 + 웜업
# 근거: docs/idea/idea_v1_5.md, docs/review/review_v1_4.md, docs/command/command_v1_5.md
# ══════════════════════════════════════════════════════════════════

# §2. Huber형 충돌 페널티 상한 — δ는 실제 메시/격자 해상도 상수를 코드에서 찾지 못해(design 세션
# 확인) 확정값 대신 런타임 인자화한다. 기본값은 review_v1_4.md 권장 범위(1.0~1.2mm)의 보수적
# 하한(더 이른 지점에서 선형 전환 = 더 안전)을 사용한다.
HUBER_DELTA_MM = 1.0         # mm. 이 값 이하는 제곱, 초과는 선형 페널티(C1 연속)로 전환

# §3. W_COL_AUX 코사인 웜업 — §4(지연 엔트로피, epoch 400~450)와 무관하게 독립 설계됨에 주의.
# 실측 로그(epoch 90~100 사이 loss 23배 폭증) 기준으로 웜업이 그 이전에 완료되도록 구간을 잡는다.
W_COL_AUX_WARMUP_START = 11   # usv18.GATE_ACTIVE_EPOCH와 동일값(게이트 학습 시작 시점)
W_COL_AUX_WARMUP_END   = 80   # 실측 폭증 시점(epoch 90) 이전에 전체 가중치(W_COL_AUX)에 도달
W_COL_AUX_FLOOR         = 0.5  # 절대값. W_COL_AUX(5.0)의 비율이 아님 — §7.2 참고
```

**§7.2 floor 설계 근거 (design 세션 Judge 판정)**: `W_COL_AUX_FLOOR`는 웜업 시작 시점(epoch 11)
의 초기 가중치를 정하는 파라미터일 뿐이며, 웜업 종료 후(epoch 80 이후) 값은 floor 선택과 무관
하게 항상 전역 상수 `W_COL_AUX=5.0`으로 수렴한다. `floor = 0.5 * W_COL_AUX`(=2.5)로 설정하면
"후반부 충돌 억제력 보강"이라는 잘못된 근거로 웜업 시작 시점의 초기 억제력을 5배 과도하게
키우는 것이므로 **채택하지 않는다**. 반드시 절대값 0.5를 사용한다.

---

## §2. Huber형 충돌 페널티 — `compute_collision_penalty_unclamped()` 수정

**대상**: `AI_design_v1_4.py`의 `compute_collision_penalty_unclamped()` (L766 부근, 관통량을
무클램프 제곱으로 합산하는 부분).

함수 시그니처는 유지한다(기존 호출부 `train_step_multi()` L1237과 100% 호환). 함수 내부에서
관통량(violation, `relu(-gap - penetration_floor)`)을 제곱해 합산하는 지점을 다음 헬퍼로
감싼다:

```python
def _huber_collision_term(violation: torch.Tensor, delta: float = HUBER_DELTA_MM) -> torch.Tensor:
    """[v1.5 §2] violation(mm, >=0)에 대한 Huber-squared 페널티.
    delta 이하: violation**2 (원본과 동일한 제곱 페널티, 얕은 관통에서 정밀한 그래디언트).
    delta 초과: delta*(2*violation - delta) (선형 성장 — 1차 미분이 delta에서 연속(C1),
    관통이 깊어져도 그래디언트가 2*delta로 상한이 걸려 폭발하지 않는다)."""
    return torch.where(
        violation <= delta,
        violation ** 2,
        delta * (2.0 * violation - delta),
    )
```

`compute_collision_penalty_unclamped()` 내부에서 기존에 `violation ** 2`(또는 동등한 제곱
연산)를 직접 합산하던 지점을 `_huber_collision_term(violation)` 호출로 교체한다. `total =
total + violation ** 2` 형태의 라인을 모두 찾아 교체할 것 — 함수 안에 관통 방향(direction)별로
루프가 있으므로 해당 루프 내부의 제곱 연산 지점을 정확히 특정해 교체한다.

**검증**: 파일 하단 또는 별도 스모크 스크립트에서 다음을 확인한다.
```python
d = HUBER_DELTA_MM
assert torch.isclose(_huber_collision_term(torch.tensor(d)), torch.tensor(d**2))
assert torch.isclose(_huber_collision_term(torch.tensor(2*d)), torch.tensor(d*(2*2*d - d)))
# 1차 미분 연속성 확인 (수치 미분)
eps = 1e-4
lo = _huber_collision_term(torch.tensor(d - eps))
hi = _huber_collision_term(torch.tensor(d + eps))
assert abs((hi - lo).item() / (2*eps) - 2*d) < 0.5   # 경계에서 기울기가 2*delta로 수렴
```

---

## §3. W_COL_AUX 코사인 웜업 스케줄

### §3.1 `run_training_multi()` 내부 스케줄러 추가

`alpha_ent_schedule(ep)` 클로저(L1383-1387) 바로 아래에 동일한 패턴으로 추가한다:

```python
# [v1.5 §3] W_COL_AUX 코사인 웜업 — §4 엔트로피 스케줄(alpha_ent_start=400)과 독립적으로 설계됨.
# floor는 절대값(design 세션 Judge 판정, §7.2). 종료 시점(80)은 실측 폭증 시점(epoch 90) 이전.
def w_col_aux_schedule(ep):
    if ep < W_COL_AUX_WARMUP_START:
        return W_COL_AUX_FLOOR
    if ep >= W_COL_AUX_WARMUP_END:
        return W_COL_AUX
    span = W_COL_AUX_WARMUP_END - W_COL_AUX_WARMUP_START
    progress = (ep - W_COL_AUX_WARMUP_START) / span
    cos_factor = (1.0 - math.cos(math.pi * progress)) / 2.0
    return W_COL_AUX_FLOOR + (W_COL_AUX - W_COL_AUX_FLOOR) * cos_factor
```

### §3.2 `train_step_multi()` 시그니처 변경

현재 L1257에서 `contrib_col_aux = W_COL_AUX * l_col_aux`로 전역 상수를 직접 참조한다. 이를
호출부에서 전달받은 값으로 바꾼다.

```python
# 변경 전 (L1118 부근)
def train_step_multi(model, data, optimizer, target_mps, target_area,
                      epoch, max_epochs, weights, curriculum,
                      curriculum_ratio, collision_spec, alda_state,
                      pruning_state=None, segment_ids=None,
                      alpha_ent=ALPHA_ENT_V13, stage4_active=False, frozen_curriculum=None):

# 변경 후 — w_col_aux_val 인자 추가 (기본값은 전역 상수로 폴백, 호출부 누락 시 안전)
def train_step_multi(model, data, optimizer, target_mps, target_area,
                      epoch, max_epochs, weights, curriculum,
                      curriculum_ratio, collision_spec, alda_state,
                      pruning_state=None, segment_ids=None,
                      alpha_ent=ALPHA_ENT_V13, stage4_active=False, frozen_curriculum=None,
                      w_col_aux_val=W_COL_AUX):
```

L1257 수정:
```python
# 변경 전
contrib_col_aux    = W_COL_AUX * l_col_aux                        # [v1.4 §2]
# 변경 후
contrib_col_aux    = w_col_aux_val * l_col_aux                    # [v1.5 §3] 웜업 스케줄 값 사용
```

### §3.3 `run_training_multi()` 호출부 수정 (L1405-1408)

```python
# 변경 전
info = train_step_multi(model, data, optimizer, target_mps, target_area,
                         epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                         collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                         alpha_ent=alpha_ent_schedule(epoch), stage4_active=False)

# 변경 후
curr_w_col_aux = w_col_aux_schedule(epoch)                        # [v1.5 §3]
info = train_step_multi(model, data, optimizer, target_mps, target_area,
                         epoch, curriculum_max_epochs, weights, curriculum, curriculum_ratio,
                         collision_spec, alda_state, pruning_state, segment_ids=seg_ids,
                         alpha_ent=alpha_ent_schedule(epoch), stage4_active=False,
                         w_col_aux_val=curr_w_col_aux)
```

**검증**: `w_col_aux_schedule` 단위 테스트 — `[0, 11, 45, 80, 200]` epoch 입력 시 반환값이
정확히 `[0.5, 0.5, 중간 코사인값(≈2.75), 5.0, 5.0]`에 근접하는지 assert로 확인.
```python
assert abs(w_col_aux_schedule(0) - 0.5) < 1e-6
assert abs(w_col_aux_schedule(11) - 0.5) < 1e-6
assert abs(w_col_aux_schedule(80) - 5.0) < 1e-6
assert abs(w_col_aux_schedule(200) - 5.0) < 1e-6
```

---

## §4. 콘솔 로깅 개선

`run_training_multi()`의 epoch 출력 루프(L1495-1511, `print(f"Epoch {epoch:4d} ...")`)에
`l_col_aux`와 현재 웜업 가중치를 추가한다. 기존 포맷 문자열 끝(alpha_ent 출력 뒤)에 추가:

```python
# 변경 전 (L1505-1511 마지막 부분)
print(f"Epoch {epoch:4d} || loss={info['loss']:.4f} | MpErr={info['mp_rel_err']*100:.2f}% "
      f"| l_continuity={info['l_continuity']:.4f}(w={info['w_continuity']:.3f}) "
      f"| area={info['area']:.1f} | {' '.join(gate_strs)} "
      f"| groups={history['num_thickness_groups'][-1]} "
      f"| log_alpha_sat={sat_ratio:.2%} | dead0={int(dead0.sum())}/{int(alive_ledger.sum())} "
      f"| suspect={sum(1 for d in death_log if d['verdict']=='SUSPECT')} "
      f"| alpha_ent={alpha_ent_schedule(epoch):.4f}")

# 변경 후 — l_col_aux / w_col_aux 추가 (curr_w_col_aux는 §3.3에서 이미 계산됨)
print(f"Epoch {epoch:4d} || loss={info['loss']:.4f} | MpErr={info['mp_rel_err']*100:.2f}% "
      f"| l_continuity={info['l_continuity']:.4f}(w={info['w_continuity']:.3f}) "
      f"| area={info['area']:.1f} | {' '.join(gate_strs)} "
      f"| groups={history['num_thickness_groups'][-1]} "
      f"| log_alpha_sat={sat_ratio:.2%} | dead0={int(dead0.sum())}/{int(alive_ledger.sum())} "
      f"| suspect={sum(1 for d in death_log if d['verdict']=='SUSPECT')} "
      f"| alpha_ent={alpha_ent_schedule(epoch):.4f} "
      f"| l_col_aux={info['l_col_aux']:.4f}(w={curr_w_col_aux:.2f})")   # [v1.5 §4]
```

`info['l_col_aux']`는 `train_step_multi()`가 이미 반환하고 있으므로(L1285, v1.4 §2.3에서
반영됨) 추가 배관 작업은 필요 없다.

---

## §5. 파일 헤더 갱신

`AI_design_v1_5.py` 파일 최상단 docstring(v1.4 항목 바로 위)에 v1.5 변경 이력을 추가한다.
기존 v1.4/v1.3/v1.2 이력 블록과 동일한 포맷을 따른다:

```
AI_design_v1_5.py — Collision Penalty Explosion Fix (v1.5)
─────────────────────────────────────
[v1.5 변경, docs/command/command_v1_5.md 실행]
  §2 Huber형 충돌 페널티: compute_collision_penalty_unclamped()의 무클램프 제곱 항을
     δ=1.0mm(런타임 조정 가능) 기준 Huber loss로 교체 — 관통이 깊어져도 그래디언트가
     2δ로 상한이 걸려 폭발하지 않는다.
  §3 W_COL_AUX 코사인 웜업: epoch 11(게이트 활성화)~80(실측 loss 폭증 시점 이전) 구간에
     걸쳐 절대 floor 0.5에서 전역 상수 5.0까지 증가. §4(지연 엔트로피, epoch 400~)와는
     독립적 스케줄 — idea_v1_5.md 원안의 "엔트로피 구간과 동기화" 가정은 이번 세션에서
     실측 오류로 정정됨(alpha_ent 시작은 실제로 epoch 400).
  §4 로깅: l_col_aux, 현재 W_COL_AUX(t)를 콘솔 출력에 추가.
  구현: /synod design 세션(Gemini flash conf 98/95, OpenAI o3 conf 92/79)에서 두 모델이
  제안한 웜업 floor 공식이 실제 W_COL_AUX=5.0 상수를 반영하지 못해 5배 어긋난 오류를
  Claude가 발견·정정했고(Judge 최종 판정), 웜업 종료 시점도 실측 로그 기반으로 100→80
  으로 앞당겼다.
그 외 로직은 AI_design_v1_4.py와 동일.
─────────────────────────────────────
```

---

## §6. 절대 변경 금지 항목 (재확인)

- §4(중립 초기화+지연 엔트로피): `INIT_LOG_ALPHA_V14`, `ALPHA_ENT_START_EPOCH`,
  `ALPHA_ENT_ANNEAL_EPOCHS`, `alpha_ent_schedule()` 로직.
- §7(R1~R3 롤백): `DEATH_MP_THRESHOLD`, `DEATH_BURST_WINDOW`, `DEATH_BURST_RATIO`,
  `AREA_REGRESS_WINDOW`, R1/R2/R3 raise 조건문(L1441-1454).
- `uni_section_v18.py` 전체.

design 세션에서 "grad_clip 조정", "mixed_precision loss_scale 조정"을 §4/§7의 예외로 허용하자는
제안이 있었으나, **코드베이스에 mixed_precision 관련 로직 자체가 존재하지 않음**이 확인되어
기각되었다 — 채택하지 않는다.

---

## §7. 검증 계획

1. **단위 테스트**: §2의 Huber 연속성 assert, §3의 `w_col_aux_schedule` 경계값 assert(위 명시).
2. **Dry-run**: 기존 `run_static_dry_run()` 호출부는 §2/§3 변경과 무관하므로 그대로 유지(수정
   불필요) — 통과 여부만 재확인.
3. **스모크 테스트 (20 epoch)**: `python AI_design_v1_5.py`를 20 epoch만 돌려 콘솔 로그에서
   - `l_col_aux(w=...)` 항목이 정상 출력되는지
   - `w=` 값이 epoch 11부터 0.50 근방에서 시작해 점진적으로 증가하는지
   확인한다.
4. **회귀 테스트 (전체 실행)**: `AI_design_v1_4.py`가 크래시했던 지점(epoch 510)을 재현하되,
   - epoch 90~110 구간에서 loss가 v1.4 로그(2.33→53.59→366.9)처럼 폭증하지 않는지 확인
   - epoch 510 부근에서 `RuntimeError [R3]`가 재발하지 않는지 확인
   - **R3가 근본 수정 후에도 재발하면**(§6 불변 원칙에 따라 R3 자체는 건드리지 않고) 그 시점에야
     review_v1_4.md의 후순위 권고(§7 완화 사다리)를 별도 세션에서 검토한다 — 이번 command에는
     포함하지 않는다.

---

## 숙의 과정

<details>
<summary>Synod 세션 상세 (design 모드, 4라운드)</summary>

**Round 1 (Solver)**: Gemini(conf98)와 OpenAI(conf92) 모두 idea_v1_5.md의 "웜업이 엔트로피
구간(50~150)과 동기화"라는 가정을 코드 실측으로 독립적으로 정정(엔트로피는 실제로 epoch 400
시작)하고 웜업 구간을 11~100으로 재설계하는 데 수렴했다. Claude(Validator, conf80)는 양쪽 모두
`W_COL_AUX=5.0` 실제 상수를 스케줄 공식에 반영하지 못해 floor/최종값이 5배 어긋나는 버그를
발견했다.

**Round 2 (Critic)**: Gemini critic(conf95)이 floor 버그를 정량 검증하고 웜업 종료 시점을
80~90으로 당길 것을 추가 제안했다. OpenAI critic(conf86)은 같은 결론에 도달했으나 존재하지 않는
실험 로그("#A42", "#B07")를 인용해 신뢰성 문제가 지적됐다(Trust Score C 항목 대폭 감점,
T=0.765).

**Round 3 (Defense/Prosecution)**: Gemini(Defense, conf95)는 floor=2.5(비율) 유지를 방어했으나
"floor는 초기값일 뿐 후반부 값과 무관하다"는 구조적 오류가 있어 Judge가 기각했다. OpenAI
(Prosecution, conf79)는 존재하지 않는 `config.yml`/로그 파일을 재차 인용해(본인의 Solver 라운드
진술과도 모순) 근거 대부분이 기각됐으나, "웜업 종료 시점이 너무 늦다"는 핵심 논지는 Gemini
critic의 독립적 분석과 일치해 채택됐다.

**Judge(Claude) 최종 판정**: floor=0.5(절대값), 웜업 구간 11~80, δ는 런타임 인자화(기본 1.0mm),
§4/§7 완전 불변(예외 없음).

**신뢰 점수**: Claude 80(T2.0), Gemini 98→95(T1.91→2.0), OpenAI 92→79(T1.91→0.765, 조작 근거
감점 반영). **최종 신뢰도 90%** (Round 1 Trust-가중 평균).

</details>
