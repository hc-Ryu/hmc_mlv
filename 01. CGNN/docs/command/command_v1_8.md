# command_v1_8.md — Order 손실 희석 버그(v19) + LSE 온도 재보정(v1.8) 지시서

작성일: 2026-08-09 (`/synod design` 세션, Gemini flash conf95 + OpenAI gpt4o conf85)
참조: `docs/idea/idea_v1_8.md`(설계 근거, 신뢰도 90%), `docs/review/review_v1_7.md`(실행 결과 리뷰),
`AI_design_v1_7.py`(기반 코드), `uni_section_v18.py`(**이번 세션부터 수정 허용 — v19로 갱신**)

> **앵커 규칙**: 본 문서의 위치 지정은 **함수/변수 이름 기준**이다. 라인 번호는 참고용이며
> 코드 변경으로 어긋날 수 있다.

> **범위 확장 승인**: 사용자가 이번 세션에서 `uni_section_v18.py` 수정을 명시적으로 허용했다.
> v1.4~v1.7 내내 "불변"이었던 이 파일을 `uni_section_v19.py`로 복사해 딱 한 함수
> (`compute_mesh_order_loss`)만 고친다 — 그 외 모든 로직은 그대로 유지한다.

> **불변 규칙(계속 유지)**: §4(중립 초기화+지연 엔트로피), §7(R1~R3 롤백)의 상수·로직은
> 여전히 절대 수정하지 않는다. `uni_section_v19.py` 내부에서도 `compute_mesh_order_loss()`
> 외의 다른 함수는 전혀 건드리지 않는다.

> **범위 제한(idea_v1_8.md §3, Judge 판정)**: 이번 버전은 §1(Order Top-K + w_order 재조정)과
> §2(LSE 온도 재보정) **두 가지 보정만** 적용한다. 손실 통합, 하드 기하 제약 등은 변수 분리
> 원칙에 따라 이번 범위에서 명시적으로 제외한다(review_v1_7.md가 이미 겪은 "개선 원인을
> 분리할 수 없었던" 문제를 반복하지 않기 위함).

---

## §0. 목표 및 산출물

`uni_section_v18.py`를 복사해 **`uni_section_v19.py`**를, `AI_design_v1_7.py`를 복사해
**`AI_design_v1_8.py`**를 만든다.

| 결함 | 근본 원인 | 대응 |
|---|---|---|
| **Order 손실이 4개 버전 내내 무한정 증가**(review_v1_5~v1_7.md 공통 관찰) | `compute_mesh_order_loss()`(불변 파일 내부, 이번에 처음 확인)가 `torch.mean()`으로 ~2992개 물리 엣지에 걸쳐 평균 — 소수의 심하게 접힌 엣지가 희석됨 | §1(Top-K 평균 풀링, `uni_section_v19.py`) + `w_order` 동시 재조정(`AI_design_v1_8.py`) |
| v1.7 smoothness LSE 온도(T=0.01) 미보정으로 손실 스케일 폭증(review_v1_7.md) | `torch.logsumexp(sq_dev/T)`가 T=0.01에서 사실상 hard max로 수렴, mm² 원시값을 그대로 반환 | §2(물리 단위 온도 T=1.5mm로 재보정) |

**적용 순서**: §1(파일 분리)을 먼저 하고, §1.1(풀링)과 §1.2(가중치)는 **반드시 같은 커밋**에서
함께 적용한다(둘 중 하나만 적용 시 v1.7과 동일한 실패 재현 위험 — idea_v1_8.md §1 참고).
§2는 §1과 독립적이므로 순서 무관하나 함께 재실행해 검증한다.

---

## §1. `compute_mesh_order_loss()` Top-K 평균 풀링 (uni_section_v19.py)

### §1.0 파일 생성

```bash
cp uni-section/code/uni_section_v18.py uni-section/code/uni_section_v19.py
```

`uni_section_v19.py` 최상단(§6에서 추가할 이력 블록 바로 아래)에 다음 경고 주석을
추가한다(구현 세션에서 OpenAI 지적 — 두 개의 거의 동일한 대형 파일이 나란히 존재하면
향후 한쪽만 수정되는 드리프트 위험이 있음):

```python
# ⚠ uni_section_v19.py는 uni_section_v18.py의 완전한 사본이며 compute_mesh_order_loss()
# 단 하나만 다르다. uni_section_v18.py를 별도로 수정하지 말 것 — 두 파일이 갈라지면
# 어느 쪽이 실제로 실행되는지(AI_design_v1_8.py는 v19를 import) 혼동을 일으킨다.

### §1.1 `compute_mesh_order_loss()` 수정

**대상**: `uni_section_v19.py`의 `compute_mesh_order_loss()`(원본 L521-533).

```python
# 변경 전
def compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr, eps=0.5):
    """[v10~v17_01 §3 유지] 2D Mesh Order Loss."""
    src, dst = edge_index
    mask = (src < dst) & (edge_attr[:, 3] == 0.0)
    if not mask.any():
        return torch.tensor(0.0, device=new_coords.device)

    e0 = base_coords[dst[mask]] - base_coords[src[mask]]
    e0_hat = e0 / (e0.norm(dim=1, keepdim=True) + 1e-8)
    e_new = new_coords[dst[mask]] - new_coords[src[mask]]
    proj = (e_new * e0_hat).sum(dim=1)
    violation = torch.relu(eps - proj)
    return torch.mean(violation ** 2)

# 변경 후
def compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr, eps=0.5,
                             top_k_fraction=0.02):
    """[v19 §1] 2D Mesh Order Loss — Top-K 평균 풀링으로 교체.
    [정정 이력] v1.4~v1.7 내내 이 함수는 mean()으로 전체 물리 엣지(~2992개)에 걸쳐
    평균을 냈다 — compute_smoothness_loss_angle()에서 이미 발견·수정했던 것과 동일한
    희석 버그가 "Order 손실"이라는 이름으로 4개 버전 내내 계속 관찰되고도 원인 불명이었던
    것은 이 함수가 그동안 불변 파일 안에 있었기 때문이다(design 세션에서 최초 확인).
    LogSumExp가 아니라 Top-K를 쓰는 이유: v1.7이 온도 미보정(T=0.01)으로 겪은
    exp(deviation/T) 오버플로우 위험이 Top-K에는 구조적으로 없다(지수 연산 자체가 없음) —
    design 세션에서 수학적으로 검증(exp(10/0.01)=exp(1000)로 float32 확정적 오버플로우)."""
    src, dst = edge_index
    mask = (src < dst) & (edge_attr[:, 3] == 0.0)
    if not mask.any():
        return torch.tensor(0.0, device=new_coords.device)

    e0 = base_coords[dst[mask]] - base_coords[src[mask]]
    e0_hat = e0 / (e0.norm(dim=1, keepdim=True) + 1e-8)
    e_new = new_coords[dst[mask]] - new_coords[src[mask]]
    proj = (e_new * e0_hat).sum(dim=1)
    violation = torch.relu(eps - proj) ** 2
    violation = violation.view(-1)   # [v19, 구현 세션 방어 코드] 항상 1D이지만 안전을 위해 명시
    k = max(1, int(violation.numel() * top_k_fraction))   # 전체 ~2992개 중 약 2%(~60개)
    top_k_violations, _ = torch.topk(violation, k=k, largest=True)
    return torch.mean(top_k_violations)
```

**k 경계값 검증(구현 세션에서 두 모델 공통 확인)**: `k = max(1, int(N*0.02))`는 N≥1인
모든 경우에 `1 <= k <= N`을 보장한다(N<50이면 k=1, N≥50이면 k=⌊0.02N⌋≥1이고 항상 N보다
작음) — `N=0`은 이미 `if not mask.any(): return ...` 가드가 topk 호출 전에 처리한다.
제곱(`**2`) 연산은 `relu` 출력이 항상 0 이상이므로 top-k 선택 순서에 영향을 주지 않는다
(단조증가 함수이므로 제곱 전/후 순위가 동일 — 구현 세션에서 수식으로 확인).

**주의**: 이 함수는 `uni_section_v19.py`의 다른 어떤 함수에서도 호출되지 않는다(자체
`run_training()`은 `AI_design_v1_8.py`가 사용하지 않음) — 시그니처 변경(`top_k_fraction`
키워드 인자 추가, 기본값 있음)은 하위 호환이며 다른 호출부에 영향 없다.

**기각된 우려(OpenAI 제기, 검증 결과 이 프로젝트에 해당 없음)**: "체크포인트를 `pickle`로
불러올 때 `sys.modules['uni_section_v18']`이 채워지지 않아 실패할 수 있다"는 우려가
제기됐으나, 이 코드베이스는 `torch.save(model.state_dict())`로 **텐서 값만** 저장하고
모듈 클래스 참조를 피클링하지 않으며, 체크포인트 재개(resume) 메커니즘 자체가 없음이
이미 이전 세션(v1.6 구현)에서 확인됐다 — 이 우려는 이 프로젝트의 실제 저장 방식과
무관하므로 기각한다.

### §1.2 `w_order` 재조정 (AI_design_v1_8.py, §1.1과 반드시 동시 적용)

**대상**: `AI_design_v1_8.py`의 `run_training_multi()` 내부 기본 `weights` 딕셔너리
(`AI_design_v1_7.py` L1544 부근, 유일한 정의처).

```python
# 변경 전
weights = {'w_phys': 10.0, 'w_order': 1.0, 'w_smooth': 0.5, 'w_anchor': 0.02, 'w_sat': 0.01,
           'w_sparse': W_SPARSE_V13,
           'w_mass':   w_mass,
           'w_area_floor': w_area_floor}

# 변경 후
weights = {'w_phys': 10.0, 'w_order': W_ORDER_V18, 'w_smooth': 0.5, 'w_anchor': 0.02, 'w_sat': 0.01,
           'w_sparse': W_SPARSE_V13,
           'w_mass':   w_mass,
           'w_area_floor': w_area_floor}
```

신규 상수 블록(§3 참고)에 `W_ORDER_V18 = 0.02`를 **시작값**으로 정의한다 — `top_k_fraction`
(0.02)과 동일하게 맞춰, Top-K 전환으로 인한 분모 축소(2992→60, 약 50배)를 대략 상쇄해
원래 `w_order * l_order`의 그래디언트 크기 균형을 유지한다(idea_v1_8.md §1, Critic
라운드에서 Gemini가 자기 수정으로 도출한 필수 요구사항 — **§1.1만 적용하고 이 조정을
빠뜨리면 v1.7과 동일한 "다른 손실 항을 압도하는" 실패가 재현된다**).

**⚠ 구현 세션 정정(OpenAI 지적, 수학적으로 검증됨)**: `K/N=0.02`를 그대로 가중치에 곱하는
것은 **정확한 보정이 아니라 근사치**다 — Top-K 평균(최악 60개만의 평균)과 "전체 평균 ×
50"은 통계적으로 다르다. 위반이 있는 엣지 수가 정확히 60개에 가까우면 근사가 맞아떨어지지만,
(a) 정상 메쉬처럼 위반 엣지가 60개보다 훨씬 적으면 이 보정값은 **과소평가**되고, (b) 지금까지
관찰된 것처럼 위반이 광범위(60개보다 훨씬 많음)하면 **과소평가**된다(두 경우 모두 하향
편향 — 자세한 유도는 idea_v1_8.md 참고할 필요 없이, `violation=relu(eps-proj)`가 만족
엣지에서 정확히 0이 되는 구조상 top-K 평균이 항상 전체 평균보다 크거나 같기 때문). 따라서
`W_ORDER_V18=0.02`는 **§7 검증 계획에서 실측 `l_order`·`contrib_order` 값을 확인 후 반드시
재보정**해야 하는 시작값으로 취급한다 — "정확히 상쇄되는 값"이라는 1차 설계 문서의 표현은
근사적 직관이었음을 명시한다.

**⚠ 반드시 확인**: `train_step_multi()` 내부의 `contrib_order = weights['w_order'] * l_order`
(L1471 부근)는 코드 변경이 필요 없다 — `weights` 딕셔너리 값만 바뀌면 자동 반영된다.

---

## §2. v1.7 smoothness LSE 온도 물리 단위 재보정 (AI_design_v1_8.py)

### §2.1 상수 수정

**대상**: `AI_design_v1_7.py`의 `SMOOTH_LSE_TEMPERATURE` 상수(§1.1 신규 상수 블록 내부).

```python
# 변경 전
SMOOTH_LSE_TEMPERATURE = 0.01   # LogSumExp 온도. §6 검증 계획에서 실측 편차 스케일 기준 재보정 필수

# 변경 후
SMOOTH_LSE_TEMPERATURE = 1.5   # [v1.8] mm 단위. 판금 제조 공차 스케일에 맞춘 물리적 재보정.
                                 # T=0.01은 exp(deviation²/T)가 10mm급 편차에서 exp(1000) 수준의
                                 # float32 오버플로우를 사실상 확정적으로 유발함(design 세션 검증,
                                 # review_v1_7.md의 "near-hard-max" 진단을 수식으로 재확인) —
                                 # 물리적으로 의미 있는 스케일(1.5mm)로 교체해 정상 수치 범위에서
                                 # "부드러운 최댓값"으로 동작하도록 한다.
```

`compute_smoothness_lse()` 함수 자체는 수정 불필요 — 기본값 참조 상수만 바뀐다.

---

## §3. 신규 상수 블록 (AI_design_v1_8.py)

기존 v1.7 상수 블록(`W_SMOOTH_LSE`, `SMOOTH_LSE_TEMPERATURE`) 옆에 추가.

```python
# ══════════════════════════════════════════════════════════════════
# [v1.8] 로컬 오버라이드 상수 — Order 손실 Top-K 풀링 동반 가중치 재조정
# 근거: docs/idea/idea_v1_8.md, docs/review/review_v1_7.md, docs/command/command_v1_8.md
# ══════════════════════════════════════════════════════════════════
W_ORDER_V18 = 0.02   # uni_section_v19.compute_mesh_order_loss()의 top_k_fraction(0.02)과 동일값.
                      # Top-K 전환으로 분모가 2992→60(~50배 축소)되므로, 원래 w_order=1.0의
                      # 그래디언트 크기를 유지하려면 이 배율만큼 가중치를 낮춰야 한다
                      # (idea_v1_8.md §1, Critic 라운드에서 Gemini가 자기 수정으로 도출).
```

---

## §4. 임포트 경로 갱신 (AI_design_v1_7.py → v1.8)

**대상**: 모듈 상단 임포트문(L178 부근). `usv18`이라는 별칭(alias)은 파일 전체에서
54회 참조되므로, **별칭은 그대로 유지**하고 임포트 대상 모듈명만 바꾼다(불필요한 전체
치환으로 인한 오류 위험 최소화).

```python
# 변경 전
import uni_section_v18 as usv18

# 변경 후
import uni_section_v19 as usv18   # [v1.8] compute_mesh_order_loss() Top-K 풀링 적용판
```

파일 하단 docstring 등에서 `uni_section_v18.py`를 문자열로 언급하는 주석들은 **과거 이력
기록이므로 수정하지 않는다**(v1.5→v1.6 전환 때도 동일 원칙 적용됨).

---

## §5. 절대 변경 금지 항목 (재확인)

- §4(중립 초기화+지연 엔트로피): `INIT_LOG_ALPHA_V14`, `ALPHA_ENT_START_EPOCH`,
  `ALPHA_ENT_ANNEAL_EPOCHS`, `alpha_ent_schedule()` 로직.
- §7(R1~R3 롤백): `DEATH_MP_THRESHOLD`, `DEATH_BURST_RATIO`, R1/R2/R3 raise 조건문.
- `uni_section_v19.py` 내부에서 `compute_mesh_order_loss()` **이외의 모든 함수**(
  `compute_smoothness_loss_angle()`, `compute_gates()`, `compute_gate_temperature()`,
  `build_collision_spec()`, `_signed_projections()` 등)는 원본 `uni_section_v18.py`와
  **1바이트도 다르지 않아야 한다.**
- 손실 항 통합(`l_order`/`l_smooth`/`l_smooth_lse` 병합)이나 하드 기하 제약(좌표 클램핑/
  투영) 도입은 이번 버전 범위 밖(idea_v1_8.md §3, 후속 버전으로 명시적 보류).
- v1.7의 §1(정적 이웃 맵)·§2(연속성 단순화)는 이번 세션과 무관하므로 그대로 유지.

---

## §6. 파일 헤더 갱신

**`uni_section_v19.py`** 최상단에 변경 이력 추가(원본에 이런 블록이 없다면 파일 최상단에
신규 삽입):

```
[v19 변경, docs/command/command_v1_8.md 실행]
  compute_mesh_order_loss(): torch.mean() → Top-K(top_k_fraction=0.02) 평균 풀링.
  v1.4~v1.7 내내 이 함수의 mean 희석이 "Order 손실 무한정 증가"의 실제 원인이었으나
  불변 파일 내부에 있어 발견되지 못했다(AI_design_v1_8.py /synod idea 세션에서 최초 확인).
  그 외 모든 함수는 uni_section_v18.py와 동일.
```

**`AI_design_v1_8.py`** 최상단 docstring(v1.7 항목 바로 위)에 추가:

```
AI_design_v1_8.py — Order Loss Dilution Fix + LSE Temperature Recalibration (v1.8)
─────────────────────────────────────
[v1.8 변경, docs/command/command_v1_8.md 실행]
  §1 Order 손실 Top-K 풀링(uni_section_v19.py) + w_order 동반 재조정(1.0→0.02): 4개 버전
     내내 원인 불명이던 Order 손실 무한 증가의 실제 원인(불변 파일 내부 mean 희석)을
     최초로 발견·수정. Top-K 분모 축소(~50배)를 상쇄하도록 가중치도 반드시 함께 조정.
  §2 smoothness LSE 온도 물리 단위 재보정(0.01→1.5mm): v1.7의 float32 오버플로우 위험
     경로를 원천 차단.
  구현: /synod idea 세션(Gemini flash conf95→90, OpenAI gpt4o conf75→85)에서 Gemini가
  Critic 라운드에서 자신의 1라운드 제안("w_order 재조정은 후순위 안전장치")을
  "Top-K와 반드시 동시 적용해야 하는 필수 선행 조건"으로 스스로 정정했다(N/K≈50배
  스케일 변화를 수식으로 도출). OpenAI가 제안한 손실 통합은 review_v1_7.md의 원인 분리
  실패 전례를 근거로 변수 분리 원칙에 따라 기각, 후속 버전으로 보류. 구현 세션(/synod
  design 재검증)에서 OpenAI가 "K/N=0.02는 정확한 보정이 아니라 근사치"임을 통계적으로
  재검증해 W_ORDER_V18을 "시작값, 스모크 테스트 후 재보정 필수"로 격하했고, Gemini가
  지적한 k 경계값·제곱 순서 문제는 코드로 이미 안전함을 확인했다. OpenAI의 pickle/
  sys.modules 우려는 이 프로젝트가 state_dict 텐서만 저장하고 체크포인트 재개가
  없어(v1.6 구현 세션에서 이미 확인) 무관하다고 판단해 기각했다.
그 외 로직은 AI_design_v1_7.py와 동일. uni_section_v19.py는 compute_mesh_order_loss()
1개 함수만 수정, 나머지는 uni_section_v18.py와 동일.
─────────────────────────────────────
```

---

## §7. 검증 계획

1. **단위 테스트**: `compute_mesh_order_loss()`에 인위적으로 소수 엣지만 심하게 위반하는
   `new_coords`를 넣어, Top-K 버전이 mean 버전보다 유의미하게 큰 손실을 반환하는지 확인
   (희석 해소 검증). `top_k_fraction`을 1.0으로 주면 기존 mean 동작과 근사적으로 일치하는지도
   확인(회귀 안전성).
2. **스모크 테스트(20 epoch)**: 콘솔 로그에서 `l_order`, `l_smooth_lse`, `w_continuity`가
   정상 출력되는지 확인. `import uni_section_v19 as usv18` 변경 후 임포트 에러 없는지 확인.
3. **회귀 테스트(전체 학습)**: `reports/v1_8/`에서
   - `l_order`가 v1.4~v1.7처럼 무한정 증가하지 않고 수렴/안정화되는지
   - 전체 손실(Total Loss) 스케일이 v1.6 수준(2~10대)으로 복귀하는지(v1.7의 65~90대
     비정상 고착이 해소됐는지)
   - Mp 목표 미달 섹션이 v1.7보다 늘지 않았는지(특히 10~13번 섹션)
   - 최종 3D 형상에서 잔존 스파이크가 추가로 개선됐는지 육안 확인
4. **`eps=0.5` 별도 확인(idea_v1_8.md 관찰 항목)**: 학습 초반 `proj` 값의 실제 분포를
   로그로 찍어, 엣지 길이 스케일 대비 임계값이 합리적인지 확인 — 필요시 별도 command
   문서로 재조정.
5. **§1.1·§1.2 동시 적용 확인**: 만약 재실행 결과가 기대와 다르면, `W_ORDER_V18`을
   `top_k_fraction`과 독립적으로 튜닝하기 전에 **두 값을 항상 같은 비율로 유지**한 채
   조정할 것(예: `top_k_fraction=0.01`로 낮추면 `W_ORDER_V18`도 비례해 낮춤).
6. **`W_ORDER_V18` 재보정(구현 세션 필수 추가 항목)**: `K/N` 근사가 부정확할 수 있음이
   확인됐으므로(위 §1.2 경고 참고), 스모크 테스트(20 epoch) 로그에서 `contrib_order`
   값을 다른 손실 항(`contrib_phys`, `L_alda_effective` 등)과 비교해, 한쪽이 압도적으로
   크면(10배 이상 차이) `W_ORDER_V18`을 그 비율만큼 조정한다 — 이 재조정은 회귀 테스트
   착수 전에 완료할 것.
