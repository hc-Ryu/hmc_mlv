# idea_v17_paratune_01.md

설계 근거: `/synod idea` 세션(Claude Judge/Validator + Gemini Architect[flash, pro는 rate limit으로 폴백] + OpenAI Explorer[gpt4o])
대상: `uni-section/code/uni_section_v17_paratune.py` (`review_v17_paratune.md`의 후속 실험)

`review_v17_paratune.md`는 신버전 v17_paratune 실행에서 4개 변수(`init_log_alpha`, `SPARSE_K`, 정적→적응형 `TAU_GATE`, `temperature` 스케줄)가 동시에 바뀐 채 관찰됐고(교란변수 문제), `TARGET_MP`도 "쉬운" 기본값 한 개만 테스트되어 적응형 `TAU_CEILING`이 실제로 발동하는 시나리오를 검증하지 못했다고 지적했다. 이 문서는 그 두 가지 갭(교란변수 분리 + `TARGET_MP` 양방향 스윕)을 **하나의 스크립트 실행(for loop)**으로 메우는 구체적 실험 설계다.

---

## 1. 왜 One-At-A-Time(OAT)인가

전체 조합(full grid)은 4개 이진/삼진 변수 × MP 스윕만으로도 수십 개 런이 되어 "하나씩 변화시켜 비교"라는 원래 요청과 배치되고 실행 시간도 감당하기 어렵다. 대신 **베이스라인(신버전 기본값) 1개를 앵커로 고정하고, 그 앵커에서 변수 하나씩만 이전 값(구버전)으로 되돌리는 OAT 구조**를 쓴다. 이렇게 하면 관찰된 차이를 정확히 그 변수 하나의 효과로 귀속시킬 수 있다 — `review_v17_paratune.md`가 요청한 ablation과 정확히 일치한다.

Synod 세션에서 Gemini(Architect)가 제안한 OAT 그리드를, OpenAI(Explorer)가 지적한 "state leakage/seed 재사용/발산" 리스크에 대한 안전장치를 더해 채택한다. OpenAI가 제안한 "전체 조합(nested loop)"은 사용자가 명시한 "하나씩 변화" 요구와 충돌하고 계산비용 문제(OpenAI 스스로도 지적)를 오히려 악화시키므로 기각한다.

---

## 2. 필수 사전 조건: `run_training`에 `init_log_alpha` 인자 추가 (1줄 수정)

`run_training` 내부에서 모델은 `init_log_alpha=2.0`이 **하드코딩된 리터럴**로 생성된다(현재 코드 1195행). 이 값은 모듈 전역 상수가 아니라 함수 내부 리터럴이므로, 다른 변수(`SPARSE_K`, `TAU_MIN`, `TAU_CEILING`, `TEMP_INIT`, `TEMP_MIN`)처럼 모듈 속성 monkeypatch로는 바꿀 수 없다. `init_log_alpha`를 ablate하려면 시그니처에 키워드 인자를 하나 추가해야 한다:

```python
def run_training(data, target_mps, target_area,
                  max_epochs=300, lr=1e-3, weights=None, curriculum=True,
                  curriculum_ratio=(0.2, 0.7), snapshot_interval=10,
                  feasibility_mp_err=0.02, feasibility_collision=0.05,
                  diagnostic=False, diagnostic_part_id=4,
                  init_log_alpha=2.0):   # <-- 추가
    ...
    model = CGDN(
        ...,
        init_log_alpha=init_log_alpha,  # <-- 리터럴 2.0 대신 인자 사용
    ).to(device)
```

이 외 4개 변수(`SPARSE_K`, `TAU_MIN`/`TAU_CEILING`, `TEMP_INIT`/`TEMP_MIN`, `TARGET_MP`)는 모두 모듈 레벨 전역 상수이거나 `__main__` 블록의 지역 변수이므로, 하니스 스크립트에서 **모듈 import 후 매 런마다 속성을 재대입(monkeypatch)**하는 것만으로 코드 수정 없이 제어 가능하다. `train_step`/`get_temperature` 등은 이 전역들을 호출 시점에 동적으로 참조하므로 재대입이 즉시 반영된다.

---

## 3. 하니스 아키텍처: 단일 프로세스 in-process 루프 (subprocess 불필요)

`uni_section_v17_paratune.py`는 `run_training(...)`이 `(history, base_coords, final_new_coords, part_labels_t, best_feasible, best_mp, pruning_state)`를 반환할 뿐, 시각화(`visualize_training`/`visualize_epoch_snapshots`) 호출은 `__main__` 블록에만 있고 `run_training` 함수 자체에는 없다. 따라서:

- **시각화를 자동으로 피할 수 있다** — 하니스가 `visualize_*`를 아예 호출하지 않으면 됨. 사용자가 요청한 "visualization 없애고 표만" 요구가 코드 수정 없이 자연스럽게 충족된다.
- **subprocess 대신 모듈을 한 번 import해 같은 프로세스에서 반복 호출**할 수 있다 — Gemini가 제안한 subprocess 방식(매 런마다 새 `py` 프로세스)보다 오버헤드가 적고, "한 번의 실행(for loop)"이라는 요청에 더 직접적으로 부합한다. 대신 OpenAI가 지적한 state leakage를 막기 위해 매 런마다:
  - `torch.manual_seed(SEED)` (+ `np.random.seed(SEED)`)로 시드 고정 재설정
  - 모델/옵티마이저/`pruning_state`/`alda_state`는 `run_training` 내부에서 매번 새로 생성되므로(1187행, 1224~1225행) 파라미터 자체의 누수는 없음 — 단, **CUDA 사용 시 GPU 메모리 파편화 방지를 위해 런 종료 후 `torch.cuda.empty_cache()`** 호출
  - 모듈 전역 상수(`SPARSE_K` 등)는 각 런 시작 직전에 명시적으로 재대입 → 이전 런의 값이 남아있지 않도록 보장

### 하니스 스크립트 골격 (`run_paratune_sweep.py`, Windows에서 `py run_paratune_sweep.py`로 실행)

```python
import copy, traceback
import numpy as np
import torch
import uni_section_v17_paratune as m   # 대상 모듈 import (경로는 code/ 디렉터리 기준)

BASELINE_TARGET_MP = 27_421_470  # N*mm

# ── OAT 그리드: 앵커(신버전 기본값) + 변수 하나씩만 되돌린 ablation + MP 양방향 스윕 ──
RUNS = [
    dict(id="anchor",        init_log_alpha=2.0, sparse_k=15.0, tau_ceiling=0.10, temp_static=None,      mp_scale=1.0),
    dict(id="abl_alpha0",    init_log_alpha=0.0, sparse_k=15.0, tau_ceiling=0.10, temp_static=None,      mp_scale=1.0),
    dict(id="abl_sparseK50", init_log_alpha=2.0, sparse_k=50.0, tau_ceiling=0.10, temp_static=None,      mp_scale=1.0),
    dict(id="abl_tauStatic", init_log_alpha=2.0, sparse_k=15.0, tau_ceiling=0.05, temp_static=None,      mp_scale=1.0),  # TAU_CEILING=TAU_MIN -> 사실상 정적 0.05
    dict(id="abl_tempStatic",init_log_alpha=2.0, sparse_k=15.0, tau_ceiling=0.10, temp_static=0.5,       mp_scale=1.0),  # 구버전과 동일한 고정 temperature=0.5
    dict(id="mp_down_0.7x",  init_log_alpha=2.0, sparse_k=15.0, tau_ceiling=0.10, temp_static=None,      mp_scale=0.7),
    dict(id="mp_down_0.5x",  init_log_alpha=2.0, sparse_k=15.0, tau_ceiling=0.10, temp_static=None,      mp_scale=0.5),
    dict(id="mp_up_1.3x",    init_log_alpha=2.0, sparse_k=15.0, tau_ceiling=0.10, temp_static=None,      mp_scale=1.3),
    dict(id="mp_up_1.6x",    init_log_alpha=2.0, sparse_k=15.0, tau_ceiling=0.10, temp_static=None,      mp_scale=1.6),
]

SEED = 42
results = []

for cfg in RUNS:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 전역 상수 monkeypatch (매 런 명시적 재대입으로 누수 방지) ──
    m.SPARSE_K = cfg["sparse_k"]
    m.TAU_MIN = 0.05
    m.TAU_CEILING = cfg["tau_ceiling"]
    if cfg["temp_static"] is not None:
        m.TEMP_INIT = cfg["temp_static"]
        m.TEMP_MIN = cfg["temp_static"]     # INIT==MIN -> 어닐링 무의미, 항상 고정값
    else:
        m.TEMP_INIT = 1.0
        m.TEMP_MIN = 0.3

    target_mp = BASELINE_TARGET_MP * cfg["mp_scale"]
    target_mps = {0: target_mp}

    row = {"id": cfg["id"], "target_mp": target_mp, "status": "OK"}
    try:
        history, base_coords, result_coords, part_labels, best_feasible, best_mp, pruning_state = m.run_training(
            data=m.data,                 # 실제 데이터 준비 로직은 기존 __main__ 블록에서 그대로 가져옴
            target_mps=target_mps,
            target_area=m.TARGET_AREA,   # 기존 __main__에서 쓰던 이름에 맞춰 조정
            init_log_alpha=cfg["init_log_alpha"],
        )

        final_pred = float(np.sum(history["pred_mp"][-1]))
        row.update({
            "final_mp_err_pct": abs(final_pred - target_mp) / target_mp * 100,
            "final_l_collision": history["l_collision"][-1],
            "final_area": history["area"][-1],
            "part4_deleted_epoch": next((i for i, s in enumerate(pruning_state.get("state_history", [])) if s.get(4) == "DELETED"), None),
            "part3_deleted_epoch": next((i for i, s in enumerate(pruning_state.get("state_history", [])) if s.get(3) == "DELETED"), None),
            "best_feasible_epoch": best_feasible.get("epoch") if best_feasible else None,
            "best_feasible_l_collision": best_feasible.get("l_collision") if best_feasible else None,
            "final_temperature": history["temperature"][-1],
            "final_tau_gate": history["current_tau_gate"][-1],
            "has_nan": bool(np.isnan(history["loss"]).any()),
        })
        if row["has_nan"]:
            row["status"] = "DIVERGED(NaN)"
    except Exception as e:
        row["status"] = f"FAILED: {type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.append(row)
    print(f"[{cfg['id']}] status={row['status']}")

# ── 요약 표만 마크다운으로 출력 (per-epoch 시각화 없음) ──
cols = ["id", "target_mp", "status", "final_mp_err_pct", "final_l_collision",
        "part4_deleted_epoch", "part3_deleted_epoch",
        "best_feasible_epoch", "best_feasible_l_collision", "final_temperature", "final_tau_gate"]

with open("paratune_sweep_summary.md", "w", encoding="utf-8") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "---|" * len(cols) + "\n")
    for r in results:
        f.write("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |\n")
```

> 실제 데이터 로딩(`m.data`)과 `TARGET_AREA` 등 기존 `__main__` 블록에서만 정의되던 변수는, 하니스에서 재사용하려면 해당 로딩 로직을 별도 함수(`load_data()`)로 뽑아내거나 하니스에서 동일하게 복붙해야 한다 — 이는 `run_training`에 `init_log_alpha`를 추가하는 것과 별개로 필요한 최소 리팩터링이다.

---

## 4. `TARGET_MP` 양방향 스윕의 안정성 안전장치

Synod Critic 라운드에서 두 모델 모두 극단적 `TARGET_MP`에서의 발산/NaN 위험을 지적했다. 안전장치:

1. **NaN/발산 감지 후 즉시 스킵, 전체 루프는 계속 진행** — 위 골격의 `try/except` + `has_nan` 체크가 이를 보장한다. 한 런이 죽어도 나머지 8개 런의 실행을 막지 않는다.
2. **낮은 `TARGET_MP` (0.7x, 0.5x)**: 목표가 작을수록 스파스 페널티가 지배적이 되어 필요한 부재까지 과도하게 가지치기할 위험(구조 붕괴/특이 강성행렬)이 있다. `l_collision`/`area`가 비정상적으로 0에 가까워지는지 별도 컬럼(`final_area`)으로 관찰하고, 필요시 후속 라운드에서 부재 두께 하한(physical floor) 도입을 검토한다 — 이번 라운드에서는 우선 "관찰"만 하고 코드 변경은 하지 않는다(교란변수 최소화 원칙 유지).
3. **높은 `TARGET_MP` (1.3x, 1.6x)**: 목표가 클수록 `l_phys`/`mp_rel_err` 관련 항의 초기 그래디언트가 커질 수 있다. 코드를 바꾸지 않는 이번 스윕에서는 `AdamW`의 기존 `lr=1e-3`을 그대로 쓰되, `has_nan` 플래그로 발산 여부만 표에 기록한다. 만약 1.6x가 실제로 발산하면, 다음 라운드에서 `1.3x`까지만 유지하거나 gradient clipping 도입을 별도로 ablate한다 — 이번 문서의 범위(교란변수 분리)를 지키기 위해 이번 스윕에는 포함하지 않는다.
4. **시드 1개(`SEED=42`) 고정** — `review_v17_paratune.md`가 지적한 "시드 1개뿐" 한계는 이번 문서에서 해소하지 않는다. 대신 표의 마지막 행에 "단일 시드 결과이므로 재현성 미검증"이라는 캡션을 남긴다(다중 시드는 별도 후속 문서로 미룸 — 9개 런 × N시드는 이번 요청의 "한 번의 실행" 범위를 벗어남).

---

## 5. 최종 산출물 표 스키마 (`paratune_sweep_summary.md`)

| id | target_mp | status | final_mp_err_pct | final_l_collision | part4_deleted_epoch | part3_deleted_epoch | best_feasible_epoch | best_feasible_l_collision | final_temperature | final_tau_gate |
|---|---|---|---|---|---|---|---|---|---|---|
| anchor | 27,421,470 | OK | ... | ... | ... | ... | ... | ... | ... | ... |
| abl_alpha0 | 27,421,470 | OK | ... | ... | ... | ... | ... | ... | ... | ... |
| abl_sparseK50 | 27,421,470 | OK | ... | ... | ... | ... | ... | ... | ... | ... |
| abl_tauStatic | 27,421,470 | OK | ... | ... | ... | ... | ... | ... | ... | ... |
| abl_tempStatic | 27,421,470 | OK | ... | ... | ... | ... | ... | ... | ... | ... |
| mp_down_0.7x | 19,195,029 | OK/FAILED | ... | ... | ... | ... | ... | ... | ... | ... |
| mp_down_0.5x | 13,710,735 | OK/FAILED | ... | ... | ... | ... | ... | ... | ... | ... |
| mp_up_1.3x | 35,647,911 | OK/FAILED | ... | ... | ... | ... | ... | ... | ... | ... |
| mp_up_1.6x | 43,874,352 | OK/FAILED | ... | ... | ... | ... | ... | ... | ... | ... |

이 표 하나만으로 `anchor` 대비 각 ablation 행의 차이를 직접 대조하면, `review_v17_paratune.md`가 판정 보류했던 "프루닝 지연이 정말 `init_log_alpha` 하나 때문인가"를 실측으로 확인할 수 있고, `mp_down_*`/`mp_up_*` 행들로 적응형 `TAU_CEILING`이 처음으로 하한(0.05) 밖에서 실제로 작동하는지도 관찰 가능하다.

---

<details>
<summary>숙의 과정 (Synod 세션 상세)</summary>

### 모델 기여
- **Claude (Judge/Validator):** OAT 구조가 사용자의 명시적 "하나씩 변화" 요구 및 `review_v17_paratune.md`의 ablation 요청과 정확히 부합한다고 판정. `init_log_alpha`가 `run_training` 내부 하드코딩 리터럴이라 모듈 monkeypatch로 제어 불가능함을 코드 확인으로 발견 — 함수 시그니처에 인자 추가가 필요한 유일한 변수로 특정. subprocess보다 in-process 루프가 "한 번의 실행" 요구에 더 부합한다고 판단해 채택.
- **Gemini (Architect, flash 폴백):** 7-9런 OAT 그리드와 앵커 베이스라인 설계, MP 스윕 방향별 안전장치(물리적 하한/그래디언트 클리핑) 아이디어 제시. Critic 라운드에서 OpenAI의 전체 조합 제안이 사용자 요구("하나씩")와 모순됨을 지적 — 채택.
- **OpenAI (Explorer, gpt4o):** state leakage, 시드 재사용, 극단 `TARGET_MP`에서의 발산/NaN, 전체 그리드의 계산비용 리스크를 지적 — 이 중 state leakage/NaN 감지 안전장치는 채택했으나, 전체 조합(nested loop) 제안은 사용자 요구 및 자신이 지적한 계산비용 문제와 충돌해 기각.

### 해결된 주요 쟁점
1. "OAT vs 전체 조합?" → OAT 채택 (사용자 명시 요구 + 계산비용)
2. "subprocess vs in-process 루프?" → in-process 채택 (시각화 함수가 `run_training`과 분리되어 있어 자동으로 시각화 회피 가능, "한 번의 실행" 요구에 더 직접 부합). 대신 CUDA 캐시 정리 등 상태 누수 방지 안전장치 추가.
3. "모든 변수를 monkeypatch로 제어 가능한가?" → 아니오, `init_log_alpha`만 함수 시그니처 수정 필요 — 코드 검증(1195행)으로 확인.

### 신뢰 점수 (최종 라운드 기준)
- Claude: 88
- Gemini: 90 (일부 제안은 Judge가 in-process 방식으로 대체)
- OpenAI: 82 (안전장치 지적은 채택, 전체 조합 제안은 기각)

</details>

### 신뢰도: 85%
(OAT 그리드/monkeypatch 하니스 설계는 코드 검증(1195행 `init_log_alpha` 하드코딩 확인 등)을 거쳤으나, 실제 데이터 로딩부(`m.data`, `TARGET_AREA`)를 `__main__`에서 분리하는 소규모 리팩터링과 9개 런의 실제 실행 결과 검증은 아직 이루어지지 않았다. 극단적 `TARGET_MP`(1.6x, 0.5x)에서 실제로 발산하는지는 표가 채워지기 전까지 미확정이다.)
