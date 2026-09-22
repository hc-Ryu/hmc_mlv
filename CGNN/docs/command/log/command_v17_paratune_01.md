# command_v17_paratune_01.md

설계 근거: `/synod idea` 세션(Claude Judge/Validator + Gemini Architect[flash, pro는 rate limit 폴백] + OpenAI Explorer[gpt4o])
대상: `idea_v17_paratune_01.md`를 실제로 구현/실행하기 위한 단계별 command 문서

Solver 라운드에서 Gemini가 제안한 하니스 코드는 `run_training`의 시그니처(`node_features, edge_index, edge_features, boundary_indices, ...`)와 상수 값(`SPARSE_K=100.0` 등), `history` 딕셔너리 키(`mp_err_pct`, `tau_gate`)를 **실제 코드와 다르게** 추정했다. Critic 라운드에서 이 오류들을 실측 코드 검증(직접 `Read`)으로 전부 특정해 기각했고, 이 문서는 검증된 실제 시그니처/상수/키 이름만 사용해 다시 작성한다.

---

## Step 0: 검증된 사실 (Ground Truth — 반드시 이 값들만 사용)

- `run_training`의 실제 시그니처(1158행):
  ```python
  def run_training(data, target_mps, target_area,
                    max_epochs=300, lr=1e-3, weights=None, curriculum=True,
                    curriculum_ratio=(0.2, 0.7), snapshot_interval=10,
                    feasibility_mp_err=0.02, feasibility_collision=0.05,
                    diagnostic=False, diagnostic_part_id=4)
  ```
  → `data`(PyG 스타일, `.x`/`.edge_index`), `target_mps`(dict, 예 `{0: TARGET_MP}`), `target_area`(None이면 초기 단면적 자동 스냅샷) 3개 위치 인자 + 키워드 인자들. 별도의 `node_features`/`edge_index`/`edge_features`/`boundary_indices` 인자는 존재하지 않는다.
- 반환값(7-tuple, 1493행): `(history, base_coords, final_new_coords, part_labels_t, best_feasible, best_mp, pruning_state)`
- `history` dict의 실제 키(1329행 및 주변): `loss`, `pred_mp`, `l_collision`, `l_order`, `l_smooth`, `l_mass`, `l_sparse`, `l_anchor`, `l_sat`, `area`, `mp_rel_err`, `t_per_part`, `temperature`, `current_tau_gate`(⚠ `tau_gate` 아님), `alda_mu_Mp`, `alda_mu_col`, `alda_rho_Mp`, `alda_rho_col`, `z_gate`, `snapshots`. `mp_err_pct`라는 키는 없다 — 오차율은 `mp_rel_err`이며 이미 상대오차(0~1)로 계산되어 있다(×100 하면 %).
- `pruning_state['state']`는 파트별 최종 상태 리스트(`'ALIVE'|'PENDING_DELETE'|'DELETED'`)만 담고 있고, "DELETED로 확정된 정확한 epoch"를 직접 담는 필드는 없다. 다만 `history['snapshots']`가 `snapshot_interval=10` epoch마다 `pruning_state['state']`의 스냅샷을 포함하므로(1422/1437행), DELETED epoch는 **±10 epoch 해상도**로만 요약 표에서 근사 가능하다. 정확한 epoch(예: review 문서의 326, 681 등)는 원 스크립트가 `print(f"[Pruning] epoch {epoch}: part {pid} DELETED ...")`(1388행)로 콘솔에만 출력하므로, 하니스에서 stdout을 캡처하지 않는 한 요약 표에는 스냅샷 해상도로만 기록한다(이번 라운드는 소스 수정 최소화를 위해 stdout 캡처를 도입하지 않음).
- 실제 데이터 로딩은 `__main__` 블록(1744행)의 `data, node_registry = build_bpillar_section()` 한 줄과, 그 아래 초기 형상 로그 출력, `TARGET_MP = 27_421_470`, `target_mps = {0: TARGET_MP}`, `weights` dict 정의로 구성된다.
- 실제 베이스라인 모듈 상수: `SPARSE_K = 15.0`, `TAU_MIN = 0.05`, `TAU_CEILING`(적응형 상한, 코드상 0.10 부근), `TEMP_INIT = 1.0`, `TEMP_MIN = 0.3`, `TEMP_WARMUP_EPOCHS = 30`.
- `run_training` 호출 시 `weights['w_sparse']`는 모듈 전역 `W_SPARSE`를 참조한다 — 이 값도 `SPARSE_K`와 함께 `__main__`에서 그대로 가져와야 한다.

---

## Step 1: 소스 수정 (최소 침습, 2곳)

### 1.1 `init_log_alpha`를 `run_training` 인자로 노출

`run_training` 시그니처에 키워드 인자 추가:

```diff
 def run_training(data, target_mps, target_area,
                   max_epochs=300, lr=1e-3, weights=None, curriculum=True,
                   curriculum_ratio=(0.2, 0.7), snapshot_interval=10,
                   feasibility_mp_err=0.02, feasibility_collision=0.05,
-                  diagnostic=False, diagnostic_part_id=4):
+                  diagnostic=False, diagnostic_part_id=4,
+                  init_log_alpha=2.0):
```

모델 생성부(1187~1196행) 수정:

```diff
     model = CGDN(
         in_channels=8,
         hidden_channels=128,
         num_layers=4,
         heads=4,
         edge_dim=4,
         max_displacement=50.0,
         num_parts=5,
-        init_log_alpha=2.0,   # [command_v17_paratune.md Phase 0] 0.0 오버라이드 제거, 생성자 기본값과 일치
+        init_log_alpha=init_log_alpha,  # [command_v17_paratune_01.md] 하니스에서 ablation 가능하도록 노출
     ).to(device)
```

기본값이 `2.0`으로 그대로이므로, `__main__`에서 인자 없이 호출하는 기존 호출부(1779행)는 **동작 변경 없음**.

### 1.2 데이터 로딩을 `load_data()`로 추출

`__main__` 블록 최상단, `build_bpillar_section()` 호출 및 초기 형상 로그를 함수로 추출한다. 기존 로그 출력(단면적/Mp 등)은 표준 실행 시 사용자가 보던 정보이므로 함수 안에 그대로 유지해 `__main__`에서 호출 시 동일 출력이 나오게 한다.

```python
def load_data():
    """build_bpillar_section() 로 데이터 생성 + 초기 형상 로그 출력. __main__과 하니스가 공용으로 사용."""
    data, node_registry = build_bpillar_section()
    print(f"\n데이터: nodes={data.x.shape} | edges={data.edge_index.shape}")

    _init_coords = data.x[:, :2]
    _init_t = data.x[:, 6:7]
    _init_fy = data.x[:, 7:8]
    _init_part_ids = data.x[:, 4].long()
    _init_area_total, _init_area_per_part = compute_section_area(
        _init_coords, _init_t, data.edge_index, part_ids=_init_part_ids)
    _init_mp_total, _init_y_pna = compute_edge_mp_pna(
        _init_coords, _init_t, _init_fy, data.edge_index)
    print(f"\n{'=' * 78}")
    print("[초기 형상 정보] (학습 시작 전 baseline)")
    print(f"  단면적(질량 proxy) : {_init_area_total:>10.1f} mm²")
    for _pid in sorted(_init_area_per_part.keys()):
        print(f"    part{_pid}: {_init_area_per_part[_pid]:>8.1f} mm²")
    print(f"  Mp(초기 형상)      : {_init_mp_total.item():>14,.0f} N·mm")
    print(f"  y_pna(초기 형상)   : {_init_y_pna.item():>10.2f} mm")
    print(f"{'=' * 78}")

    return data, node_registry
```

`__main__` 블록은 다음과 같이 이 함수를 호출하도록 교체(그 아래 `TARGET_MP`/`target_mps`/`weights`/`run_training(...)` 호출부는 **변경 없음**):

```diff
-    data, node_registry = build_bpillar_section()
-    print(f"\n데이터: nodes={data.x.shape} | edges={data.edge_index.shape}")
-    ... (초기 형상 로그 블록 전체) ...
+    data, node_registry = load_data()
```

이 리팩터링은 순수 코드 이동이며 로직 변경이 없으므로, `py uni_section_v17_paratune.py` 단독 실행 결과는 patch 전후로 100% 동일해야 한다(Step 4 검증 기준 참고).

---

## Step 2: 하니스 스크립트 (`uni-section/code/paratune_harness_v1.py`)

대상 스크립트와 같은 디렉터리에 생성. 모듈을 한 번만 import하고, 9개 런을 for-loop로 순회하며 모듈 전역을 재대입(monkeypatch)한다.

```python
import gc
import numpy as np
import torch
import uni_section_v17_paratune as m

BASELINE_TARGET_MP = 27_421_470  # N*mm, m.py __main__과 동일
BASELINE = dict(sparse_k=15.0, tau_min=0.05, tau_ceiling=m.TAU_CEILING,
                 temp_init=1.0, temp_min=0.3)

RUNS = [
    dict(id="anchor",         init_log_alpha=2.0, sparse_k=BASELINE["sparse_k"], tau_ceiling=BASELINE["tau_ceiling"], temp_static=None, mp_scale=1.0),
    dict(id="abl_alpha0",     init_log_alpha=0.0, sparse_k=BASELINE["sparse_k"], tau_ceiling=BASELINE["tau_ceiling"], temp_static=None, mp_scale=1.0),
    dict(id="abl_sparseK50",  init_log_alpha=2.0, sparse_k=50.0,                 tau_ceiling=BASELINE["tau_ceiling"], temp_static=None, mp_scale=1.0),
    dict(id="abl_tauStatic",  init_log_alpha=2.0, sparse_k=BASELINE["sparse_k"], tau_ceiling=0.05,                    temp_static=None, mp_scale=1.0),  # ceiling=min -> 사실상 정적 0.05
    dict(id="abl_tempStatic", init_log_alpha=2.0, sparse_k=BASELINE["sparse_k"], tau_ceiling=BASELINE["tau_ceiling"], temp_static=0.5,  mp_scale=1.0),  # 구버전과 동일한 고정 0.5
    dict(id="mp_down_0.7x",   init_log_alpha=2.0, sparse_k=BASELINE["sparse_k"], tau_ceiling=BASELINE["tau_ceiling"], temp_static=None, mp_scale=0.7),
    dict(id="mp_down_0.5x",   init_log_alpha=2.0, sparse_k=BASELINE["sparse_k"], tau_ceiling=BASELINE["tau_ceiling"], temp_static=None, mp_scale=0.5),
    dict(id="mp_up_1.3x",     init_log_alpha=2.0, sparse_k=BASELINE["sparse_k"], tau_ceiling=BASELINE["tau_ceiling"], temp_static=None, mp_scale=1.3),
    dict(id="mp_up_1.6x",     init_log_alpha=2.0, sparse_k=BASELINE["sparse_k"], tau_ceiling=BASELINE["tau_ceiling"], temp_static=None, mp_scale=1.6),
]

SEED = 42


def reset_globals():
    """매 런 시작 전 베이스라인으로 명시적 재대입 — 이전 런 값 잔류 방지."""
    m.SPARSE_K = BASELINE["sparse_k"]
    m.TAU_MIN = BASELINE["tau_min"]
    m.TAU_CEILING = BASELINE["tau_ceiling"]
    m.TEMP_INIT = BASELINE["temp_init"]
    m.TEMP_MIN = BASELINE["temp_min"]


def deleted_epoch_from_snapshots(snapshots, part_id, snapshot_interval):
    """history['snapshots']에서 해당 part가 처음 DELETED로 기록된 스냅샷의 epoch을 반환(±snapshot_interval 해상도)."""
    for snap in snapshots:
        state = snap.get("state") or snap.get("pruning_state", {}).get("state")
        if state is not None and len(state) > part_id and state[part_id] == "DELETED":
            return snap.get("epoch")
    return None


def run_sweep():
    data, node_registry = m.load_data()
    target_area_baseline = None  # run_training 기본 동작(None -> 초기 단면적 자동 스냅샷)과 동일

    weights = {
        'w_phys': 10.0, 'w_collision': 5.0, 'w_order': 1.0, 'w_mass': 2.0,
        'w_smooth': 0.5, 'w_anchor': 0.02, 'w_sat': 0.01, 'w_sparse': m.W_SPARSE,
    }

    results = []
    for cfg in RUNS:
        print(f"\n=== Run: {cfg['id']} ===")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        reset_globals()
        m.SPARSE_K = cfg["sparse_k"]
        m.TAU_CEILING = cfg["tau_ceiling"]
        if cfg["temp_static"] is not None:
            m.TEMP_INIT = cfg["temp_static"]
            m.TEMP_MIN = cfg["temp_static"]

        target_mp = BASELINE_TARGET_MP * cfg["mp_scale"]
        target_mps = {0: target_mp}

        row = {"id": cfg["id"], "target_mp": f"{target_mp:,.0f}", "status": "OK"}
        history = None
        try:
            history, base_coords, result_coords, part_labels, best_feasible, best_mp, pruning_state = m.run_training(
                data=data,
                target_mps=target_mps,
                target_area=target_area_baseline,
                max_epochs=1500,
                lr=1e-3,
                weights=weights,
                curriculum=True,
                curriculum_ratio=(0.2, 0.7),
                snapshot_interval=10,
                feasibility_mp_err=0.02,
                feasibility_collision=0.05,
                init_log_alpha=cfg["init_log_alpha"],
            )

            has_nan = bool(np.isnan(history["loss"]).any())
            final_pred = float(np.sum(history["pred_mp"][-1]))
            row.update({
                "final_mp_err_pct": f"{abs(final_pred - target_mp) / target_mp * 100:.2f}",
                "final_l_collision": f"{history['l_collision'][-1]:.4f}",
                "final_area": f"{history['area'][-1]:.1f}",
                "part4_deleted_epoch": deleted_epoch_from_snapshots(history["snapshots"], 4, 10),
                "part3_deleted_epoch": deleted_epoch_from_snapshots(history["snapshots"], 3, 10),
                "best_feasible_epoch": best_feasible.get("epoch") if best_feasible else None,
                "best_feasible_l_collision": best_feasible.get("l_collision") if best_feasible else None,
                "final_temperature": f"{history['temperature'][-1]:.3f}",
                "final_tau_gate": f"{history['current_tau_gate'][-1]:.4f}",
            })
            if has_nan:
                row["status"] = "DIVERGED(NaN)"
        except Exception as e:
            row["status"] = f"FAILED: {type(e).__name__}: {e}"
            import traceback; traceback.print_exc()
        finally:
            history = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results.append(row)
        print(f"[{cfg['id']}] status={row['status']}")

    write_summary(results)


def write_summary(results):
    cols = ["id", "target_mp", "status", "final_mp_err_pct", "final_l_collision", "final_area",
            "part4_deleted_epoch", "part3_deleted_epoch",
            "best_feasible_epoch", "best_feasible_l_collision", "final_temperature", "final_tau_gate"]
    tmp_path = "paratune_sweep_summary.md.tmp"
    final_path = "paratune_sweep_summary.md"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "---|" * len(cols) + "\n")
        for r in results:
            f.write("| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |\n")
    import os
    os.replace(tmp_path, final_path)  # 원자적 교체 -> 도중에 죽어도 이전 결과 파일은 보존
    print(f"\n요약 표 작성 완료: {final_path}")


if __name__ == "__main__":
    run_sweep()
```

### 하니스 설계에서 Critic 라운드가 지적한 리스크 반영

- **원자적 파일 쓰기**: 표를 `.tmp`에 먼저 쓰고 `os.replace()`로 교체 — 스윕 도중 프로세스가 죽어도 마지막으로 완성된 요약 파일은 훼손되지 않는다(OpenAI Explorer 지적 반영).
- **UTF-8 명시**: `encoding="utf-8"`을 `open()`에 명시 — Windows 기본 로케일 인코딩(cp949 등)으로 한글/특수문자가 깨지는 문제 방지.
- **매 런 전역 재대입**: `reset_globals()`을 루프 시작마다 호출해 이전 런의 override가 다음 런에 잔류하지 않도록 함(OpenAI가 지적한 "module-level state persistence" 리스크 반영).
- **시드 고정 + CUDA 캐시 정리**: 매 런 시드 재설정, `finally` 블록에서 `gc.collect()` + `torch.cuda.empty_cache()`로 메모리 누수 방지.
- **DELETED epoch은 스냅샷 해상도(±10 epoch)로만 근사** — 실제 코드에 정확한 epoch을 반환하는 인터페이스가 없음을 확인했으므로, 있지도 않은 정밀도를 표에 표기하지 않는다(Gemini의 원안이 존재하지 않는 `history` 키를 가정했던 것과 같은 오류를 반복하지 않기 위함).

---

## Step 3: 실행 절차

```cmd
cd C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\uni-section\code
py paratune_harness_v1.py
```

출력: 같은 디렉터리에 `paratune_sweep_summary.md` 생성(9개 행, per-epoch 시각화 없음).

---

## Step 4: 검증/수용 기준

1. **패치 무영향 확인**: Step 1 패치 적용 후 `py uni_section_v17_paratune.py`를 단독 실행하고, 콘솔 출력(단면적/Mp 초기값, 최종 epoch 1500 요약)이 `review_v17_paratune.md`에 기록된 신버전 로그값과 정확히 일치하는지 확인한다. 어긋나면 Step 1의 리팩터링이 로직을 바꾼 것이므로 즉시 되돌린다.
2. **anchor 행 일치**: `paratune_sweep_summary.md`의 `anchor` 행이 위 단독 실행 결과와 일치해야 한다(동일 시드 42, 동일 파라미터이므로 부동소수점 오차 수준까지 동일해야 함).
3. **monkeypatch 실동작 확인**: `mp_down_0.5x` 행의 `target_mp`가 `13,710,735`(anchor의 정확히 절반)로 찍히는지 확인 — 모듈 전역 재대입이 실제로 `run_training` 내부에 반영됐다는 직접 증거.
4. **9개 행 모두 존재**: 일부 런이 `FAILED`/`DIVERGED(NaN)`이어도 나머지 런의 실행을 막지 않았는지 확인 — try/except가 루프를 계속 진행시키는지 검증.

---

<details>
<summary>숙의 과정 (Synod 세션 상세)</summary>

### 모델 기여
- **Claude (Judge/Validator):** Solver 라운드에서 Gemini가 제안한 `run_training` 시그니처/상수/`history` 키가 실제 코드와 다름을 직접 코드 읽기로 확인, Critic 프롬프트에 ground-truth로 주입. 최종 command 문서는 검증된 사실만 사용해 재작성.
- **Gemini (Architect, flash 폴백):** in-process 하니스 구조(모듈 1회 import, 전역 monkeypatch, 매 런 시드/CUDA 캐시 정리), `load_data()` 추출이 단독 실행 동작을 바꾸지 않아야 한다는 원칙, 검증 기준(anchor 일치, monkeypatch 증거) 제시. 단, 구체적 API 가정(시그니처/상수/키)은 Critic 라운드에서 대부분 기각됨.
- **OpenAI (Explorer, gpt4o):** 모듈 상태 잔류, 리팩터링 안전성, 파일 I/O 중단 시 부분 쓰기 위험을 지적 — 원자적 파일 교체(`os.replace`)와 매 런 전역 재대입으로 반영.

### 해결된 주요 쟁점
1. "Gemini가 제안한 `run_training` 호출 시그니처가 맞는가?" → 아니오, 실제 시그니처(`data`/`target_mps`/`target_area`)로 교체
2. "`history`에서 `mp_err_pct`/`tau_gate` 키를 쓸 수 있는가?" → 아니오, 실제 키는 `mp_rel_err`/`current_tau_gate` (표에서는 최종 오차를 직접 계산해 사용)
3. "DELETED epoch을 정확히 기록할 수 있는가?" → 아니오, 코드에 정밀 인터페이스가 없어 스냅샷 해상도(±10epoch)로만 근사— 존재하지 않는 정밀도를 표에 표기하지 않기로 결정

### 신뢰 점수 (최종 라운드 기준)
- Claude: 92 (코드 직접 검증 근거)
- Gemini: 60 (구조적 아이디어는 채택, API 세부사항은 대부분 기각)
- OpenAI: 85 (리스크 지적은 정확했으나 구체적 대안 코드는 제시하지 않음)

</details>

### 신뢰도: 88%
(핵심 API 시그니처/상수/키는 실제 소스 코드 직접 검증을 거쳤으므로 신뢰도가 높으나, `load_data()` 추출 및 `init_log_alpha` 인자 추가 패치와 하니스 스크립트 자체는 아직 실행해보지 않았다 — Step 4 검증을 통과하기 전까지 "패치가 기존 동작을 바꾸지 않는다"는 주장은 미확정이다.)
