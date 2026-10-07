"""UQ 샘플별 FEM → 섹션별 최대 변형량 + 전체 섹션 최대 변형량 (샘플마다 CSV 하나).

사용 (conda env fem):
    python max_deormation_per_sample.py                     # 267개 전체, 병렬, 이미 끝난 샘플은 건너뜀
    python max_deormation_per_sample.py --samples 1 2 3     # 일부 샘플만
    python max_deormation_per_sample.py --force 800 --workers 8   # 기본 동시 실행 10개

해석 (FEM/bpillar_fem_demo.py 함수를 그대로 사용)
    - 샘플 단면마다 격자 모델을 만들고 sec 5–11 Outer 윗면에 균일 압력으로 정적 변위제어.
    - 총 하중이 --force(기본 1000 kN)에 닿을 때까지 누르고, 그 하중에 처음 도달한 상태에서
      섹션별 변형량을 읽는다 (모든 샘플을 같은 하중으로 비교).
    - 최대 변형량 판단 기준은 Plate 침입량 (탑승자 쪽 판의 −y 최대 이동량).
    - 기본 격자는 빠른 모드(섹션 사이 분할 없음, --n-sub 1, 2 mm 증분). 샘플당 약 5–7분.

출력 (이 스크립트 폴더)
    bpillar_sample_001_deformation.csv …
        sec 0–16 행 + 마지막 행 sec = "max"(전체 섹션 최대값)
        열: sec, plate_intrusion_mm(★), plate_distortion_mm, outer_intrusion_mm, d_crush_mm,
            max_sec(★ 최대 Plate 침입 섹션, max 행만), F_eval_kN, status, control_disp_mm
        status: stable(1차 한계하중 이하) | snap_through(한계하중을 넘어 다음 평형) | not_reached(수렴 실패 등)
    logs/bpillar_sample_001.log …   진행 로그와 OpenSees 경고
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
UQ_ROOT = HERE.parents[1]                                   # 03. uq
FEM_DIR = UQ_ROOT.parent / "FEM"
DEFAULT_SAMPLE_DIR = UQ_ROOT / "sample"
METRICS = ("plate_intrusion", "plate_distortion", "outer_intrusion", "d_crush")


def run_one(csv_path: str, out_dir: str, force_kN: float, n_sub: int, du: float, max_disp: float) -> dict:
    """샘플 하나: FEM → 하중 force_kN에서 섹션별 변형량 → CSV. (병렬 작업자에서 실행)"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")            # 작업자끼리 BLAS 스레드 경쟁 방지
    sys.path.insert(0, str(FEM_DIR))
    import numpy as np
    import openseespy.opensees as ops
    import pandas as pd
    import bpillar_fem_demo as fem

    csv_path, out_dir = Path(csv_path), Path(out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_f = open(log_dir / f"{csv_path.stem}.log", "w", encoding="utf-8")

    def log(msg):
        log_f.write(msg + "\n")
        log_f.flush()

    t0 = time.time()
    ops.logFile(str(log_dir / f"{csv_path.stem}_opensees.log"), "-noEcho")
    parts = fem.read_parts(csv_path)
    mesh = fem.build_mesh(parts, n_sub, fem.HEIGHT / (fem.N_SEC - 1))
    fem.build_model(mesh)
    F_eval = force_kN * 1e3
    disp, force, U, col = fem.run_static(mesh, plate_target=np.inf, max_disp=max_disp, du=du,
                                         force_target=F_eval, log=log)
    metrics = fem.section_metrics(mesh, U, col)
    states, F_lim = fem.force_states(force, [F_eval])
    st = states[0]
    ok = st["j"] is not None

    rows = []
    for s in range(fem.N_SEC):
        rows.append(dict(sec=s, **{f"{m}_mm": fem.interp(metrics[m], st)[s] if ok else np.nan for m in METRICS}))
    df = pd.DataFrame(rows)
    top = df.loc[df["plate_intrusion_mm"].idxmax(), "sec"] if ok else np.nan
    df.loc[len(df)] = dict(sec="max", **{f"{m}_mm": df[f"{m}_mm"].max() for m in METRICS})
    df["max_sec"] = [""] * fem.N_SEC + [top]
    df["F_eval_kN"] = force_kN
    df["status"] = st["status"]
    df["control_disp_mm"] = fem.interp(disp, st) if ok else np.nan
    df.to_csv(out_dir / f"{csv_path.stem}_deformation.csv", index=False, float_format="%.4f")

    log(f"done {time.time() - t0:.0f}s status={st['status']} F_lim1={F_lim / 1e3:.0f} kN")
    log_f.close()
    return dict(sample=csv_path.stem, status=st["status"], time=time.time() - t0,
                plate_max=float(df["plate_intrusion_mm"].iloc[-1]), max_sec=top)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR), help="샘플 단면 CSV 폴더")
    ap.add_argument("--out-dir", default=str(HERE), help="출력 폴더")
    ap.add_argument("--samples", type=int, nargs="+", help="돌릴 샘플 번호 (기본: 전체)")
    ap.add_argument("--force", type=float, default=1000.0, help="변형량을 읽을 총 하중 (kN)")
    ap.add_argument("--n-sub", type=int, default=1, help="섹션 사이 분할 수 (4 = 기본 격자, 매우 느림)")
    ap.add_argument("--du", type=float, default=2.0, help="변위 증분 (mm)")
    ap.add_argument("--max-disp", type=float, default=300.0, help="제어 변위 상한 (mm)")
    ap.add_argument("--workers", type=int, default=10, help="동시 실행 수 (CPU 코어, 기본 10)")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 결과도 다시 계산")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.sample_dir).glob("bpillar_sample_[0-9]*.csv"))
    if a.samples:
        wanted = set(a.samples)
        files = [f for f in files if int(f.stem.rsplit("_", 1)[1]) in wanted]
    todo = [f for f in files if a.overwrite or not (out_dir / f"{f.stem}_deformation.csv").exists()]
    print(f"[start] 샘플 {len(files)}개 중 {len(todo)}개 계산 (나머지는 결과 있음), 하중 {a.force:.0f} kN, "
          f"n_sub={a.n_sub}, 동시 {a.workers}개", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, str(f), str(out_dir), a.force, a.n_sub, a.du, a.max_disp): f for f in todo}
        for i, fut in enumerate(as_completed(futs), start=1):
            f = futs[fut]
            try:
                r = fut.result()
                print(f"  [{i}/{len(todo)}] {r['sample']}: {r['status']}, Plate 최대 {r['plate_max']:.1f} mm "
                      f"(sec {r['max_sec']}), {r['time']:.0f}s  | 경과 {(time.time() - t0) / 60:.0f}분", flush=True)
            except Exception as e:                           # 한 샘플이 실패해도 나머지는 계속
                print(f"  [{i}/{len(todo)}] {f.stem}: 실패 — {e}", flush=True)
    print(f"[done] {(time.time() - t0) / 60:.1f}분 → {out_dir}")


if __name__ == "__main__":
    main()
