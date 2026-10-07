"""UQ 샘플별 FEM (v23, bpillar_fem_v2) → 에너지 수준 × 섹션별 변형량 + 수준별 전체 섹션 최대 (샘플마다 CSV 하나).

사용 (conda env fem):
    python max_deformation_per_sample.py                    # 267개 전체, 병렬, 이미 끝난 샘플은 건너뜀
    python max_deformation_per_sample.py --samples 1 2 3    # 일부 샘플만
    python max_deformation_per_sample.py --workers 8 --n-sub 4 --energies 10 15 20 25 30

해석 (FEM/bpillar_fem_v2.py 함수를 그대로 사용)
    - 샘플 단면마다 격자 모델(n_sub = 4, PCHIP 보간, sec 1 꺾임)을 만들고 sec 5–11 Outer 윗면에 균일 압력으로
      정적 변위제어 (1 mm 증분).
    - 외력 일(흡수 에너지)이 --energy-target(기본 30 kJ)에 닿을 때까지 누르고, --energies(기본 10/15/20/25/30 kJ)
      수준마다 그 에너지에 처음 도달한 상태에서 섹션별 변형량을 읽는다 (모든 샘플을 같은 충돌 에너지로 비교,
      근거: FEM/docs/collision_energy.md — B-pillar 흡수 에너지 기준값 20 kJ, 범위 10–30 kJ).
    - 최대 변형량 판단 기준은 Plate 침입량 (탑승자 쪽 판의 −y 최대 이동량).

출력 (이 스크립트 폴더)
    bpillar_sample_001_deformation.csv …
        에너지 수준마다 sec 0–14 행 + sec = "max"(전체 섹션 최대값) 행
        열: E_kJ, sec, plate_intrusion_mm(★), plate_distortion_mm, outer_intrusion_mm, d_crush_mm,
            max_sec(★ 최대 Plate 침입 섹션, max 행만), F_kN(그 시점 총 하중), status, control_disp_mm
        status: stable(1차 한계하중 전) | snap_through(한계하중을 넘은 뒤 평형) | not_reached(수렴 실패 등)
    bpillar_sample_001_curve.csv …  제어 변위 – 총 하중 – 외력 일 – Plate 최대 침입량
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
UQ_ROOT = HERE.parents[2]                                   # 03. uq
FEM_DIR = UQ_ROOT.parent / "FEM"
DEFAULT_SAMPLE_DIR = UQ_ROOT / "sample" / "v23"
METRICS = ("plate_intrusion", "plate_distortion", "outer_intrusion", "d_crush")


def run_one(csv_path: str, out_dir: str, energy_target: float, energies: list[float], n_sub: int,
            du: float, max_disp: float) -> dict:
    """샘플 하나: FEM → 에너지 수준별 섹션 변형량 → CSV. (병렬 작업자에서 실행)"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")            # 작업자끼리 BLAS 스레드 경쟁 방지
    sys.path.insert(0, str(FEM_DIR))
    import numpy as np
    import openseespy.opensees as ops
    import pandas as pd
    import bpillar_fem_v2 as fem

    csv_path, out_dir = Path(csv_path), Path(out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_f = open(log_dir / f"{csv_path.stem}.log", "w", encoding="utf-8")

    def log(msg, **_):
        log_f.write(msg + "\n")
        log_f.flush()

    t0 = time.time()
    ops.wipe()
    ops.logFile(str(log_dir / f"{csv_path.stem}_opensees.log"), "-noEcho")
    parts = fem.read_parts(csv_path)
    mesh = fem.build_mesh(parts, n_sub, {1}, "pchip")
    n_sec = mesh["n_sec"]
    fem.build_model(mesh)
    disp, force, energy, U, col = fem.run_static(mesh, plate_target=np.inf, max_disp=max_disp, du=du,
                                                 energy_target=energy_target * 1e6, log=log)
    metrics = fem.section_metrics(mesh, U, col)
    states, F_lim = fem.energy_states(force, energy, [E * 1e6 for E in energies])

    tables = []
    for E0, st in zip(energies, states):
        ok = st["j"] is not None
        df = pd.DataFrame([dict(E_kJ=E0, sec=s, **{f"{m}_mm": fem.interp(metrics[m], st)[s] if ok else np.nan
                                                   for m in METRICS}) for s in range(n_sec)])
        top = df.loc[df["plate_intrusion_mm"].idxmax(), "sec"] if ok else np.nan
        df.loc[len(df)] = dict(E_kJ=E0, sec="max", **{f"{m}_mm": df[f"{m}_mm"].max() for m in METRICS})
        df["max_sec"] = [""] * n_sec + [top]
        df["F_kN"] = st["F0"] / 1e3 if ok else np.nan
        df["status"] = st["status"]
        df["control_disp_mm"] = fem.interp(disp, st) if ok else np.nan
        tables.append(df)
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(out_dir / f"{csv_path.stem}_deformation.csv", index=False, float_format="%.4f")
    pd.DataFrame(dict(control_disp_mm=disp, force_kN=force / 1e3, energy_kJ=energy / 1e6,
                      plate_intrusion_max_mm=metrics["plate_intrusion"].max(axis=1))).to_csv(
        out_dir / f"{csv_path.stem}_curve.csv", index=False, float_format="%.4f")

    log(f"done {time.time() - t0:.0f}s F_lim1={F_lim / 1e3:.0f} kN, 외력 일 {energy[-1] / 1e6:.2f} kJ")
    log_f.close()
    mx = out[out["sec"] == "max"].set_index("E_kJ")
    return dict(sample=csv_path.stem, time=time.time() - t0, F_lim=F_lim / 1e3,
                status=dict(mx["status"]), plate=dict(mx["plate_intrusion_mm"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR), help="샘플 단면 CSV 폴더")
    ap.add_argument("--out-dir", default=str(HERE), help="출력 폴더")
    ap.add_argument("--samples", type=int, nargs="+", help="돌릴 샘플 번호 (기본: 전체)")
    ap.add_argument("--energy-target", type=float, default=30.0, help="외력 일 목표 (kJ) — 여기까지 누른다")
    ap.add_argument("--energies", type=float, nargs="+", default=[10.0, 15.0, 20.0, 25.0, 30.0],
                    help="변형량을 읽을 에너지 수준 (kJ)")
    ap.add_argument("--n-sub", type=int, default=4, help="섹션 사이 분할 수 (짝수)")
    ap.add_argument("--du", type=float, default=1.0, help="변위 증분 (mm)")
    ap.add_argument("--max-disp", type=float, default=300.0, help="제어 변위 상한 (mm)")
    ap.add_argument("--workers", type=int, default=8, help="동시 실행 수 (CPU 코어, 기본 8)")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 결과도 다시 계산")
    a = ap.parse_args()
    if a.n_sub % 2:
        ap.error("--n-sub는 짝수여야 합니다")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.sample_dir).glob("bpillar_sample_[0-9]*.csv"))
    if a.samples:
        wanted = set(a.samples)
        files = [f for f in files if int(f.stem.rsplit("_", 1)[1]) in wanted]
    todo = [f for f in files if a.overwrite or not (out_dir / f"{f.stem}_deformation.csv").exists()]
    print(f"[start] 샘플 {len(files)}개 중 {len(todo)}개 계산 (나머지는 결과 있음), 외력 일 {a.energy_target:.0f} kJ까지 "
          f"(수준 {a.energies} kJ), n_sub={a.n_sub}, du={a.du} mm, 동시 {a.workers}개", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, str(f), str(out_dir), a.energy_target, a.energies, a.n_sub, a.du, a.max_disp): f
                for f in todo}
        for i, fut in enumerate(as_completed(futs), start=1):
            f = futs[fut]
            try:
                r = fut.result()
                pl = " / ".join(f"{v:.1f}" for v in r["plate"].values())
                bad = [E for E, s in r["status"].items() if s == "not_reached"]
                print(f"  [{i}/{len(todo)}] {r['sample']}: Plate 최대 {pl} mm (E = {'/'.join(f'{E:g}' for E in r['plate'])} kJ), "
                      f"한계하중 {r['F_lim']:.0f} kN{', not_reached ' + str(bad) if bad else ''}, {r['time']:.0f}s"
                      f"  | 경과 {(time.time() - t0) / 60:.0f}분", flush=True)
            except Exception as e:                           # 한 샘플이 실패해도 나머지는 계속
                print(f"  [{i}/{len(todo)}] {f.stem}: 실패 — {e!r}", flush=True)
    print(f"[done] {(time.time() - t0) / 60:.1f}분 → {out_dir}")


if __name__ == "__main__":
    main()
