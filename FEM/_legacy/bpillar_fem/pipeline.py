"""해석 1회 실행 → 후처리 → 결과 폴더 저장.

결과 폴더 (정적 해석 기준)
  section_deformation.csv  ★ 하중 수준(F0) × 섹션별 변형량 (δ_dir, δ_rel, δ_dist, d_crush, status)
  section_delta_dir.csv    ★ 위 표의 δ_dir만 섹션 × 하중(kN) 피벗
  force_levels.png         힘-변위 곡선 + 하중 수준별 섹션 침입량 분포
  static_limits.json       1차 한계하중, 선형 한계, 유효 구간
  static_alpha_summary.csv 하중 수준별 wide 표 (control_disp 포함)
  part_onset.csv/.png      파트별 변형 시작 하중 (최초 항복, 찌그러짐 ≥ 1/5 mm)
  section_history.csv      프레임 × 섹션 지표 이력
  section_max_deform.csv/.png  전 구간 최대값
  history.npz, config.json, log.txt
"""
from __future__ import annotations

import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots, post, solve_ops
from .config import Config
from .geometry import Mesh, build_mesh
from .io_csv import read_parts
from .solve_ops import History


class RunLog:
    """콘솔과 log.txt에 동시에 쓴다."""

    def __init__(self, path: Path, prefix: str):
        self.f = open(path, "w", encoding="utf-8")
        self.prefix = prefix

    def __call__(self, msg: str) -> None:
        line = f"[{self.prefix}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def _fmt(df: pd.DataFrame, digits: int = 2) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.{digits}f}")


def run_case(cfg: Config, alphas=None) -> tuple[float, float, pd.DataFrame, str]:
    """해석 1회. 정적 해석이면 alphas(하중 수준 = α·P_c)별 섹션 변형 표까지 만든다."""
    out = Path(cfg.out_dir) / cfg.name()
    out.mkdir(parents=True, exist_ok=True)
    log = RunLog(out / "log.txt", cfg.name())
    # OpenSees 경고(수렴 재시도, zeroLength 길이 등)는 콘솔 대신 파일로
    import openseespy.opensees as ops
    ops.logFile(str(out / "opensees.log"), "-noEcho")
    log(f"csv={cfg.csv_path}")
    log(f"mode={cfg.mode} load_mode={cfg.load_mode} bc={cfg.bc} n_sub={cfg.n_sub} "
        f"P_c={cfg.P_c / 1e3:.1f} kN" + (f" F0={cfg.F0 / 1e3:.1f} kN" if cfg.mode == "explicit" else ""))
    t0 = time.time()
    if cfg.mode == "static":
        mesh, hist = solve_ops.run_static(cfg, log=log)
    elif cfg.mode == "explicit":
        mesh, hist = solve_ops.run_explicit(cfg, log=log)
    else:
        raise ValueError(cfg.mode)
    table = write_results(out, mesh, hist, alphas, log)
    extra = {k: v for k, v in hist.extra.items() if not isinstance(v, (list, dict))}
    log(f"done in {time.time() - t0:.0f}s  {extra}")
    log.close()
    return cfg.alpha, cfg.F0, table, str(out)


def write_results(out: Path, mesh: Mesh, hist: History, alphas, log=print) -> pd.DataFrame:
    """후처리 계산 → 파일 저장. 섹션별 전 구간 최대값 표를 돌려준다."""
    cfg = mesh.cfg
    cfg.save(out / "config.json")
    np.savez_compressed(out / "history.npz", **post.history_arrays(hist))

    table, series = post.section_metrics(mesh, hist)
    table.to_csv(out / "section_max_deform.csv", index=False, float_format="%.5g")
    series.to_csv(out / "section_history.csv", index=False, float_format="%.5g")
    plots.plot_run(out, cfg, table, series)

    if cfg.mode == "static" and alphas:
        force_table, limits = post.static_force_table(series, alphas, cfg.P_c)
        force_table.to_csv(out / "static_alpha_summary.csv", index=False, float_format="%.5g")
        (out / "static_limits.json").write_text(json.dumps(limits, indent=2), encoding="utf-8")
        long = post.deformation_long(force_table, cfg.dz)
        long.to_csv(out / "section_deformation.csv", index=False, float_format="%.4g")
        long.pivot(index="sec", columns="F0_kN", values="delta_dir").to_csv(
            out / "section_delta_dir.csv", float_format="%.3f")
        plots.plot_force_levels(out, cfg, series, force_table, limits)

        onset, part_hist = post.part_onset(post.part_metrics(mesh, hist), cfg.P_c)
        onset.to_csv(out / "part_onset.csv", index=False, float_format="%.4g")
        part_hist.to_csv(out / "part_history.csv", index=False, float_format="%.5g")
        plots.plot_part_onset(out, part_hist, force_table["F0_kN"])

        log(f"1차 한계하중 {limits['F_lim1_kN']:.0f} kN (α {limits['alpha_lim1']:.2f}), "
            f"유효 구간 ~{limits['valid_until_control_disp']:.0f} mm")
        s_a, s_b = cfg.load_secs
        cols = ["F0_kN", "status"] + [f"delta_dir_s{s}" for s in range(s_a, s_b + 1)]
        log("하중 수준별 섹션 δ_dir (mm)\n" + _fmt(force_table[[c for c in cols if c in force_table]], 1))
        log("파트별 변형 시작 (kN)\n" + _fmt(onset[["part", "first_yield_F_kN", "distort5mm_F_kN"]], 0))
    else:
        log("섹션별 최대 변형량 (mm)\n"
            + _fmt(table[["sec", "delta_dir", "delta_rel", "delta_dist", "d_crush"]], 2))
    return table


def reprocess(case_dir: Path, alphas) -> pd.DataFrame:
    """저장된 history.npz + config.json으로 후처리만 다시 한다 (해석은 다시 돌리지 않음)."""
    saved = json.loads((case_dir / "config.json").read_text(encoding="utf-8"))
    names = {f.name for f in fields(Config)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in saved.items() if k in names}
    # 옵션이 추가되기 전 결과: 당시 동작으로 읽는다
    kw.setdefault("contact", False)
    kw.setdefault("contact_x", False)
    kw.setdefault("load_mode", "crown6")
    cfg = Config(**kw)
    mesh = build_mesh(read_parts(cfg.csv_path), cfg)
    hist = post.history_from_npz(np.load(case_dir / "history.npz"))
    return write_results(case_dir, mesh, hist, alphas)
