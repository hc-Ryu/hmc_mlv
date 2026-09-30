"""결과 그림 (matplotlib, 파일로만 저장)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import N_SECTIONS, Config  # noqa: E402

METRIC_LABELS = [("delta_dir", "δ_dir"), ("delta_rel", "δ_rel"),
                 ("delta_dist", "δ_dist"), ("d_crush", "d_crush")]


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def section_bars(ax, sec: np.ndarray, values: dict, cfg: Config, title: str) -> None:
    """섹션별 막대 (지표 여러 개) + 하중 구간 음영."""
    w = 0.8 / len(values)
    for i, (label, v) in enumerate(values.items()):
        ax.bar(sec + (i - (len(values) - 1) / 2) * w, v, w, label=label)
    s_a, s_b = cfg.load_secs
    ax.axvspan(s_a - 0.5, s_b + 0.5, color="0.92", zorder=0, label="load secs")
    ax.set_xticks(sec)
    ax.set_xlabel(f"section (0 = sill, {N_SECTIONS - 1} = roof rail)")
    ax.set_ylabel("deformation (mm)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def plot_run(out: Path, cfg: Config, table: pd.DataFrame, series: pd.DataFrame) -> None:
    """해석 1회: 섹션별 전 구간 최대값 + 하중 구간 섹션의 δ_dir 이력."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    section_bars(axes[0], table["sec"].to_numpy(), {lab: table[c] for c, lab in METRIC_LABELS},
                 cfg, f"{cfg.name()} — max over run")
    ax = axes[1]
    static = cfg.mode == "static"
    scale = 1.0 if static else 1e3
    s_a, s_b = cfg.load_secs
    for s in range(s_a, s_b + 1):
        d = series[series["sec"] == s]
        ax.plot(d["t"] * scale, d["delta_dir"], label=f"sec {s}")
    ax2 = ax.twinx()
    d = series[series["sec"] == s_a]
    ax2.plot(d["t"] * scale, d["force"] / 1e3, "k--", lw=1)
    ax2.set_ylabel("F_tot (kN, dashed)")
    ax.set_xlabel("control disp (mm)" if static else "time (ms)")
    ax.set_ylabel("δ_dir (mm)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out / "section_max_deform.png")


def plot_force_levels(out: Path, cfg: Config, series: pd.DataFrame, force_table: pd.DataFrame,
                      limits: dict) -> None:
    """정적 해석: 힘-변위 곡선(유효 구간, 1차 한계) + 하중 수준별 섹션 δ_dir 분포."""
    curve = series[series["sec"] == 0].sort_values("frame")
    d, F = curve["t"].to_numpy(), curve["force"].to_numpy() / 1e3
    valid = d <= limits["valid_until_control_disp"] + 1e-9

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(d[valid], F[valid], "k-", label="valid")
    if (~valid).any():
        ax.plot(d[~valid], F[~valid], "k:", label="penetration (invalid)")
    ax.plot([limits["control_disp_at_lim1"]], [limits["F_lim1_kN"]], "ro",
            label=f"F_lim1 = {limits['F_lim1_kN']:.0f} kN")
    for F0 in force_table["F0_kN"]:
        ax.axhline(F0, color="0.88", lw=0.8)
    ax.set_xlabel("control disp (mm)")
    ax.set_ylabel("F_tot (kN)")
    ax.set_title(f"{cfg.name()} — load mode: {cfg.load_mode}", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ok = force_table[force_table["status"].isin(["stable", "snap_through"])]
    for _, r in ok.iterrows():
        prof = [r[f"delta_dir_s{s}"] for s in range(N_SECTIONS)]
        ls = "-" if r["status"] == "stable" else "--"
        ax.plot(range(N_SECTIONS), prof, ls, marker="o", ms=3, label=f"{r['F0_kN']:.0f} kN ({r['status']})")
    s_a, s_b = cfg.load_secs
    ax.axvspan(s_a - 0.5, s_b + 0.5, color="0.93", zorder=0)
    ax.set_xlabel(f"section (0 = sill, {N_SECTIONS - 1} = roof rail)")
    ax.set_ylabel("δ_dir (mm)")
    ax.set_title("section intrusion at each force level", fontsize=10)
    ax.legend(fontsize=7)
    for a in axes:
        a.grid(alpha=0.3)
    _save(fig, out / "force_levels.png")


def plot_part_onset(out: Path, part_hist: pd.DataFrame, force_levels_kN) -> None:
    """파트별 찌그러짐·항복 비율 이력."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for part, d in part_hist.groupby("part", sort=False):
        axes[0].plot(d["t"], d["distortion"], label=part)
        if "yield_frac" in d:
            axes[1].plot(d["t"], 100 * d["yield_frac"], label=part)
        axes[2].plot(d["force"] / 1e3, d["distortion"], label=part)
    axes[0].set_xlabel("control disp (mm)")
    axes[0].set_ylabel("part distortion (mm)")
    axes[1].set_xlabel("control disp (mm)")
    axes[1].set_ylabel("yielded elements (%)")
    axes[2].set_xlabel("F_tot (kN)")
    axes[2].set_ylabel("part distortion (mm)")
    for F0 in force_levels_kN:
        axes[2].axvline(F0, color="0.88", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    _save(fig, out / "part_onset.png")


def plot_sweep(out: Path, sweep: pd.DataFrame) -> None:
    """explicit α 스윕: 하중 – 섹션별 최대 δ_dir / δ_dist."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for s in sorted(sweep["sec"].unique()):
        d = sweep[sweep["sec"] == s]
        axes[0].plot(d["F0_kN"], d["delta_dir"], "o-", label=f"sec {s}")
        axes[1].plot(d["F0_kN"], d["delta_dist"], "o-", label=f"sec {s}")
    axes[0].set_ylabel("δ_dir max (mm)")
    axes[1].set_ylabel("δ_dist max (mm)")
    for ax in axes:
        ax.set_xlabel("F0 (kN)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3)
    _save(fig, out / "sweep_summary.png")
