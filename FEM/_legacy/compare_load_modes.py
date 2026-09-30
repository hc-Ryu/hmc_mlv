"""하중 적용 방식 비교 (idea_v1.md §7.7): crown6 / top / impactor.

방식마다 제어 변위 정의가 다르므로(crown6·top: sec 8 crown 절점, impactor: 판 변위)
공통 축은 sec 8 최대 침입량 δ_dir(sec 8)로 맞춘다.

    python compare_load_modes.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results"     # hmc_mlv/FEM/results
CASES = {
    "crown6": ROOT / "t150_static_fixed_n4_contactxy",
    "top": ROOT / "t150_static_fixed_n4_contactxy_top",
    "impactor": ROOT / "t150_static_fixed_n4_contactxy_impactor",
}
OUT = ROOT / "compare_load_modes"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    series, limits, alpha, onset = {}, {}, {}, {}
    for name, d in CASES.items():
        series[name] = pd.read_csv(d / "section_history.csv")
        limits[name] = json.loads((d / "static_limits.json").read_text(encoding="utf-8"))
        alpha[name] = pd.read_csv(d / "static_alpha_summary.csv")
        onset[name] = pd.read_csv(d / "part_onset.csv")

    # ── 요약 표 ──
    rows = []
    for name in CASES:
        s8 = series[name][series[name]["sec"] == 8]
        L = limits[name]
        rows.append(dict(mode=name, F_lim1_kN=L["F_lim1_kN"], alpha_lim1=L["alpha_lim1"],
                         d8_at_lim1=float(np.interp(L["control_disp_at_lim1"], s8["t"], s8["delta_dir"])),
                         F_linear_limit_kN=L["F_linear_limit_kN"],
                         valid_until=L["valid_until_control_disp"],
                         F_at_valid_end_kN=L["F_at_valid_end_kN"]))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False, float_format="%.4g")

    cols = [f"delta_dir_s{s}" for s in range(5, 12)]
    at = []
    for name in CASES:
        a = alpha[name][["alpha", "F0_kN", "status"] + [c for c in cols if c in alpha[name]]].copy()
        a.insert(0, "mode", name)
        at.append(a)
    at = pd.concat(at, ignore_index=True)
    at.to_csv(OUT / "alpha_compare.csv", index=False, float_format="%.4g")

    on = []
    for name in CASES:
        o = onset[name][["part", "first_yield_F_kN", "distort5mm_F_kN"]].copy()
        o.insert(0, "mode", name)
        on.append(o)
    on = pd.concat(on, ignore_index=True)
    on.to_csv(OUT / "part_onset_compare.csv", index=False, float_format="%.4g")

    # ── 그림 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for name in CASES:
        s = series[name]
        s8 = s[s["sec"] == 8].sort_values("frame")
        axes[0].plot(s8["delta_dir"], s8["force"] / 1e3, label=name)
        wide = s.pivot(index="frame", columns="sec", values="delta_dir")
        F = s8["force"].to_numpy()
        for F0, ls in [(178.7e3, "-"), (357.3e3, "--")]:
            i = int(np.argmax(F >= F0)) if (F >= F0).any() else None
            if i:
                axes[1].plot(wide.columns, wide.iloc[i], ls, label=f"{name} F={F0 / 1e3:.0f} kN")
    for a in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
        axes[0].axhline(a * 446.6, color="0.88", lw=0.8)
    axes[0].set_xlabel("δ_dir sec 8 (mm)")
    axes[0].set_ylabel("F_tot (kN)")
    axes[0].set_title("force vs sec 8 intrusion")
    axes[1].set_xlabel("section (0 = sill, 16 = roof)")
    axes[1].set_ylabel("δ_dir (mm)")
    axes[1].set_title("intrusion profile at first reach of F0")
    axes[1].axvspan(4.5, 11.5, color="0.93", zorder=0)
    for name in CASES:
        o = onset[name].set_index("part")
        axes[2].plot(o.index, o["first_yield_F_kN"], "o-", label=f"{name} first yield")
        axes[2].plot(o.index, o["distort5mm_F_kN"], "s--", label=f"{name} distortion ≥ 5 mm")
    axes[2].set_ylabel("F_tot (kN)")
    axes[2].set_title("part deformation onset")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "compare_load_modes.png", dpi=120)
    plt.close(fig)

    pd.set_option("display.width", 200)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print(at.to_string(index=False, float_format=lambda v: f"{v:7.1f}"))
    print(on.pivot(index="part", columns="mode").round(0).to_string())


if __name__ == "__main__":
    main()
