"""B-pillar Plate 최대 변형량 PDD surrogate + 전역 민감도 분석 (v23 — 에너지 기준 FEM 결과).

사용 (conda env fem):
    python "max_deform_PDD+Sensitivity_v23.py"                   # FEM이 끝난 샘플만 모아, 에너지 수준마다 학습
    python "max_deform_PDD+Sensitivity_v23.py" --energies 20     # 20 kJ(기준값)만
    python "max_deform_PDD+Sensitivity_v23.py" --m 3 --sec 8     # 차수 3, sec 8 Plate 침입량을 출력으로

입력 X  : sample/v23/bpillar_samples_design.csv 의 ξ
          (22개 변수, 기본값 ±10 % 균일분포를 [−1, 1]로 표준화: ξ = (x / nominal − 1) / 0.1)
          x1–x7 두께(Outer, Plate, Inner, Patch1, Patch2, Patch3, Patch4), x8–x22 섹션 0–14 Inner–Plate 거리
출력 y  : results/v23/deformation/bpillar_sample_XXX_deformation.csv 의 Plate 침입량 (mm)
          에너지 수준(E_kJ = 10/15/20/25/30, 외력 일이 그 값에 처음 닿은 상태)마다 따로 학습한다.
          기본은 "max" 행 = 전체 섹션 최대값 (--sec로 특정 섹션 선택)
          FEM 결과가 없거나 status가 not_reached인 샘플은 제외 (해석 진행 중에도 실행 가능).
          stable / snap_through는 같은 에너지에서 읽은 값이라 구분 없이 함께 쓴다.

PDD (S = 1, 1변수 분해)
    y(ξ) ≈ c0 + Σ_j Σ_{k=1..m} c_jk ψ_k(ξ_j),   ψ_k(t) = P_k(t)·√(2k+1)  (균일분포에 대해 정규직교)
    평균 = c0,  분산 = Σ c_jk²,  1차 Sobol 지수 S_j = Σ_k c_jk² / 분산   (S = 1이므로 Σ S_j = 1)
    계수 수 = 1 + N·m (N = 22, m = 2 → 45, m = 4 → 89),  권장 학습 샘플 = 계수 수 × 3
    차수 선택 참고용으로 m = 1–4의 LOOCV R²·RMSE를 요약에 함께 적는다.

출력 (results/v23/pdd_sensitivity/E{에너지}kJ/ — 에너지 수준마다 아래 파일 한 벌)
    pdd_coefficients.csv       계수 c0, c_jk
    pdd_sensitivity.csv        변수별 1차 Sobol 지수, 분산 기여 (큰 순)
    pdd_summary.txt            학습 샘플 수, 평균·분산(실측 vs PDD vs MC), R², LOOCV 오차, 차수별 비중, 차수 비교
    pdd_training_data.csv      학습에 쓴 샘플 번호, y, PDD 예측, LOOCV 예측, status
    fea_output_pdf.png         FEA 출력 분포
    fea_vs_surrogate_pdf.png   FEA vs surrogate 몬테카를로 분포
    surrogate_fit.png          실측 vs 예측 (학습 / LOOCV)
    sensitivity_bar.png        22개 변수 1차 Sobol 지수
    top4_main_effects.png      상위 4개 변수 주효과 곡선
    top4_scatter.png           상위 4개 변수 surrogate 산점도
(results/v23/pdd_sensitivity/ 바로 아래)
    sensitivity_by_energy.csv  에너지 수준 × 변수 1차 Sobol 지수 (%) 와 수준별 평균·표준편차·LOOCV R²
    sensitivity_by_energy.png  에너지 수준에 따른 상위 변수 Sobol 지수 변화
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import legendre

HERE = Path(__file__).resolve().parent                       # 03. uq
DESIGN_CSV = HERE / "sample" / "v23" / "bpillar_samples_design.csv"
DEFORM_DIR = HERE / "results" / "v23" / "deformation"
OUT_DIR = HERE / "results" / "v23" / "pdd_sensitivity"
Y_LABEL = "Plate max intrusion (mm)"


# ─────────────────────────────── 데이터 ───────────────────────────────
def load_inputs(design_csv) -> tuple[pd.DataFrame, list[str]]:
    """샘플 번호 → 표준화 입력 ξ. 반환: (sample을 인덱스로 한 ξ 표, 변수 이름)."""
    d = pd.read_csv(design_csv)
    if "case" in d.columns:                                   # v21 overview 형식 (nominal 행 포함)
        d = d[d["case"] == "sample"]
    d = d.set_index("sample")
    xi_cols = [c for c in d.columns if c.startswith("xi_")]
    return d[xi_cols], [c[3:] for c in xi_cols]


def load_outputs(deform_dir, sec="max", column="plate_intrusion_mm") -> tuple[pd.DataFrame, pd.DataFrame]:
    """샘플별 FEM 결과 → (sample × E_kJ) 출력값 y 표와 status 표.
    읽기 실패(쓰는 중·빈 파일)인 샘플은 제외, not_reached / NaN인 (샘플, 에너지)는 NaN."""
    y, status, skipped = {}, {}, []
    for f in sorted(Path(deform_dir).glob("bpillar_sample_*_deformation.csv")):
        i = int(f.stem.split("_")[2])
        try:
            df = pd.read_csv(f, dtype={"sec": str})
            rows = df[df["sec"] == str(sec)].set_index("E_kJ")
            if rows.empty:
                raise ValueError(f"sec {sec} 없음")
        except Exception as e:
            skipped.append(f"{i}({type(e).__name__}: {e})")
            continue
        y[i] = rows[column].where(rows["status"] != "not_reached")
        status[i] = rows["status"]
    if skipped:
        print(f"[skip] {len(skipped)}개 제외: " + ", ".join(skipped))
    return pd.DataFrame(y).T.sort_index(), pd.DataFrame(status).T.sort_index()


# ─────────────────────────────── PDD ───────────────────────────────
def psi(t: np.ndarray, k: int) -> np.ndarray:
    """정규직교 Legendre ψ_k(t) = P_k(t)·√(2k+1)  (t ∈ [−1, 1] 균일분포에 대해 E[ψ_k²] = 1)."""
    coef = np.zeros(k + 1)
    coef[k] = 1.0
    return legendre.legval(t, coef) * np.sqrt(2 * k + 1)


def basis(X: np.ndarray, m: int) -> np.ndarray:
    """S = 1 PDD 기저 행렬 (n, 1 + N·m): [1, ψ_1(ξ_1)…ψ_m(ξ_1), ψ_1(ξ_2)…, …]."""
    n, N = X.shape
    cols = [np.ones(n)]
    for j in range(N):
        cols += [psi(X[:, j], k) for k in range(1, m + 1)]
    return np.column_stack(cols)


def fit(X, y, m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """최소제곱 계수, 학습 예측, LOOCV 예측 (hat 행렬로 한 번에: e_loo = e / (1 − h_ii))."""
    A = basis(X, m)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_fit = A @ c
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))          # hat 행렬 대각 H = A (AᵀA)⁻¹ Aᵀ
    y_loo = y - (y - y_fit) / np.clip(1.0 - h, 1e-12, None)
    return c, y_fit, y_loo


def sobol_first_order(c, N, m) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """계수 → 변수별 1차 Sobol 지수, 분산 기여, 전체 분산, 차수별 비중."""
    cj = np.asarray(c[1:]).reshape(N, m)
    var = float((cj ** 2).sum())
    contrib = (cj ** 2).sum(axis=1)
    if var == 0:
        return np.zeros(N), contrib, var, np.zeros(m)
    return contrib / var, contrib, var, (cj ** 2).sum(axis=0) / var


def main_effect(c, j, m, t) -> np.ndarray:
    """변수 j만 움직이고 나머지는 평균일 때의 surrogate: c0 + Σ_k c_jk ψ_k(t)."""
    return c[0] + sum(c[1 + j * m + k - 1] * psi(t, k) for k in range(1, m + 1))


def r2(y, p) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


# ─────────────────────────────── 그림 ───────────────────────────────
def kde(data, x) -> np.ndarray:
    """가우시안 KDE (Scott 대역폭)."""
    data = np.asarray(data, float)
    bw = 1.06 * data.std(ddof=1) * len(data) ** (-1 / 5)
    if bw <= 0:
        return np.zeros_like(x)
    z = (x[:, None] - data[None]) / bw
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(data) * bw * np.sqrt(2 * np.pi))


def make_plots(out, y, y_fit, y_loo, y_mc, X_mc, c, S1, names, m, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10})

    lo, hi = min(y.min(), y_mc.min()), max(y.max(), y_mc.max())
    x = np.linspace(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo), 512)

    fig = plt.figure(figsize=(9, 4.8))
    plt.hist(y, bins=max(10, len(y) // 10), density=True, alpha=0.5)
    plt.plot(x, kde(y, x), lw=2, label="PDF (KDE)")
    plt.xlabel(Y_LABEL)
    plt.ylabel("Density")
    plt.title(f"FEA output distribution at {tag} (n = {len(y)})")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fea_output_pdf.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 4.8))
    plt.hist(y, bins=max(10, len(y) // 10), density=True, alpha=0.35, label="FEA data")
    plt.hist(y_mc, bins=100, density=True, alpha=0.5, label="Surrogate (MC)")
    plt.plot(x, kde(y, x), lw=1.5, label="FEA KDE")
    plt.xlabel(Y_LABEL)
    plt.ylabel("Density")
    plt.title(f"Probability distribution at {tag}: FEA vs PDD surrogate")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fea_vs_surrogate_pdf.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, p, lab in zip(axes, (y_fit, y_loo), ("training fit", "leave-one-out")):
        ax.scatter(y, p, s=12)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"FEA {Y_LABEL}")
        ax.set_ylabel("PDD prediction")
        ax.set_title(f"{lab}: R² = {r2(y, p):.3f}, RMSE = {np.sqrt(np.mean((y - p) ** 2)):.2f}")
        ax.grid(alpha=0.3)
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(out / "surrogate_fit.png", dpi=200)
    plt.close(fig)

    order = np.argsort(S1)[::-1]
    fig = plt.figure(figsize=(12, 4.8))
    plt.bar(range(len(S1)), S1[order] * 100)
    plt.xticks(range(len(S1)), [names[j] for j in order], rotation=60, ha="right")
    plt.ylabel("First-order Sobol index (%)")
    plt.title(f"Global sensitivity of Plate max intrusion at {tag} (PDD, S = 1)")
    plt.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "sensitivity_bar.png", dpi=200)
    plt.close(fig)

    top4 = order[:4]
    t = np.linspace(-1, 1, 400)
    fig = plt.figure(figsize=(8, 5))
    for j in top4:
        plt.plot(t, main_effect(c, j, m, t), lw=2, label=f"{names[j]} ({S1[j] * 100:.1f}%)")
    plt.xlabel("Scaled variable ξ [-1, 1]  (−1 = nominal −10 %, +1 = nominal +10 %)")
    plt.ylabel(Y_LABEL)
    plt.title(f"Top 4 variables: main effects ({tag})")
    plt.grid(alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out / "top4_main_effects.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, j in zip(axes.ravel(), top4):
        ax.scatter(X_mc[:, j], y_mc, s=1, alpha=0.2)
        ax.set_title(f"{names[j]} ({S1[j] * 100:.1f}%)")
        ax.set_xlim(-1, 1)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Scaled variable ξ [-1, 1]")
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_LABEL)
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(out / "top4_scatter.png", dpi=200)
    plt.close(fig)


def plot_by_energy(table: pd.DataFrame, path, n_top=6) -> None:
    """에너지 수준(가로) × 변수 Sobol 지수(세로): 어느 수준에서든 상위 n_top에 드는 변수."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = sorted({v for _, row in table.iterrows() for v in row.nlargest(n_top).index},
                 key=lambda v: -table[v].mean())
    fig = plt.figure(figsize=(9, 5))
    for v in top:
        plt.plot(table.index, table[v], "o-", lw=2, label=v)
    plt.xlabel("Absorbed energy (kJ)")
    plt.ylabel("First-order Sobol index (%)")
    plt.title("Sensitivity of Plate max intrusion vs energy level")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ─────────────────────────────── 에너지 수준 하나 ───────────────────────────────
def run_level(X, y, samples, status, names, a, E, out) -> dict:
    """에너지 수준 E (kJ) 하나에 대해 PDD 학습·민감도·그림. 반환: 요약 값."""
    N, n = X.shape[1], len(y)
    L = 1 + N * a.m
    tag = f"E = {E:g} kJ"
    print(f"\n[{tag}] 학습 {n}개 (계수 {L}개, 권장 {3 * L}개)")
    if n <= L:
        print(f"  [skip] 학습 샘플({n})이 계수 수({L})보다 많아야 합니다.")
        return {}
    if n < 2 * L:
        print(f"  [경고] 학습 샘플이 계수의 2배({2 * L}) 미만이라 surrogate가 불안정할 수 있습니다.")

    c, y_fit, y_loo = fit(X, y, a.m)
    S1, contrib, var, per_deg = sobol_first_order(c, N, a.m)

    rng = np.random.default_rng(a.seed)
    X_mc = rng.uniform(-1.0, 1.0, (a.n_mc, N))
    y_mc = basis(X_mc, a.m) @ c

    out.mkdir(parents=True, exist_ok=True)
    coef_names = ["c0"] + [f"{names[j]}_k{k}" for j in range(N) for k in range(1, a.m + 1)]
    pd.DataFrame(dict(term=coef_names, coef=c)).to_csv(out / "pdd_coefficients.csv", index=False,
                                                       float_format="%.8g")
    sens = pd.DataFrame(dict(variable=names, S1=S1, S1_pct=S1 * 100, var_contrib=contrib)).sort_values(
        "S1", ascending=False)
    sens.to_csv(out / "pdd_sensitivity.csv", index=False, float_format="%.6g")
    pd.DataFrame(dict(sample=samples, y=y, y_pdd=y_fit, y_loo=y_loo, status=status)).to_csv(
        out / "pdd_training_data.csv", index=False, float_format="%.6g")

    # 차수 비교 (참고): m = 1–4 LOOCV
    deg_lines = []
    for mm in range(1, 5):
        if n > 1 + N * mm:
            _, _, yl = fit(X, y, mm)
            deg_lines.append(f"  m = {mm} (계수 {1 + N * mm:3d}): LOOCV R² = {r2(y, yl):.4f}, "
                             f"RMSE = {np.sqrt(np.mean((y - yl) ** 2)):.4f} mm")

    rmse_loo = float(np.sqrt(np.mean((y - y_loo) ** 2)))
    n_snap = int((status == "snap_through").sum())
    lines = [
        f"output            : Plate intrusion, sec = {a.sec}  ({Y_LABEL}) at absorbed energy {E:g} kJ",
        f"PDD               : N = {N}, m = {a.m}, S = 1, coefficients = {L}, training samples = {n}"
        f"  (stable {n - n_snap}, snap_through {n_snap})",
        f"mean  (FEA / PDD c0 / MC) : {y.mean():.4f} / {c[0]:.4f} / {y_mc.mean():.4f}",
        f"var   (FEA / PDD Σc² / MC): {y.var(ddof=1):.4f} / {var:.4f} / {y_mc.var(ddof=1):.4f}",
        f"std   (FEA / PDD)         : {y.std(ddof=1):.4f} / {np.sqrt(var):.4f}",
        f"R² training / LOOCV       : {r2(y, y_fit):.4f} / {r2(y, y_loo):.4f}",
        f"RMSE LOOCV                : {rmse_loo:.4f} mm  ({100 * rmse_loo / y.mean():.2f} % of mean)",
        f"sum of S1                 : {S1.sum():.4f}",
        "degree share (k=1..m)     : " + ", ".join(f"{p * 100:.1f}%" for p in per_deg),
        "", "degree comparison (LOOCV, reference):", *deg_lines,
        "", "first-order Sobol indices (descending):", sens.to_string(index=False),
    ]
    (out / "pdd_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join("  " + s for s in lines[2:9]))
    print("\n".join(deg_lines))
    print(sens.head(6).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    make_plots(out, y, y_fit, y_loo, y_mc, X_mc, c, S1, names, a.m, tag)
    return dict(E_kJ=E, n=n, mean=y.mean(), std=y.std(ddof=1), r2_loo=r2(y, y_loo), rmse_loo=rmse_loo,
                **{v: s * 100 for v, s in zip(names, S1)})


# ─────────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=2, help="Legendre 최대 차수 (v21: LOOCV로 m=2 선택)")
    ap.add_argument("--sec", default="max", help='출력 섹션: "max"(전체 섹션 최대) 또는 0–14')
    ap.add_argument("--energies", type=float, nargs="+", default=None,
                    help="학습할 에너지 수준 (kJ, 기본: FEM 결과에 있는 모든 수준)")
    ap.add_argument("--n-mc", type=int, default=10000, help="surrogate 몬테카를로 샘플 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--design", default=str(DESIGN_CSV))
    ap.add_argument("--deform-dir", default=str(DEFORM_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()

    Xi, names = load_inputs(a.design)
    Y, ST = load_outputs(a.deform_dir, a.sec)
    energies = a.energies or list(Y.columns)
    print(f"[data] FEM 결과 {len(Y)}개, 변수 {len(names)}개, 에너지 수준 {[f'{E:g}' for E in energies]} kJ")

    root = Path(a.out_dir)
    summary = []
    for E in energies:
        yE = Y[E].dropna()
        common = yE.index.intersection(Xi.index)
        res = run_level(Xi.loc[common].to_numpy(), yE.loc[common].to_numpy(), common.to_numpy(),
                        ST.loc[common, E].to_numpy(), names, a, E, root / f"E{E:g}kJ")
        if res:
            summary.append(res)

    if summary:
        tab = pd.DataFrame(summary).set_index("E_kJ")
        tab.to_csv(root / "sensitivity_by_energy.csv", float_format="%.4f")
        if len(tab) > 1:
            plot_by_energy(tab[names], root / "sensitivity_by_energy.png")
        print("\n[에너지 수준별] 평균·표준편차 (mm), LOOCV R²")
        print(tab[["n", "mean", "std", "r2_loo", "rmse_loo"]].round(4).to_string())
    print(f"[done] → {root}")


if __name__ == "__main__":
    main()
