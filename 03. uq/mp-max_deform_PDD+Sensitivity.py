"""Mp 제1주성분 + t_Plate + t_Inner → B-pillar Plate 변형량(sec 8, sec 9 각각) PDD surrogate + 전역 민감도 분석.

사용 (conda env fem):
    python "mp-max_deform_PDD+Sensitivity.py"               # sec 8, 9 각각 surrogate, FEM이 끝난 샘플만 학습
    python "mp-max_deform_PDD+Sensitivity.py" --m 1         # 차수 1
    python "mp-max_deform_PDD+Sensitivity.py" --secs 8      # sec 8만

입력 X (3개)
    PC1     : results/mp/bpillar_sample_XXX_mp.csv 의 섹션 0–16 Mp를 표준화한 뒤의 제1주성분 점수
              (17개 Mp 분산의 약 98 %, 사실상 "전체 Mp 수준" ≈ t_Outer)
    t_Plate, t_Inner : sample/bpillar_samples_design.csv 의 ξ ([−1, 1], nominal ±10 %)
    Mp는 두께의 Outer 몫만 잘 담고 Plate·Inner 두께 효과는 거의 담지 못해 두 두께를 직접 추가했다
    (docs/review/mp-deformation_review.md 8절: 17개 Mp·PC1–3 입력보다 LOOCV가 확실히 좋음).
출력 y  : results/deformation/bpillar_sample_XXX_deformation.csv 의 sec 8, sec 9 Plate 침입량 (mm), 섹션마다 따로.
          FEM 결과가 없거나 not_reached / NaN / 읽기 실패인 샘플은 제외 (해석 진행 중에도 실행 가능)

PDD (S = 1, 1변수 분해)
    기저 : 변수마다 267개 전체 샘플의 경험분포에 대해 정규직교인 다항식 ψ_jk
           (표준화 z = (x − 평균)/표준편차의 단항식을 모멘트 행렬 Cholesky로 직교화)
    모델 : y ≈ c0 + Σ_j Σ_{k=1..m} c_jk ψ_jk(X_j)
    민감도: 입력끼리 약하게 상관(PC1–t_Inner ρ ≈ 0.3)되어 있어 공분산 분해(ANCOVA)를 쓴다.
           S_j = Cov(y_j, ŷ)/Var(ŷ) (Σ S_j = 1) = 구조 기여 Var(y_j)/Var(ŷ) + 상관 기여
    계수 수 = 1 + 3·m (m = 2 → 7)

출력 (results/mp-max_deform/hybrid_sec08/, hybrid_sec09/)
    pdd_coefficients.csv       계수 c0, c_jk
    pdd_sensitivity.csv        변수별 S, 구조 S^U, 상관 S^C (큰 순)
    pdd_summary.txt            학습 샘플 수, 평균·분산, R², LOOCV 오차, 입력 상관
    pdd_training_data.csv      학습 샘플 번호, y, PDD 예측, LOOCV 예측
    pdd_all_samples.csv        267개 전체 샘플의 입력과 surrogate 예측 (FEM 결과 없는 샘플 포함)
    fea_vs_surrogate_pdf.png   FEA vs surrogate(267개 전체 입력) 분포
    surrogate_fit.png          실측 vs 예측 (학습 / LOOCV)
    sensitivity_bar.png        변수별 민감도 (구조 + 상관)
    main_effects.png           변수별 주효과 곡선 + FEA 부분잔차
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                       # 03. uq
MP_DIR = HERE / "results" / "mp"
DESIGN_CSV = HERE / "sample" / "bpillar_samples_design.csv"
DEFORM_DIR = HERE / "results" / "deformation"
OUT_DIR = HERE / "results" / "mp-max_deform"
N_SEC = 17
OUT_SECS = (8, 9)
INPUTS = ("PC1", "t_Plate", "t_Inner")


def y_label(sec) -> str:
    return f"Plate intrusion, sec {sec} (mm)"


# ─────────────────────────────── 데이터 ───────────────────────────────
def sample_no(f: Path) -> int:
    return int(f.stem.split("_")[2])


def load_mp(mp_dir) -> pd.DataFrame:
    """샘플 번호 → 섹션 0–16 Mp (kN·m)."""
    rows = {}
    for f in sorted(Path(mp_dir).glob("bpillar_sample_*_mp.csv")):
        mp = pd.read_csv(f).sort_values("sec")["Mp_Nmm"].to_numpy()
        if len(mp) == N_SEC and np.isfinite(mp).all():
            rows[sample_no(f)] = mp / 1e6                    # N·mm → kN·m
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"Mp_sec{s:02d}" for s in range(N_SEC)]).sort_index()


def mp_pc1(mp: pd.DataFrame) -> tuple[pd.Series, np.ndarray, float]:
    """표준화 Mp의 제1주성분 점수. loading 합이 양수가 되도록 부호를 맞춘다 (PC1 증가 = Mp 전체 증가).
    반환: (PC1 점수, loading (17,), 설명 분산 비율)."""
    X = mp.to_numpy()
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    ev, V = np.linalg.eigh(np.corrcoef(X.T))
    v = V[:, -1] * (1.0 if V[:, -1].sum() >= 0 else -1.0)
    return pd.Series(Z @ v, index=mp.index, name="PC1"), v, float(ev[-1] / ev.sum())


def load_inputs(mp_dir, design_csv) -> tuple[pd.DataFrame, np.ndarray, float]:
    """샘플 번호 → [PC1, t_Plate, t_Inner]. Mp와 설계 CSV에 모두 있는 샘플만."""
    pc1, loading, share = mp_pc1(load_mp(mp_dir))
    d = pd.read_csv(design_csv).set_index("sample")
    thk = d[["xi_t_Plate", "xi_t_Inner"]].rename(columns=lambda c: c[3:])
    return pd.concat([pc1, thk], axis=1, join="inner")[list(INPUTS)].sort_index(), loading, share


def load_outputs(deform_dir, sec, column="plate_intrusion_mm") -> pd.Series:
    """샘플별 sec Plate 침입량. not_reached / NaN / 읽기 실패(쓰는 중)인 샘플은 제외."""
    y, skipped = {}, []
    for f in sorted(Path(deform_dir).glob("bpillar_sample_*_deformation.csv")):
        try:
            df = pd.read_csv(f, dtype={"sec": str})
            if (df["status"] == "not_reached").any():
                raise ValueError("not_reached")
            val = float(df.loc[df["sec"] == str(sec), column].iloc[0])
            if not np.isfinite(val):
                raise ValueError("NaN")
        except Exception as e:                                # 수렴 실패·빈 파일·열 누락 등
            skipped.append(f"{sample_no(f):03d}({type(e).__name__}: {e})")
            continue
        y[sample_no(f)] = val
    if skipped:
        print(f"[skip] {len(skipped)}개 제외: " + ", ".join(skipped))
    return pd.Series(y, name="y").sort_index()


# ─────────────────────────────── PDD ───────────────────────────────
class Marginal:
    """변수 하나의 경험분포에 대한 정규직교 다항식 ψ_1..ψ_m (ψ_0 = 1)."""

    def __init__(self, x: np.ndarray, m: int):
        self.mu, self.sd, self.m = float(x.mean()), float(x.std()), m
        z = (x - self.mu) / self.sd
        G = np.array([[np.mean(z ** (i + j)) for j in range(m + 1)] for i in range(m + 1)])  # 모멘트 행렬
        self.W = np.linalg.inv(np.linalg.cholesky(G))        # ψ = W · [1, z, …, z^m]  →  E[ψψᵀ] = I

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, float) - self.mu) / self.sd
        return (np.vander(z, self.m + 1, increasing=True) @ self.W.T)[:, 1:]   # (n, m), ψ_0 제외


def basis(X: np.ndarray, marg: list[Marginal]) -> np.ndarray:
    """S = 1 PDD 기저 행렬 (n, 1 + N·m): [1, ψ_11…ψ_1m, ψ_21…, …]."""
    return np.column_stack([np.ones(len(X))] + [mg(X[:, j]) for j, mg in enumerate(marg)])


def fit(A, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """최소제곱 계수, 학습 예측, LOOCV 예측 (hat 행렬로 한 번에: e_loo = e / (1 − h_ii))."""
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_fit = A @ c
    h = np.einsum("ij,ji->i", A, np.linalg.pinv(A))
    y_loo = y - (y - y_fit) / np.clip(1.0 - h, 1e-12, None)
    return c, y_fit, y_loo


def component(x, j, c, marg) -> np.ndarray:
    """성분함수 y_j(x) = Σ_k c_jk ψ_jk(x)."""
    m = marg[j].m
    return marg[j](x) @ c[1 + j * m: 1 + (j + 1) * m]


def ancova(Yc) -> tuple[np.ndarray, np.ndarray, float]:
    """공분산 분해: S_j = Cov(y_j, ŷ)/Var(ŷ), 구조 S_j^U = Var(y_j)/Var(ŷ). 반환 (S, S_U, Var(ŷ))."""
    yhat = Yc.sum(axis=1)
    var = float(yhat.var())
    Yd = Yc - Yc.mean(axis=0)
    S = (Yd * (yhat - yhat.mean())[:, None]).mean(axis=0) / var
    return S, Yd.var(axis=0) / var, var


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


def make_plots(out, X, y, y_fit, y_loo, X_all, y_all, c, marg, S, S_U, ylab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10})
    names = list(INPUTS)

    lo, hi = min(y.min(), y_all.min()), max(y.max(), y_all.max())
    x = np.linspace(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo), 512)
    fig = plt.figure(figsize=(9, 4.8))
    plt.hist(y, bins=max(10, len(y) // 10), density=True, alpha=0.35, label=f"FEA data (n = {len(y)})")
    plt.hist(y_all, bins=max(10, len(y_all) // 10), density=True, alpha=0.5,
             label=f"Surrogate, all samples (n = {len(y_all)})")
    plt.plot(x, kde(y, x), lw=1.5, label="FEA KDE")
    plt.plot(x, kde(y_all, x), lw=1.5, label="Surrogate KDE")
    plt.xlabel(ylab)
    plt.ylabel("Density")
    plt.title("Probability distribution: FEA vs PDD surrogate")
    plt.legend()
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fea_vs_surrogate_pdf.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, p, lab in zip(axes, (y_fit, y_loo), ("training fit", "leave-one-out")):
        ax.scatter(y, p, s=12)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1)
        ax.set_xlabel(f"FEA {ylab}")
        ax.set_ylabel("PDD prediction")
        ax.set_title(f"{lab}: R² = {r2(y, p):.3f}, RMSE = {np.sqrt(np.mean((y - p) ** 2)):.2f}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "surrogate_fit.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4.8))
    idx = np.arange(len(names))
    plt.bar(idx, S_U * 100, label="structural $S^U$")
    plt.bar(idx, (S - S_U) * 100, bottom=S_U * 100, label="correlative $S^C$")
    plt.plot(idx, S * 100, "k_", ms=18, mew=2, label="total $S$")
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks(idx, names)
    plt.ylabel("Sensitivity index (%)")
    plt.title(f"Sensitivity of {ylab.split(' (')[0]}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "sensitivity_bar.png", dpi=200)
    plt.close(fig)

    # 주효과: 곡선 c0 + y_j(x), 점 = 부분잔차 y − (다른 변수 성분) (학습 샘플)
    A = basis(X, marg)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.8), sharey=True)
    for j, ax in enumerate(axes):
        t = np.linspace(X_all[:, j].min(), X_all[:, j].max(), 300)
        own = component(X[:, j], j, c, marg)
        ax.scatter(X[:, j], y - (A @ c - c[0] - own), s=10, alpha=0.5, label="FEA partial residual")
        ax.plot(t, c[0] + component(t, j, c, marg), "r", lw=2.5, label="PDD main effect")
        ax.set_title(f"{names[j]} ({S[j] * 100:.1f}%)")
        ax.set_xlabel("PC1 score of standardized Mp" if j == 0 else f"ξ ({names[j]}), −1 = −10 %, +1 = +10 %")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylab)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "main_effects.png", dpi=200)
    plt.close(fig)


# ─────────────────────────────── main ───────────────────────────────
def run_section(sec, Xdf, marg, loading, share, m, deform_dir, out) -> None:
    """출력 섹션 하나: FEM 결과 로드 → PDD 학습 → 민감도 → 파일·그림 저장."""
    ylab = y_label(sec)
    print(f"\n===== sec {sec} =====")
    y_s = load_outputs(deform_dir, sec)
    common = y_s.index.intersection(Xdf.index)
    X_all = Xdf.to_numpy()
    X, y = Xdf.loc[common].to_numpy(), y_s.loc[common].to_numpy()
    N, n = X.shape[1], len(y)
    L = 1 + N * m
    print(f"[data] 입력 샘플 {len(Xdf)}개, FEM 결과 {len(y_s)}개 → 학습 {n}개 (계수 {L}개, 권장 {3 * L}개)")
    if n <= L:
        print(f"[skip] 학습 샘플({n})이 계수 수({L})보다 많아야 합니다. FEM 결과가 더 쌓인 뒤 실행하세요.")
        return

    c, y_fit, y_loo = fit(basis(X, marg), y)
    y_all = basis(X_all, marg) @ c
    S, S_U, var = ancova(np.column_stack([component(X_all[:, j], j, c, marg) for j in range(N)]))
    corr = np.corrcoef(X_all.T)

    out.mkdir(parents=True, exist_ok=True)
    coef_names = ["c0"] + [f"{INPUTS[j]}_k{k}" for j in range(N) for k in range(1, m + 1)]
    pd.DataFrame(dict(term=coef_names, coef=c)).to_csv(out / "pdd_coefficients.csv", index=False,
                                                       float_format="%.8g")
    sens = pd.DataFrame(dict(variable=INPUTS, S=S, S_pct=S * 100, S_structural_pct=S_U * 100,
                             S_correlative_pct=(S - S_U) * 100)).sort_values("S", ascending=False)
    sens.to_csv(out / "pdd_sensitivity.csv", index=False, float_format="%.6g")
    pd.DataFrame(dict(sample=common, y=y, y_pdd=y_fit, y_loo=y_loo)).to_csv(
        out / "pdd_training_data.csv", index=False, float_format="%.6g")
    Xdf.assign(y_pdd=y_all, y_fea=y_s.reindex(Xdf.index)).rename_axis("sample").to_csv(
        out / "pdd_all_samples.csv", float_format="%.6g")

    rmse_loo = float(np.sqrt(np.mean((y - y_loo) ** 2)))
    lines = [
        f"input             : PC1 of standardized section Mp ({100 * share:.1f} % of Mp variance, "
        f"loading {loading.min():.3f}–{loading.max():.3f}), t_Plate ξ, t_Inner ξ; {len(Xdf)} samples",
        f"output            : {ylab}",
        f"PDD               : N = {N}, m = {m}, S = 1, coefficients = {L}, training samples = {n}",
        f"mean  (FEA / PDD c0 / all-sample surrogate): {y.mean():.4f} / {c[0]:.4f} / {y_all.mean():.4f}",
        f"var   (FEA / all-sample surrogate)         : {y.var(ddof=1):.4f} / {var:.4f}",
        f"R² training / LOOCV       : {r2(y, y_fit):.4f} / {r2(y, y_loo):.4f}",
        f"RMSE LOOCV                : {rmse_loo:.4f} mm  ({100 * rmse_loo / y.mean():.2f} % of mean)",
        f"sum of S                  : {S.sum():.4f}  (structural Σ S^U = {S_U.sum():.4f})",
        "input correlation ρ       : " + ", ".join(
            f"{INPUTS[i]}–{INPUTS[j]} {corr[i, j]:+.3f}" for i in range(N) for j in range(i + 1, N)),
        "", "covariance-based sensitivity (descending):",
        sens.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
    ]
    (out / "pdd_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    make_plots(out, X, y, y_fit, y_loo, X_all, y_all, c, marg, S, S_U, ylab)
    print(f"[done] → {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=2, help="다항식 최대 차수")
    ap.add_argument("--secs", type=int, nargs="+", default=list(OUT_SECS), help="출력 섹션 (섹션마다 surrogate 하나)")
    ap.add_argument("--mp-dir", default=str(MP_DIR))
    ap.add_argument("--design", default=str(DESIGN_CSV), help="설계변수 ξ CSV (t_Plate, t_Inner)")
    ap.add_argument("--deform-dir", default=str(DEFORM_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()

    Xdf, loading, share = load_inputs(a.mp_dir, a.design)
    X_all = Xdf.to_numpy()
    marg = [Marginal(X_all[:, j], a.m) for j in range(X_all.shape[1])]   # 기저는 전체 입력 분포로
    for sec in a.secs:
        run_section(sec, Xdf, marg, loading, share, a.m, a.deform_dir, Path(a.out_dir) / f"hybrid_sec{sec:02d}")


if __name__ == "__main__":
    main()
