"""Mp 제1주성분 + t_Plate + t_Inner → B-pillar Plate 변형량(sec 8, sec 9 각각) PDD surrogate + 전역 민감도 분석
(v23 — 에너지 기준 FEM 결과, 섹션 0–14).

사용 (conda env fem):
    python "mp-max_deform_PDD+Sensitivity_v23.py"                 # 에너지 수준 × sec 8, 9 각각 surrogate
    python "mp-max_deform_PDD+Sensitivity_v23.py" --energies 20   # 20 kJ(기준값)만
    python "mp-max_deform_PDD+Sensitivity_v23.py" --m 1 --secs 8  # 차수 1, sec 8만

입력 X (3개)
    PC1     : results/v23/mp/bpillar_sample_XXX_mp.csv 의 섹션 0–14 Mp를 표준화한 뒤의 제1주성분 점수
              (v21에서는 Mp 분산의 약 98 %, 사실상 "전체 Mp 수준" ≈ t_Outer; v23 값은 요약과
              pc_vs_design_correlation.csv 에 적는다)
    t_Plate, t_Inner : sample/v23/bpillar_samples_design.csv 의 ξ ([−1, 1], nominal ±10 %)
    Mp는 두께의 Outer 몫만 잘 담고 Plate·Inner 두께 효과는 거의 담지 못해 두 두께를 직접 추가했다
    (docs/review/mp-deformation_review.md 8절: 17개 Mp·PC1–3 입력보다 LOOCV가 확실히 좋음).
출력 y  : results/v23/deformation/bpillar_sample_XXX_deformation.csv 의 sec 8, sec 9 Plate 침입량 (mm).
          에너지 수준(E_kJ = 10/15/20/25/30, 외력 일이 그 값에 처음 닿은 상태) × 섹션마다 따로 학습한다
          (v23에서도 최대 침입 섹션은 모든 수준·샘플에서 8 또는 9).
          읽기 실패 샘플과 not_reached / NaN인 (샘플, 에너지)는 제외 (해석 진행 중에도 실행 가능).
          stable / snap_through는 같은 에너지에서 읽은 값이라 구분 없이 함께 쓴다.

PDD (S = 1, 1변수 분해)
    기저 : 변수마다 267개 전체 샘플의 경험분포에 대해 정규직교인 다항식 ψ_jk
           (표준화 z = (x − 평균)/표준편차의 단항식을 모멘트 행렬 Cholesky로 직교화)
    모델 : y ≈ c0 + Σ_j Σ_{k=1..m} c_jk ψ_jk(X_j)
    민감도: 입력끼리 약하게 상관(PC1–t_Inner ρ ≈ 0.3)되어 있어 공분산 분해(ANCOVA)를 쓴다.
           S_j = Cov(y_j, ŷ)/Var(ŷ) (Σ S_j = 1) = 구조 기여 Var(y_j)/Var(ŷ) + 상관 기여
    계수 수 = 1 + 3·m (m = 2 → 7)

출력 (results/v23/mp-max_deform/E{에너지}kJ/hybrid_sec08/, hybrid_sec09/ — 에너지 수준 × 섹션마다 한 벌)
    pdd_coefficients.csv       계수 c0, c_jk
    pdd_sensitivity.csv        변수별 S, 구조 S^U, 상관 S^C (큰 순)
    pdd_summary.txt            학습 샘플 수, 평균·분산, R², LOOCV 오차, 입력 상관
    pdd_training_data.csv      학습 샘플 번호, y, PDD 예측, LOOCV 예측, status
    pdd_all_samples.csv        267개 전체 샘플의 입력과 surrogate 예측 (FEM 결과 없는 샘플 포함)
    fea_vs_surrogate_pdf.png   FEA vs surrogate(267개 전체 입력) 분포
    surrogate_fit.png          실측 vs 예측 (학습 / LOOCV)
    sensitivity_bar.png        변수별 민감도 (구조 + 상관)
    main_effects.png           변수별 주효과 곡선 + FEA 부분잔차
(results/v23/mp-max_deform/ 바로 아래)
    pc_vs_design_correlation.csv  Mp PC1–3 점수와 설계변수 ξ의 상관 (PC1 ≈ t_Outer 가정 확인용)
    sensitivity_by_energy.csv     섹션 × 에너지 수준별 S (%), 평균·표준편차, LOOCV R²·RMSE
    sensitivity_by_energy.png     에너지 수준에 따른 변수별 S (섹션마다 한 칸)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                       # 03. uq
MP_DIR = HERE / "results" / "v23" / "mp"
DESIGN_CSV = HERE / "sample" / "v23" / "bpillar_samples_design.csv"
DEFORM_DIR = HERE / "results" / "v23" / "deformation"
OUT_DIR = HERE / "results" / "v23" / "mp-max_deform"
N_SEC = 15
OUT_SECS = (8, 9)
INPUTS = ("PC1", "t_Plate", "t_Inner")


def y_label(sec) -> str:
    return f"Plate intrusion, sec {sec} (mm)"


# ─────────────────────────────── 데이터 ───────────────────────────────
def sample_no(f: Path) -> int:
    return int(f.stem.split("_")[2])


def load_mp(mp_dir) -> pd.DataFrame:
    """샘플 번호 → 섹션 0–14 Mp (kN·m)."""
    rows = {}
    for f in sorted(Path(mp_dir).glob("bpillar_sample_*_mp.csv")):
        mp = pd.read_csv(f).sort_values("sec")["Mp_Nmm"].to_numpy()
        if len(mp) == N_SEC and np.isfinite(mp).all():
            rows[sample_no(f)] = mp / 1e6                    # N·mm → kN·m
    return pd.DataFrame.from_dict(rows, orient="index", columns=[f"Mp_sec{s:02d}" for s in range(N_SEC)]).sort_index()


def mp_pcs(mp: pd.DataFrame, k=3) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """표준화 Mp의 주성분 점수 PC1..PCk. loading 합이 양수가 되도록 부호를 맞춘다 (PC1 증가 = Mp 전체 증가).
    반환: (점수 표, loading (N_SEC, k), 설명 분산 비율 (k,))."""
    X = mp.to_numpy()
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    ev, V = np.linalg.eigh(np.corrcoef(X.T))
    ev, V = ev[::-1][:k], V[:, ::-1][:, :k]
    V = V * np.where(V.sum(axis=0) >= 0, 1.0, -1.0)
    return pd.DataFrame(Z @ V, index=mp.index, columns=[f"PC{i + 1}" for i in range(k)]), V, ev / len(X.T)


def load_inputs(mp_dir, design_csv, out) -> tuple[pd.DataFrame, np.ndarray, float]:
    """샘플 번호 → [PC1, t_Plate, t_Inner]. Mp와 설계 CSV에 모두 있는 샘플만.
    PC1–3과 설계변수 ξ의 상관을 out/pc_vs_design_correlation.csv 로 남긴다."""
    pcs, V, share = mp_pcs(load_mp(mp_dir))
    d = pd.read_csv(design_csv).set_index("sample")
    xi = d.filter(regex="^xi_").rename(columns=lambda c: c[3:])
    both = pcs.index.intersection(xi.index)
    corr = pd.DataFrame({p: xi.loc[both].corrwith(pcs.loc[both, p]) for p in pcs.columns})
    out.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out / "pc_vs_design_correlation.csv", float_format="%.4f")
    print(f"[Mp PCA] 설명 분산 PC1–3 = {', '.join(f'{100 * s:.1f} %' for s in share)}; "
          f"PC1과 상관 큰 설계변수: " + ", ".join(f"{v} {r:+.3f}" for v, r in
                                         corr["PC1"].reindex(corr["PC1"].abs().nlargest(3).index).items()))
    Xdf = pd.concat([pcs["PC1"], xi[["t_Plate", "t_Inner"]]], axis=1, join="inner")[list(INPUTS)].sort_index()
    return Xdf, V[:, 0], float(share[0])


def load_outputs(deform_dir, sec, column="plate_intrusion_mm") -> tuple[pd.DataFrame, pd.DataFrame]:
    """샘플별 FEM 결과 → (sample × E_kJ) sec Plate 침입량 표와 status 표.
    읽기 실패(쓰는 중·빈 파일)인 샘플은 제외, not_reached / NaN인 (샘플, 에너지)는 NaN."""
    y, status, skipped = {}, {}, []
    for f in sorted(Path(deform_dir).glob("bpillar_sample_*_deformation.csv")):
        try:
            df = pd.read_csv(f, dtype={"sec": str})
            rows = df[df["sec"] == str(sec)].set_index("E_kJ")
            if rows.empty:
                raise ValueError(f"sec {sec} 없음")
        except Exception as e:                                # 빈 파일·열 누락 등
            skipped.append(f"{sample_no(f):03d}({type(e).__name__}: {e})")
            continue
        y[sample_no(f)] = rows[column].where(rows["status"] != "not_reached")
        status[sample_no(f)] = rows["status"]
    if skipped:
        print(f"[skip] {len(skipped)}개 제외: " + ", ".join(skipped))
    return pd.DataFrame(y).T.sort_index(), pd.DataFrame(status).T.sort_index()


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


def make_plots(out, X, y, y_fit, y_loo, X_all, y_all, c, marg, S, S_U, ylab, tag):
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
    plt.title(f"Probability distribution at {tag}: FEA vs PDD surrogate")
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
    fig.suptitle(tag)
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
    plt.title(f"Sensitivity of {ylab.split(' (')[0]} at {tag}")
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
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(out / "main_effects.png", dpi=200)
    plt.close(fig)


def plot_by_energy(tab: pd.DataFrame, path) -> None:
    """에너지 수준(가로) × 변수별 S (세로), 섹션마다 한 칸."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    secs = sorted(tab["sec"].unique())
    fig, axes = plt.subplots(1, len(secs), figsize=(6.5 * len(secs), 4.8), sharey=True, squeeze=False)
    for ax, sec in zip(axes[0], secs):
        t = tab[tab["sec"] == sec].sort_values("E_kJ")
        for v in INPUTS:
            ax.plot(t["E_kJ"], t[f"S_{v}_pct"], "o-", lw=2, label=v)
        ax.set_title(f"Plate intrusion, sec {sec}")
        ax.set_xlabel("Absorbed energy (kJ)")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Covariance-based sensitivity S (%)")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ─────────────────────────────── 섹션 × 에너지 하나 ───────────────────────────────
def run_case(sec, E, Xdf, y_s, st, marg, loading, share, m, out) -> dict:
    """출력 섹션 sec, 에너지 수준 E (kJ) 하나: PDD 학습 → 민감도 → 파일·그림 저장. 반환: 요약 값."""
    ylab, tag = y_label(sec), f"E = {E:g} kJ"
    common = y_s.index.intersection(Xdf.index)
    X_all = Xdf.to_numpy()
    X, y = Xdf.loc[common].to_numpy(), y_s.loc[common].to_numpy()
    N, n = X.shape[1], len(y)
    L = 1 + N * m
    print(f"\n===== sec {sec}, {tag} ===== 입력 샘플 {len(Xdf)}개, FEM 결과 {len(y_s)}개 → 학습 {n}개 "
          f"(계수 {L}개, 권장 {3 * L}개)")
    if n <= L:
        print(f"[skip] 학습 샘플({n})이 계수 수({L})보다 많아야 합니다. FEM 결과가 더 쌓인 뒤 실행하세요.")
        return {}

    c, y_fit, y_loo = fit(basis(X, marg), y)
    y_all = basis(X_all, marg) @ c
    S, S_U, var = ancova(np.column_stack([component(X_all[:, j], j, c, marg) for j in range(N)]))
    corr = np.corrcoef(X_all.T)
    status = st.reindex(common).to_numpy()
    n_snap = int((status == "snap_through").sum())

    out.mkdir(parents=True, exist_ok=True)
    coef_names = ["c0"] + [f"{INPUTS[j]}_k{k}" for j in range(N) for k in range(1, m + 1)]
    pd.DataFrame(dict(term=coef_names, coef=c)).to_csv(out / "pdd_coefficients.csv", index=False,
                                                       float_format="%.8g")
    sens = pd.DataFrame(dict(variable=INPUTS, S=S, S_pct=S * 100, S_structural_pct=S_U * 100,
                             S_correlative_pct=(S - S_U) * 100)).sort_values("S", ascending=False)
    sens.to_csv(out / "pdd_sensitivity.csv", index=False, float_format="%.6g")
    pd.DataFrame(dict(sample=common, y=y, y_pdd=y_fit, y_loo=y_loo, status=status)).to_csv(
        out / "pdd_training_data.csv", index=False, float_format="%.6g")
    Xdf.assign(y_pdd=y_all, y_fea=y_s.reindex(Xdf.index)).rename_axis("sample").to_csv(
        out / "pdd_all_samples.csv", float_format="%.6g")

    rmse_loo = float(np.sqrt(np.mean((y - y_loo) ** 2)))
    lines = [
        f"input             : PC1 of standardized section Mp ({100 * share:.1f} % of Mp variance, "
        f"loading {loading.min():.3f}–{loading.max():.3f}), t_Plate ξ, t_Inner ξ; {len(Xdf)} samples",
        f"output            : {ylab} at absorbed energy {E:g} kJ",
        f"PDD               : N = {N}, m = {m}, S = 1, coefficients = {L}, training samples = {n}"
        f"  (stable {n - n_snap}, snap_through {n_snap})",
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
    print("\n".join(lines[2:]))

    make_plots(out, X, y, y_fit, y_loo, X_all, y_all, c, marg, S, S_U, ylab, tag)
    return dict(sec=sec, E_kJ=E, n=n, n_snap=n_snap, mean=y.mean(), std=y.std(ddof=1),
                r2_train=r2(y, y_fit), r2_loo=r2(y, y_loo), rmse_loo=rmse_loo,
                **{f"S_{v}_pct": s * 100 for v, s in zip(INPUTS, S)},
                **{f"SU_{v}_pct": s * 100 for v, s in zip(INPUTS, S_U)})


# ─────────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=2, help="다항식 최대 차수")
    ap.add_argument("--secs", type=int, nargs="+", default=list(OUT_SECS), help="출력 섹션 (섹션마다 surrogate 하나)")
    ap.add_argument("--energies", type=float, nargs="+", default=None,
                    help="학습할 에너지 수준 (kJ, 기본: FEM 결과에 있는 모든 수준)")
    ap.add_argument("--mp-dir", default=str(MP_DIR))
    ap.add_argument("--design", default=str(DESIGN_CSV), help="설계변수 ξ CSV (t_Plate, t_Inner)")
    ap.add_argument("--deform-dir", default=str(DEFORM_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()

    root = Path(a.out_dir)
    Xdf, loading, share = load_inputs(a.mp_dir, a.design, root)
    X_all = Xdf.to_numpy()
    marg = [Marginal(X_all[:, j], a.m) for j in range(X_all.shape[1])]   # 기저는 전체 입력 분포로
    summary = []
    for sec in a.secs:
        Y, ST = load_outputs(a.deform_dir, sec)
        for E in a.energies or list(Y.columns):
            res = run_case(sec, E, Xdf, Y[E].dropna(), ST[E], marg, loading, share, a.m,
                           root / f"E{E:g}kJ" / f"hybrid_sec{sec:02d}")
            if res:
                summary.append(res)

    if summary:
        tab = pd.DataFrame(summary)
        tab.to_csv(root / "sensitivity_by_energy.csv", index=False, float_format="%.4f")
        if tab["E_kJ"].nunique() > 1:
            plot_by_energy(tab, root / "sensitivity_by_energy.png")
        print("\n[섹션 × 에너지 수준별] 평균·표준편차 (mm), LOOCV R², S (%)")
        print(tab[["sec", "E_kJ", "n", "mean", "std", "r2_loo", "rmse_loo"]
                  + [f"S_{v}_pct" for v in INPUTS]].round(3).to_string(index=False))
    print(f"[done] → {root}")


if __name__ == "__main__":
    main()
