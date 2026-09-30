"""B-pillar UQ 샘플 단면 생성 — 22차원 라틴 하이퍼큐브 (PDD 학습용).

사용 (conda env fem):
    python bpillar_samples.py                      # 267개 = PDD(N=22, m=4, S=1) 계수 89개 × 3
    python bpillar_samples.py --m 4 --s 1 --oversample 3 --seed 2026

설계(불확실성) 변수 22개, 모두 기본값 × (1 ± 10 %) 균일분포
    x1–x5   : 파트 두께 t_Outer, t_Plate, t_Inner, t_Patch1, t_Patch2
    x6–x22  : 섹션 0–16별 Inner–Plate 거리 d_s
              = Inner 판 중심선과 바로 아래 Plate 판 중심선 사이 최대 수직 거리 (Inner hat 꼭대기 높이)

샘플 단면 생성 규칙 (Plate는 고정)
    1. 두께를 바꾼다.
    2. 플랜지 접합 유지: 접합 간격은 판 표면이 맞닿는 (t_a + t_b)/2 이므로, Plate에 붙은 파트
       (Outer, Inner, Patch1 위 / Patch2 아래)를 섹션마다 그 변화량만큼 통째로 위아래로 옮기고
       접합 point를 정확히 Plate + 새 간격에 맞춘다 → 접합 공차 0.
    3. Inner 거리: Inner 플랜지(pt 1–3, 16–18)는 그대로 두고, 몸체(pt 4–15)의 Plate 기준 높이를
       같은 비율로 늘이거나 줄여 꼭대기 높이를 d_s에 맞춘다.
    4. 겹침 방지: 쌓인 파트 쌍(위 판 중심선 − 아래 판 중심선 ≥ (t_위 + t_아래)/2)을 검사하고,
       어기는 point만 필요한 만큼 위로 올린다 (주로 Inner 플랜지 근처, 보정 수는 요약표에 기록).
    5. 검증: 접합 오차, 최소 간격, 실제 d_s가 목표와 같은지 확인한다.

PDD 샘플 수: 계수 개수 L = Σ_{k=0..S} C(N, k)·m^k  (S = 1이면 1 + N·m = 89),  샘플 = L × oversample

출력 (이 스크립트 폴더)
    bpillar_sample_001.csv …       샘플 단면 (입력과 같은 형식 — FEM 입력으로 바로 사용)
    bpillar_samples_design.csv     샘플별 22개 변수: 정규화 값 ξ ∈ [−1, 1] (Legendre PDD용)과 실제 값
    bpillar_samples_summary.csv    샘플별 검증: 접합 오차, 최소 간격, d_s 오차, 겹침 보정 point 수
    bpillar_samples_overview.png   기본 단면과 샘플 단면 (sec 0, 8, 16), 변수 분포 확인
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parents[1] / "02. post_processing" / "results" / "final_section_v1_21_0_x1.3_post.csv"

PART_NAMES = ["Outer", "Plate", "Inner", "Patch1", "Patch2"]
OUTER, PLATE, INNER, PATCH1, PATCH2 = range(5)
N_SEC = 17
# 플랜지 접합 (Plate point, 붙은 파트, 그 파트 point)
FLANGE_PAIRS = [
    (1, OUTER, 1), (2, OUTER, 2), (3, OUTER, 3), (28, OUTER, 28), (29, OUTER, 29), (30, OUTER, 30),
    (7, INNER, 1), (8, INNER, 2), (9, INNER, 3), (22, INNER, 16), (23, INNER, 17), (24, INNER, 18),
    (10, PATCH1, 1), (21, PATCH1, 12),
    (22, PATCH2, 1), (23, PATCH2, 2), (24, PATCH2, 3),
]
JOINT_PTS = {(PLATE, kp) for kp, _, _ in FLANGE_PAIRS} | {(pb, kb) for _, pb, kb in FLANGE_PAIRS}
INNER_BODY = np.arange(4, 16) - 1                 # Inner 몸체 point (0-base), 플랜지 pt 1–3, 16–18 제외
# 쌓인 파트 쌍 (위, 아래) — 겹침 검사·보정. 보정은 위 파트를 올린다 (Plate는 고정)
STACKED = [(PATCH1, PLATE), (INNER, PLATE), (INNER, PATCH1), (OUTER, PLATE), (OUTER, INNER)]
CLEAR_MARGIN = 0.05                               # 겹침 보정 시 남길 최소 여유 (mm)
REL_RANGE = 0.10                                  # 기본값 ± 10 %

VAR_NAMES = [f"t_{n}" for n in PART_NAMES] + [f"d_sec{s:02d}" for s in range(N_SEC)]


# ─────────────────────────────── 입출력 ───────────────────────────────
def read_parts(path) -> list[dict]:
    """CSV → [{name, t, xy (n_sec, n_pt, 2)}]. 파트 블록 끝 구분 행(point_idx = 0)에 두께."""
    df = pd.read_csv(path)
    df["block"] = (df["point_idx"] == 0).cumsum().shift(fill_value=0)
    parts = []
    for pid, blk in df.groupby("block"):
        data = blk[blk["point_idx"] > 0].sort_values(["floor_idx", "point_idx"])
        parts.append(dict(name=PART_NAMES[pid], t=float(blk.loc[blk["point_idx"] == 0, "t_mm"].iloc[0]),
                          xy=data[["x_mm", "y_mm"]].to_numpy().reshape(data["floor_idx"].nunique(), -1, 2)))
    return parts


def write_parts(parts, path) -> None:
    rows = []
    for p in parts:
        for s, sec in enumerate(p["xy"]):
            rows += [(float(s), float(k), x, y, 0.0, 0.0) for k, (x, y) in enumerate(sec, start=1)]
        rows.append((0.0, 0.0, 0.0, 0.0, 0.0, p["t"]))
    pd.DataFrame(rows, columns=["floor_idx", "point_idx", "x_mm", "y_mm", "r_mm", "t_mm"]).to_csv(
        path, index=False)


# ─────────────────────────────── 기하 도우미 ───────────────────────────────
def surface_y(poly: np.ndarray, x: np.ndarray) -> np.ndarray:
    """폴리라인 poly 위에서 x 위치의 y (x 범위 밖은 NaN; 여러 선분이 걸리면 가장 높은 값)."""
    x = np.atleast_1d(x)[:, None]
    x1, x2, y1, y2 = poly[:-1, 0], poly[1:, 0], poly[:-1, 1], poly[1:, 1]
    inside = (x >= np.minimum(x1, x2)) & (x <= np.maximum(x1, x2)) & (np.abs(x2 - x1) > 1e-12)
    y = y1 + (x - x1) / np.where(inside, x2 - x1, 1.0) * (y2 - y1)
    out = np.where(inside, y, -np.inf).max(axis=1)
    return np.where(np.isfinite(out), out, np.nan)


def inner_distance(parts, s) -> float:
    """섹션 s의 Inner–Plate 거리 d_s = Inner 몸체 point의 (y − 같은 x의 Plate y) 최대값."""
    body = parts[INNER]["xy"][s, INNER_BODY]
    return float(np.nanmax(body[:, 1] - surface_y(parts[PLATE]["xy"][s], body[:, 0])))


def clearances(parts, s):
    """(위 파트, point 번호, 간격) 목록: 간격 = 위 y − 아래 y − (t_위 + t_아래)/2 (접합 point 제외)."""
    out = []
    for up, lo in STACKED:
        if s >= len(parts[up]["xy"]) or s >= len(parts[lo]["xy"]):
            continue
        a = parts[up]["xy"][s]
        yl = surface_y(parts[lo]["xy"][s], a[:, 0])
        half = 0.5 * (parts[up]["t"] + parts[lo]["t"])
        for k in range(len(a)):
            if (up, k + 1) not in JOINT_PTS and np.isfinite(yl[k]):
                out.append((up, k, a[k, 1] - yl[k] - half))
    return out


# ─────────────────────────────── 샘플 단면 생성 ───────────────────────────────
def make_sample(base, t_new, d_new) -> tuple[list[dict], int]:
    """기본 단면 base에 두께 t_new(5)와 Inner 거리 d_new(17)를 적용. 반환: (단면, 겹침 보정 point 수)."""
    parts = [dict(p, xy=p["xy"].copy()) for p in base]
    t_old = [p["t"] for p in base]
    for p, t in zip(parts, t_new):
        p["t"] = float(t)

    # 2. 플랜지 접합: 붙은 파트를 새 접합 간격만큼 옮기고 접합 point를 정확히 맞춘다
    plate = parts[PLATE]["xy"]
    for pid in (OUTER, INNER, PATCH1, PATCH2):
        pairs = [(kp, kb) for kp, pb, kb in FLANGE_PAIRS if pb == pid]
        n = len(parts[pid]["xy"])
        old = np.array([base[pid]["xy"][:, kb - 1, 1] - plate[:n, kp - 1, 1] for kp, kb in pairs])  # (pair, sec)
        sign = np.sign(old.mean())                                      # 위(+) / 아래(−)
        new_gap = sign * 0.5 * (t_new[PLATE] + t_new[pid])
        parts[pid]["xy"][:, :, 1] += (new_gap - old.mean(axis=0))[:, None]
        for kp, kb in pairs:
            parts[pid]["xy"][:, kb - 1] = plate[:n, kp - 1] + np.array([0.0, new_gap])

    # 3. Inner 거리: 몸체의 Plate 기준 높이를 비율로 조정 (플랜지 고정)
    for s in range(N_SEC):
        body = parts[INNER]["xy"][s, INNER_BODY]
        yp = surface_y(plate[s], body[:, 0])
        gap = body[:, 1] - yp
        parts[INNER]["xy"][s, INNER_BODY, 1] = yp + gap * (d_new[s] / np.nanmax(gap))

    # 4. 겹침 방지: 어기는 point만 위로 올린다 (아래 파트부터 순서대로)
    repaired = 0
    for s in range(N_SEC):
        for _ in range(3):                                              # 올린 파트가 다음 쌍에 영향 → 반복
            bad = [(up, k, c) for up, k, c in clearances(parts, s) if c < CLEAR_MARGIN - 1e-9]
            if not bad:
                break
            for up, k, c in bad:
                parts[up]["xy"][s, k, 1] += CLEAR_MARGIN - c
                repaired += 1
    return parts, repaired


def validate(parts, d_target) -> dict:
    plate = parts[PLATE]["xy"]
    joint_err = max(abs(abs(parts[pb]["xy"][:, kb - 1, 1] - plate[:len(parts[pb]["xy"]), kp - 1, 1])
                        - 0.5 * (parts[PLATE]["t"] + parts[pb]["t"])).max() for kp, pb, kb in FLANGE_PAIRS)
    min_clear = min(c for s in range(N_SEC) for _, _, c in clearances(parts, s))
    d_err = max(abs(inner_distance(parts, s) - d_target[s]) for s in range(N_SEC))
    return dict(joint_err_mm=joint_err, min_clearance_mm=min_clear, d_err_mm=d_err)


# ─────────────────────────────── 라틴 하이퍼큐브 ───────────────────────────────
def lhs(n, dim, rng, candidates=50) -> np.ndarray:
    """[0, 1]^dim 라틴 하이퍼큐브 n점. candidates개 중 최소 거리가 가장 큰 것(maximin)을 고른다."""
    best, best_d = None, -1.0
    for _ in range(candidates):
        u = (np.argsort(rng.random((n, dim)), axis=0) + rng.random((n, dim))) / n   # 층마다 1점
        d = np.sqrt(((u[:, None] - u[None]) ** 2).sum(-1))
        d_min = d[np.triu_indices(n, 1)].min()
        if d_min > best_d:
            best, best_d = u, d_min
    return best


def pdd_sample_count(N, m, S, oversample) -> tuple[int, int]:
    """PDD 계수 개수 L = Σ_{k=0..S} C(N, k)·m^k 와 샘플 수 L × oversample."""
    L = sum(comb(N, k) * m ** k for k in range(S + 1))
    return L, L * oversample


def plot_overview(base, samples, design, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(24, 5), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1.3, 1]))
    for ax, s in zip(axes, (0, 8, 16)):
        for smp in samples[:30]:
            for pid, p in enumerate(smp):
                if s < len(p["xy"]):
                    ax.plot(*p["xy"][s].T, "-", color=f"C{pid}", lw=0.4, alpha=0.35)
        for pid, p in enumerate(base):
            if s < len(p["xy"]):
                ax.plot(*p["xy"][s].T, "k-", lw=1.0, label="nominal" if pid == 0 else None)
        ax.set_title(f"sec {s}: nominal (black) + first 30 samples", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[3].scatter(design["xi_t_Outer"], design["xi_d_sec08"], s=6)
    axes[3].set_xlabel("ξ t_Outer")
    axes[3].set_ylabel("ξ d_sec08")
    axes[3].set_title("LHS projection (2 of 22 variables)", fontsize=10)
    axes[3].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ─────────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="기본(명목) 단면 CSV")
    ap.add_argument("--out-dir", default=str(HERE), help="출력 폴더")
    ap.add_argument("--m", type=int, default=4, help="PDD 다항식 차수")
    ap.add_argument("--s", type=int, default=1, help="PDD 분해 차수 (S = 1: 1변수 성분)")
    ap.add_argument("--oversample", type=int, default=3, help="샘플 수 = PDD 계수 수 × oversample")
    ap.add_argument("--seed", type=int, default=2026, help="난수 시드 (같은 시드 = 같은 샘플)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = read_parts(a.csv)
    t_nom = np.array([p["t"] for p in base])
    d_nom = np.array([inner_distance(base, s) for s in range(N_SEC)])
    nominal = np.r_[t_nom, d_nom]
    N = len(nominal)
    L, n = pdd_sample_count(N, a.m, a.s, a.oversample)
    print(f"[PDD] N={N}, m={a.m}, S={a.s} → 계수 {L}개 × {a.oversample} = 샘플 {n}개")

    rng = np.random.default_rng(a.seed)
    xi = 2.0 * lhs(n, N, rng) - 1.0                                   # ξ ∈ [−1, 1]
    X = nominal * (1.0 + REL_RANGE * xi)                              # 실제 값 = 기본값 × (1 ± 10 %)

    width = max(2, len(str(n)))
    rows, samples = [], []
    for i in range(n):
        parts, repaired = make_sample(base, X[i, :5], X[i, 5:])
        chk = validate(parts, X[i, 5:])
        fname = f"bpillar_sample_{i + 1:0{width}d}.csv"
        write_parts(parts, out_dir / fname)
        rows.append(dict(sample=i + 1, file=fname, repaired_points=repaired, **chk))
        samples.append(parts)

    design = pd.DataFrame(xi, columns=[f"xi_{v}" for v in VAR_NAMES])
    for j, v in enumerate(VAR_NAMES):
        design[v] = X[:, j]
    design.insert(0, "file", [r["file"] for r in rows])
    design.insert(0, "sample", np.arange(1, n + 1))
    design.to_csv(out_dir / "bpillar_samples_design.csv", index=False, float_format="%.6f")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "bpillar_samples_summary.csv", index=False, float_format="%.6g")
    pd.DataFrame(dict(variable=VAR_NAMES, nominal=nominal, lower=nominal * (1 - REL_RANGE),
                      upper=nominal * (1 + REL_RANGE))).to_csv(
        out_dir / "bpillar_samples_variables.csv", index=False, float_format="%.6f")
    plot_overview(base, samples, design, out_dir / "bpillar_samples_overview.png")

    print(f"[done] 샘플 {n}개 → {out_dir}")
    print(f"  접합 오차 최대 {summary['joint_err_mm'].max():.2e} mm (공차 0.01 mm)")
    print(f"  최소 간격 {summary['min_clearance_mm'].min():.3f} mm (≥ 0 이면 겹침 없음)")
    print(f"  d_s 오차 최대 {summary['d_err_mm'].max():.2e} mm")
    print(f"  겹침 보정: {(summary['repaired_points'] > 0).sum()}개 샘플, "
          f"샘플당 최대 {summary['repaired_points'].max()} point")


if __name__ == "__main__":
    main()
