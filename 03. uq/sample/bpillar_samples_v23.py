"""B-pillar UQ 샘플 단면 생성 (v23) — 22차원 라틴 하이퍼큐브 (PDD 학습용).

사용 (conda env fem):
    python bpillar_samples_v23.py                  # 267개 = PDD(N=22, m=4, S=1) 계수 89개 × 3 → ./v23
    python bpillar_samples_v23.py --m 4 --s 1 --oversample 3 --seed 2026

기본 단면: 02. post_processing/results/v23/final_section_v23_ref_post.csv
    7파트 (Outer, Plate, Inner, Patch1, Patch2, Patch3, Patch4), 15섹션.
    일부 섹션에만 있는 파트: Patch1·Patch2·Patch3 = sec 11–14, Patch4 = sec 0,1,5,6,8–14 (없는 섹션은 내부적으로 NaN)

설계(불확실성) 변수 22개, 모두 기본값 × (1 ± 10 %) 균일분포 (서로 독립)
    x1–x7   : 파트 두께 t_Outer, t_Plate, t_Inner, t_Patch1, t_Patch2, t_Patch3, t_Patch4
    x8–x22  : 섹션 0–14별 Inner–Plate 거리 d_s
              = Inner 판 중심선과 바로 아래 Plate 판 중심선 사이 최대 수직 거리 (Inner hat 꼭대기 높이)

샘플 단면 생성 규칙 (Plate는 고정)
    1. 두께를 바꾼다.
    2. 플랜지 접합 유지 (Outer, Inner): 접합 간격 = 판 표면이 맞닿는 (t_Plate + t_파트)/2.
       파트를 섹션마다 그 변화량만큼 통째로 위아래로 옮기고, 접합 point를 정확히 Plate + 새 간격에 맞춘다.
    3. 밀착 덧판 (Patch1·Patch2·Patch3 → Plate, Patch4 → Outer): 기준 파트 point와의 오프셋 벡터를
       (t_기준 + t_덧판)/2 의 변화 비율만큼 늘이거나 줄인다. 오프셋이 판 법선 방향 접촉 간격이므로
       경사 구간에서도 밀착이 유지된다 (플랜지 접합인 Patch2·Patch3는 2와 같은 결과).
       Patch4는 2에서 옮겨진 새 Outer를 따라간다.
    4. Inner 거리: Inner 플랜지(pt 1–3, 16–18)는 그대로 두고, 몸체(pt 4–15)의 Plate 기준 높이를
       같은 비율로 늘이거나 줄여 꼭대기 높이를 d_s에 맞춘다.
    5. 겹침 방지: 밀착 덧판이 아닌 쌓인 파트 쌍(위 판 중심선 − 아래 판 중심선 ≥ (t_위 + t_아래)/2)을
       검사하고, 어기는 point만 필요한 만큼 위로 올린다 (보정 수는 요약표에 기록).
    6. 검증: 접합·밀착 오차, 최소 간격, 실제 d_s가 목표와 같은지 확인한다.

PDD 샘플 수: 계수 개수 L = Σ_{k=0..S} C(N, k)·m^k  (S = 1이면 1 + N·m = 89),  샘플 = L × oversample

출력 (--out-dir, 기본 이 스크립트 폴더의 v23/)
    bpillar_sample_001.csv …       샘플 단면 (입력과 같은 형식 — FEM 입력으로 바로 사용)
    bpillar_samples_design.csv     샘플별 22개 변수: 정규화 값 ξ ∈ [−1, 1] (Legendre PDD용)과 실제 값
    bpillar_samples_summary.csv    샘플별 검증: 접합·밀착 오차, 최소 간격, d_s 오차, 겹침 보정 point 수
    bpillar_samples_variables.csv  변수별 기본값·하한·상한
    bpillar_samples_overview.png   기본 단면과 샘플 단면 (sec 0, 7, 14), LHS 투영
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parents[1] / "02. post_processing" / "results" / "v23" / "final_section_v23_ref_post.csv"
DEFAULT_OUT = HERE / "v23"

PART_NAMES = ["Outer", "Plate", "Inner", "Patch1", "Patch2", "Patch3", "Patch4"]
OUTER, PLATE, INNER, PATCH1, PATCH2, PATCH3, PATCH4 = range(7)
N_PART = len(PART_NAMES)
# 플랜지 접합 (Plate point, 붙은 파트, 그 파트 point) — 파트를 통째로 옮기는 접합 (Outer, Inner)
FLANGE_PAIRS = [
    (1, OUTER, 1), (2, OUTER, 2), (3, OUTER, 3), (28, OUTER, 28), (29, OUTER, 29), (30, OUTER, 30),
    (7, INNER, 1), (8, INNER, 2), (9, INNER, 3), (22, INNER, 16), (23, INNER, 17), (24, INNER, 18),
]
# 밀착 덧판 (덧판, 기준 파트, 기준 파트의 첫 point 번호) — 덧판 point k ↔ 기준 point (첫+k−1)
FOLLOWERS = [(PATCH1, PLATE, 10), (PATCH2, PLATE, 22), (PATCH3, PLATE, 7), (PATCH4, OUTER, 9)]
JOINT_PTS = {(PLATE, kp) for kp, _, _ in FLANGE_PAIRS} | {(pb, kb) for _, pb, kb in FLANGE_PAIRS}
INNER_BODY = np.arange(4, 16) - 1                 # Inner 몸체 point (0-base), 플랜지 pt 1–3, 16–18 제외
# 쌓인 파트 쌍 (위, 아래) — 겹침 검사·보정 (밀착 덧판 쌍은 3에서 간격이 정해지므로 제외). 보정은 위 파트를 올린다
STACKED = [(INNER, PLATE), (INNER, PATCH1), (OUTER, PLATE), (OUTER, INNER), (PATCH4, INNER)]
CLEAR_MARGIN = 0.05                               # 겹침 보정 시 남길 최소 여유 (mm)
REL_RANGE = 0.10                                  # 기본값 ± 10 %


# ─────────────────────────────── 입출력 ───────────────────────────────
def read_parts(path) -> list[dict]:
    """CSV → [{name, t, xy (n_sec, n_pt, 2)}]. 파트 블록 끝 구분 행(point_idx = 0)에 두께.
    파트가 없는 섹션의 xy는 NaN."""
    df = pd.read_csv(path)
    df["block"] = (df["point_idx"] == 0).cumsum().shift(fill_value=0)
    n_sec = int(df.loc[df["point_idx"] > 0, "floor_idx"].max()) + 1
    parts = []
    for pid, blk in df.groupby("block"):
        d = blk[blk["point_idx"] > 0]
        floors, pts = d["floor_idx"].astype(int).to_numpy(), d["point_idx"].astype(int).to_numpy()
        xy = np.full((n_sec, pts.max(), 2), np.nan)
        xy[floors, pts - 1] = d[["x_mm", "y_mm"]].to_numpy(float)
        parts.append(dict(name=PART_NAMES[pid], t=float(blk.loc[blk["point_idx"] == 0, "t_mm"].iloc[0]), xy=xy))
    return parts


def alive(p) -> np.ndarray:
    """섹션별 파트 존재 여부."""
    return ~np.isnan(p["xy"][:, 0, 0])


def write_parts(parts, path) -> None:
    """입력과 같은 형식: 파트마다 있는 섹션의 데이터 행 뒤에 구분 행(point_idx = 0, t_mm = 두께)."""
    rows = []
    for p in parts:
        for s in np.flatnonzero(alive(p)):
            rows += [(float(s), float(k), x, y, 0.0, 0.0) for k, (x, y) in enumerate(p["xy"][s], start=1)]
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


def half_gap(t, a, b) -> float:
    """두 판이 맞닿을 때 중심선 간격 (t_a + t_b)/2."""
    return 0.5 * (t[a] + t[b])


def clearances(parts, s):
    """(위 파트, point 번호, 간격) 목록: 간격 = 위 y − 아래 y − (t_위 + t_아래)/2 (접합 point 제외)."""
    out = []
    for up, lo in STACKED:
        if not (alive(parts[up])[s] and alive(parts[lo])[s]):
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
    """기본 단면 base에 두께 t_new(7)와 Inner 거리 d_new(15)를 적용. 반환: (단면, 겹침 보정 point 수)."""
    parts = [dict(p, xy=p["xy"].copy()) for p in base]
    t_old = [p["t"] for p in base]
    for p, t in zip(parts, t_new):
        p["t"] = float(t)
    n_sec = len(d_new)

    # 2. 플랜지 접합 (Outer, Inner): 새 접합 간격만큼 통째로 옮기고 접합 point를 정확히 맞춘다
    plate = parts[PLATE]["xy"]
    for pid in (OUTER, INNER):
        pairs = [(kp, kb) for kp, pb, kb in FLANGE_PAIRS if pb == pid]
        old = np.array([base[pid]["xy"][:, kb - 1, 1] - plate[:, kp - 1, 1] for kp, kb in pairs])  # (pair, sec)
        sign = np.sign(np.nanmean(old))                                 # 위(+) / 아래(−)
        new_gap = sign * half_gap(t_new, PLATE, pid)
        parts[pid]["xy"][:, :, 1] += (new_gap - old.mean(axis=0))[:, None]
        for kp, kb in pairs:
            parts[pid]["xy"][:, kb - 1] = plate[:, kp - 1] + np.array([0.0, new_gap])

    # 3. 밀착 덧판: 기준 파트(새 위치) + 오프셋 × 접촉 간격 비율
    for f, ref, k0 in FOLLOWERS:
        n = base[f]["xy"].shape[1]
        off = base[f]["xy"] - base[ref]["xy"][:, k0 - 1:k0 - 1 + n]
        ratio = half_gap(t_new, ref, f) / half_gap(t_old, ref, f)
        parts[f]["xy"] = parts[ref]["xy"][:, k0 - 1:k0 - 1 + n] + off * ratio   # 없는 섹션은 NaN 유지

    # 4. Inner 거리: 몸체의 Plate 기준 높이를 비율로 조정 (플랜지 고정)
    for s in range(n_sec):
        body = parts[INNER]["xy"][s, INNER_BODY]
        yp = surface_y(plate[s], body[:, 0])
        gap = body[:, 1] - yp
        parts[INNER]["xy"][s, INNER_BODY, 1] = yp + gap * (d_new[s] / np.nanmax(gap))

    # 5. 겹침 방지: 어기는 point만 위로 올린다
    repaired = 0
    for s in range(n_sec):
        for _ in range(3):                                              # 올린 파트가 다음 쌍에 영향 → 반복
            bad = [(up, k, c) for up, k, c in clearances(parts, s) if c < CLEAR_MARGIN - 1e-9]
            if not bad:
                break
            for up, k, c in bad:
                parts[up]["xy"][s, k, 1] += CLEAR_MARGIN - c
                repaired += 1
    return parts, repaired


def validate(base, parts, d_target) -> dict:
    """접합 오차 (플랜지 간격 − (t_a+t_b)/2), 밀착 오차 (덧판 오프셋 길이 − 기본 길이 × 간격 비율),
    최소 간격, d_s 오차."""
    plate, t_old, t_new = parts[PLATE]["xy"], [p["t"] for p in base], [p["t"] for p in parts]
    joint_err = max(np.nanmax(abs(abs(parts[pb]["xy"][:, kb - 1, 1] - plate[:, kp - 1, 1])
                                  - half_gap(t_new, PLATE, pb))) for kp, pb, kb in FLANGE_PAIRS)
    follow_err = 0.0
    for f, ref, k0 in FOLLOWERS:
        n = base[f]["xy"].shape[1]
        L0 = np.linalg.norm(base[f]["xy"] - base[ref]["xy"][:, k0 - 1:k0 - 1 + n], axis=-1)
        L1 = np.linalg.norm(parts[f]["xy"] - parts[ref]["xy"][:, k0 - 1:k0 - 1 + n], axis=-1)
        follow_err = max(follow_err, np.nanmax(abs(L1 - L0 * half_gap(t_new, ref, f) / half_gap(t_old, ref, f))))
    n_sec = len(d_target)
    min_clear = min(c for s in range(n_sec) for _, _, c in clearances(parts, s))
    d_err = max(abs(inner_distance(parts, s) - d_target[s]) for s in range(n_sec))
    return dict(joint_err_mm=joint_err, follow_err_mm=follow_err, min_clearance_mm=min_clear, d_err_mm=d_err)


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

    n_sec = base[OUTER]["xy"].shape[0]
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1.3, 1]))
    for ax, s in zip(axes, (0, n_sec // 2, n_sec - 1)):
        for smp in samples[:30]:
            for pid, p in enumerate(smp):
                if alive(p)[s]:
                    ax.plot(*p["xy"][s].T, "-", color=f"C{pid}", lw=0.4, alpha=0.35)
        for pid, p in enumerate(base):
            if alive(p)[s]:
                ax.plot(*p["xy"][s].T, "k-", lw=1.0, label="nominal" if pid == 0 else None)
        ax.set_title(f"sec {s}: nominal (black) + first 30 samples", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[3].scatter(design["xi_t_Outer"], design[f"xi_d_sec{n_sec // 2:02d}"], s=6)
    axes[3].set_xlabel("ξ t_Outer")
    axes[3].set_ylabel(f"ξ d_sec{n_sec // 2:02d}")
    axes[3].set_title(f"LHS projection (2 of {design.shape[1] // 2 - 1} variables)", fontsize=10)
    axes[3].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ─────────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="기본(명목) 단면 CSV")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="출력 폴더")
    ap.add_argument("--m", type=int, default=4, help="PDD 다항식 차수")
    ap.add_argument("--s", type=int, default=1, help="PDD 분해 차수 (S = 1: 1변수 성분)")
    ap.add_argument("--oversample", type=int, default=3, help="샘플 수 = PDD 계수 수 × oversample")
    ap.add_argument("--seed", type=int, default=2026, help="난수 시드 (같은 시드 = 같은 샘플)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = read_parts(a.csv)
    assert len(base) == N_PART, f"v23 형식(7파트)이 아님: {len(base)}파트"
    n_sec = base[OUTER]["xy"].shape[0]
    var_names = [f"t_{n}" for n in PART_NAMES] + [f"d_sec{s:02d}" for s in range(n_sec)]
    t_nom = np.array([p["t"] for p in base])
    d_nom = np.array([inner_distance(base, s) for s in range(n_sec)])
    nominal = np.r_[t_nom, d_nom]
    N = len(nominal)
    L, n = pdd_sample_count(N, a.m, a.s, a.oversample)
    print(f"[PDD] N={N} (두께 {N_PART} + d_s {n_sec}), m={a.m}, S={a.s} → 계수 {L}개 × {a.oversample} = 샘플 {n}개")

    rng = np.random.default_rng(a.seed)
    xi = 2.0 * lhs(n, N, rng) - 1.0                                   # ξ ∈ [−1, 1]
    X = nominal * (1.0 + REL_RANGE * xi)                              # 실제 값 = 기본값 × (1 ± 10 %)

    width = max(3, len(str(n)))
    rows, samples = [], []
    for i in range(n):
        parts, repaired = make_sample(base, X[i, :N_PART], X[i, N_PART:])
        chk = validate(base, parts, X[i, N_PART:])
        fname = f"bpillar_sample_{i + 1:0{width}d}.csv"
        write_parts(parts, out_dir / fname)
        rows.append(dict(sample=i + 1, file=fname, repaired_points=repaired, **chk))
        samples.append(parts)

    design = pd.DataFrame(xi, columns=[f"xi_{v}" for v in var_names])
    for j, v in enumerate(var_names):
        design[v] = X[:, j]
    design.insert(0, "file", [r["file"] for r in rows])
    design.insert(0, "sample", np.arange(1, n + 1))
    design.to_csv(out_dir / "bpillar_samples_design.csv", index=False, float_format="%.6f")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "bpillar_samples_summary.csv", index=False, float_format="%.6g")
    pd.DataFrame(dict(variable=var_names, nominal=nominal, lower=nominal * (1 - REL_RANGE),
                      upper=nominal * (1 + REL_RANGE))).to_csv(
        out_dir / "bpillar_samples_variables.csv", index=False, float_format="%.6f")
    plot_overview(base, samples, design, out_dir / "bpillar_samples_overview.png")

    print(f"[done] 샘플 {n}개 → {out_dir}")
    print(f"  접합 오차 최대 {summary['joint_err_mm'].max():.2e} mm, 밀착 오차 최대 {summary['follow_err_mm'].max():.2e} mm")
    print(f"  최소 간격 {summary['min_clearance_mm'].min():.3f} mm (≥ 0 이면 겹침 없음)")
    print(f"  d_s 오차 최대 {summary['d_err_mm'].max():.2e} mm")
    print(f"  겹침 보정: {(summary['repaired_points'] > 0).sum()}개 샘플, "
          f"샘플당 최대 {summary['repaired_points'].max()} point")


if __name__ == "__main__":
    main()
