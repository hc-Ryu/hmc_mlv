"""GNN 설계 단면 후처리: 좌우 비대칭·뒤틀림을 살짝 다듬고 섹션별 소성 모멘트 Mp를 유지한다.

사용 (conda env fem):
    python post_processing.py                                  # 기본 입력 (v1_21_0_x1.3 최종 단면)
    python post_processing.py --csv 설계.csv --no-mp-correct   # Mp 보정 없이 기하 후처리만

처리 순서 (결정적 알고리즘 — 같은 입력이면 항상 같은 결과)
    1. 좌우 대칭화   : Outer/Plate/Inner/Patch1의 point k와 대칭 짝 point(n+1−k)를 섹션 중심선
                       (x_c = Outer pt1과 pt n의 중점) 기준으로 평균낸다. Patch2는 한쪽(Plate pt 22–24)에만
                       있으므로 대칭화하지 않고 접합된 Plate point를 따라 움직인다.
    2. 높이 방향 스무딩: 같은 point 번호의 좌표를 17개 섹션에 걸쳐 Savitzky-Golay(2차, 5점)로 다듬어
                       섹션 사이 뒤틀림을 줄인다 (선형·2차 테이퍼는 그대로 보존).
    3. 단면 안 스무딩: 플랜지 접합 point와 파트 양 끝을 뺀 point에 약한 라플라시안(이웃 평균 쪽으로 λ)을 준다.
    4. 접합 복원     : 플랜지 접합 쌍의 원래 간격(오프셋)을 되돌린다 (접합 공차 유지).
    5. Mp 보정       : 섹션마다 소성 중립축 y_p 기준 y 좌표를 c배(c ≈ 1) 스케일해 Mp를 원래 값에 맞춘다
                       (할선법, 허용 0.01 %). 대칭은 유지된다.

출력 (이 스크립트 폴더)
    <stem>_post.csv          후처리 단면 (입력과 같은 형식 — FEM 입력으로 바로 사용 가능)
    <stem>_post_mp.csv       섹션별 Mp·단면적·비대칭·이동량·파트 간 최소 간격 (전/후)
    <stem>_post_compare.png  섹션별 단면 비교 (점선 = 후처리 전, 실선 = 후처리 후)
    <stem>_post_mp.png       섹션별 Mp 전/후와 변화율
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "01. CGNN" / "reports" / "v1_21_0_x1.3" / "final_section_v1_21_0_x1.3.csv"

PART_NAMES = ["Outer", "Plate", "Inner", "Patch1", "Patch2"]
PART_FY = [1470.0, 980.0, 1470.0, 980.0, 440.0]          # MPa, CGNN FY_BY_PART
OUTER, PLATE, INNER, PATCH1, PATCH2 = range(5)
SYMMETRIC_PARTS = (OUTER, PLATE, INNER, PATCH1)
# 플랜지 접합 (part_a, point_a, part_b, point_b) — CGNN FLANGE_PAIR_SPEC. a가 기준(아래/허브) 쪽
FLANGE_PAIRS = [
    (PLATE, 1, OUTER, 1), (PLATE, 2, OUTER, 2), (PLATE, 3, OUTER, 3),
    (PLATE, 28, OUTER, 28), (PLATE, 29, OUTER, 29), (PLATE, 30, OUTER, 30),
    (PLATE, 7, INNER, 1), (PLATE, 8, INNER, 2), (PLATE, 9, INNER, 3),
    (PLATE, 22, INNER, 16), (PLATE, 23, INNER, 17), (PLATE, 24, INNER, 18),
    (PLATE, 10, PATCH1, 1), (PLATE, 21, PATCH1, 12),
    (PLATE, 22, PATCH2, 1), (PLATE, 23, PATCH2, 2), (PLATE, 24, PATCH2, 3),
]
# 위아래로 쌓인 파트 쌍 (위, 아래) — 겹침(간격) 검사용
STACKED = [(OUTER, INNER), (OUTER, PLATE), (INNER, PATCH1), (INNER, PLATE), (PATCH1, PLATE)]


# ─────────────────────────────── 입출력 ───────────────────────────────
def read_parts(path) -> list[dict]:
    """CSV → 파트 목록 [{name, t, xy (n_sec, n_pt, 2)}]. 파트 블록 끝 구분 행(point_idx = 0)에 두께."""
    df = pd.read_csv(path)
    df["block"] = (df["point_idx"] == 0).cumsum().shift(fill_value=0)
    parts = []
    for pid, blk in df.groupby("block"):
        data = blk[blk["point_idx"] > 0].sort_values(["floor_idx", "point_idx"])
        n_sec = data["floor_idx"].nunique()
        parts.append(dict(name=PART_NAMES[pid], t=float(blk.loc[blk["point_idx"] == 0, "t_mm"].iloc[0]),
                          xy=data[["x_mm", "y_mm"]].to_numpy().reshape(n_sec, -1, 2).astype(float)))
    return parts


def write_parts(parts, path) -> None:
    """입력과 같은 형식으로 저장: 파트마다 데이터 행 뒤에 구분 행(point_idx = 0, t_mm = 두께)."""
    rows = []
    for p in parts:
        for s, sec in enumerate(p["xy"]):
            for k, (x, y) in enumerate(sec, start=1):
                rows.append((float(s), float(k), x, y, 0.0, 0.0))
        rows.append((0.0, 0.0, 0.0, 0.0, 0.0, p["t"]))
    pd.DataFrame(rows, columns=["floor_idx", "point_idx", "x_mm", "y_mm", "r_mm", "t_mm"]).to_csv(
        path, index=False)


def copy_parts(parts) -> list[dict]:
    return [dict(p, xy=p["xy"].copy()) for p in parts]


# ─────────────────────────────── 단면 성질 ───────────────────────────────
def plastic_moment(parts, s) -> tuple[float, float, float]:
    """섹션 s의 소성 모멘트 Mp (N·mm, y 방향 굽힘), 소성 중립축 y_p, 단면적.
    선분마다 면적 L·t, 항복응력 fy인 점으로 보고 인장·압축 합력이 같아지는 y_p를 찾는다."""
    y, force, area = [], [], 0.0
    for pid, p in enumerate(parts):
        if s >= len(p["xy"]):
            continue
        xy = p["xy"][s]
        L = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        y.append(0.5 * (xy[1:, 1] + xy[:-1, 1]))
        force.append(L * p["t"] * PART_FY[pid])
        area += float((L * p["t"]).sum())
    y, force = np.concatenate(y), np.concatenate(force)
    o = np.argsort(y)
    y_p = float(y[o][np.searchsorted(np.cumsum(force[o]), 0.5 * force.sum())])
    return float(np.sum(force * np.abs(y - y_p))), y_p, area


def asymmetry(xy, xc) -> float:
    """대칭 짝 point를 중심선에 대해 뒤집었을 때 최대 어긋남 (mm)."""
    mirrored = xy[::-1].copy()
    mirrored[:, 0] = 2 * xc - mirrored[:, 0]
    return float(np.abs(xy - mirrored).max())


def min_clearance(parts, s, joint_pts) -> float:
    """쌓인 파트 쌍의 최소 간격 (mm) = 위 판 중심선 − 같은 x의 아래 판 중심선 − (t_위 + t_아래)/2.
    접합 point는 제외. 음수면 판 두께 범위가 겹친다."""
    worst = np.inf
    for up, lo in STACKED:
        if s >= len(parts[up]["xy"]) or s >= len(parts[lo]["xy"]):
            continue
        a, b = parts[up]["xy"][s], parts[lo]["xy"][s]
        half = 0.5 * (parts[up]["t"] + parts[lo]["t"])
        for k, (x, y) in enumerate(a, start=1):
            if (up, k) in joint_pts:
                continue
            x1, x2, y1, y2 = b[:-1, 0], b[1:, 0], b[:-1, 1], b[1:, 1]
            inside = (x >= np.minimum(x1, x2)) & (x <= np.maximum(x1, x2)) & (np.abs(x2 - x1) > 1e-9)
            if inside.any():
                yl = (y1 + (x - x1) / np.where(inside, x2 - x1, 1.0) * (y2 - y1))[inside].max()
                worst = min(worst, y - yl - half)
    return float(worst)


# ─────────────────────────────── 후처리 단계 ───────────────────────────────
def section_center(parts) -> np.ndarray:
    """섹션별 중심선 x_c = Outer pt1과 마지막 point의 중점."""
    o = parts[OUTER]["xy"]
    return 0.5 * (o[:, 0, 0] + o[:, -1, 0])


def symmetrize(parts, xc, weight=1.0) -> None:
    """대칭 파트의 point k와 짝 point(n+1−k)를 중심선 기준으로 평균 (weight = 1이면 완전 대칭)."""
    for pid in SYMMETRIC_PARTS:
        xy = parts[pid]["xy"]
        mirrored = xy[:, ::-1].copy()
        mirrored[..., 0] = 2 * xc[: len(xy), None] - mirrored[..., 0]
        xy += weight * (0.5 * (xy + mirrored) - xy)


def savgol(a: np.ndarray, window=5, order=2) -> np.ndarray:
    """첫 축(섹션) 방향 Savitzky-Golay: 점마다 주변 window개로 order차 다항식을 맞춰 그 점 값을 쓴다.
    끝에서는 창을 안쪽으로 밀어 같은 개수를 쓴다."""
    n = len(a)
    if n < window:
        return a.copy()
    out = np.empty_like(a)
    flat = a.reshape(n, -1)
    for i in range(n):
        lo = min(max(i - window // 2, 0), n - window)
        idx = np.arange(lo, lo + window)
        V = np.vander(idx - i, order + 1)               # 다항식 기저 (i 기준 → 상수항이 i의 값)
        coef, *_ = np.linalg.lstsq(V, flat[idx], rcond=None)
        out.reshape(n, -1)[i] = coef[-1]
    return out


def smooth_along_height(parts, window, order) -> None:
    for p in parts:
        p["xy"][:] = savgol(p["xy"], window, order)


def smooth_in_section(parts, fixed_pts, lam, iters) -> None:
    """고정 point(접합, 파트 양 끝)를 뺀 point를 이웃 평균 쪽으로 lam만큼 당긴다 (직선 구간은 변하지 않음)."""
    for pid, p in enumerate(parts):
        n = p["xy"].shape[1]
        free = np.array([k for k in range(2, n) if (pid, k) not in fixed_pts]) - 1   # 0-base
        if not len(free):
            continue
        for _ in range(iters):
            xy = p["xy"]
            xy[:, free] += lam * (0.5 * (xy[:, free - 1] + xy[:, free + 1]) - xy[:, free])


def restore_joints(parts, offsets) -> None:
    """접합 쌍의 두 번째 point를 기준 point + 원래 오프셋으로 되돌린다."""
    for (pa, ka, pb, kb), off in offsets.items():
        n = min(len(parts[pa]["xy"]), len(parts[pb]["xy"]))
        parts[pb]["xy"][:n, kb - 1] = parts[pa]["xy"][:n, ka - 1] + off[:n]


def correct_mp(parts, target_mp, offsets, tol=1e-4, max_iter=20) -> np.ndarray:
    """섹션마다 y를 y_p 기준 c배 스케일해 Mp를 target에 맞춘다 (할선법). 반환: 섹션별 c."""
    n_sec = len(parts[OUTER]["xy"])
    scales = np.ones(n_sec)
    for s in range(n_sec):
        base = copy_parts(parts)
        _, y_p, _ = plastic_moment(base, s)

        def mp_with(c):
            for p, b in zip(parts, base):
                if s < len(p["xy"]):
                    p["xy"][s, :, 1] = y_p + c * (b["xy"][s, :, 1] - y_p)
            restore_joints(parts, offsets)
            return plastic_moment(parts, s)[0]

        c0, c1 = 1.0, 1.01
        f0, f1 = mp_with(c0) - target_mp[s], mp_with(c1) - target_mp[s]
        for _ in range(max_iter):
            if abs(f1) <= tol * target_mp[s] or f1 == f0:
                break
            c0, c1, f0 = c1, c1 - f1 * (c1 - c0) / (f1 - f0), f1
            f1 = mp_with(c1) - target_mp[s]
        scales[s] = c1
    return scales


# ─────────────────────────────── 보고 ───────────────────────────────
def report(before, after, xc, joint_pts) -> pd.DataFrame:
    rows = []
    for s in range(len(before[OUTER]["xy"])):
        mp0, _, a0 = plastic_moment(before, s)
        mp1, _, a1 = plastic_moment(after, s)
        shift = max(float(np.linalg.norm(p1["xy"][s] - p0["xy"][s], axis=1).max())
                    for p0, p1 in zip(before, after) if s < len(p0["xy"]))
        rows.append(dict(sec=s, Mp_before_Nmm=mp0, Mp_after_Nmm=mp1, Mp_change_pct=100 * (mp1 / mp0 - 1),
                         area_before_mm2=a0, area_after_mm2=a1, area_change_pct=100 * (a1 / a0 - 1),
                         asym_before_mm=max(asymmetry(before[p]["xy"][s], xc[s]) for p in SYMMETRIC_PARTS
                                            if s < len(before[p]["xy"])),
                         asym_after_mm=max(asymmetry(after[p]["xy"][s], xc[s]) for p in SYMMETRIC_PARTS
                                           if s < len(after[p]["xy"])),
                         max_shift_mm=shift,
                         clearance_before_mm=min_clearance(before, s, joint_pts),
                         clearance_after_mm=min_clearance(after, s, joint_pts)))
    return pd.DataFrame(rows)


def plot_compare(before, after, path, title) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sec = len(before[OUTER]["xy"])
    fig, axes = plt.subplots(3, 6, figsize=(26, 11))
    for s, ax in enumerate(axes.flat):
        if s >= n_sec:
            ax.axis("off")
            continue
        for pid, (p0, p1) in enumerate(zip(before, after)):
            if s < len(p0["xy"]):
                c = f"C{pid}"
                ax.plot(*p0["xy"][s].T, ":", color=c, lw=1.0)
                ax.plot(*p1["xy"][s].T, "-o", color=c, lw=1.1, ms=1.8, label=p1["name"])
        ax.set_title(f"sec {s}", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"{title} — dotted: before, solid: after post-processing", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_mp(rep: pd.DataFrame, path, title) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    x = rep["sec"].to_numpy()
    axes[0].bar(x - 0.2, rep["Mp_before_Nmm"] / 1e6, 0.4, label="before")
    axes[0].bar(x + 0.2, rep["Mp_after_Nmm"] / 1e6, 0.4, label="after")
    axes[0].set_ylabel("Mp (kN·m)")
    axes[1].bar(x, rep["Mp_change_pct"], color="C3")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("Mp change (%)")
    axes[2].plot(x, rep["asym_before_mm"], "o-", label="asymmetry before")
    axes[2].plot(x, rep["asym_after_mm"], "o-", label="asymmetry after")
    axes[2].plot(x, rep["max_shift_mm"], "s--", label="max point shift")
    axes[2].set_ylabel("mm")
    for ax in axes:
        ax.set_xlabel("section")
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ─────────────────────────────── main ───────────────────────────────
def post_process(parts, sym_weight=1.0, z_window=5, z_order=2, lam=0.25, lam_iters=1, mp_correct=True):
    """후처리된 파트 목록과 섹션별 Mp 보정 배율을 돌려준다."""
    out = copy_parts(parts)
    xc = section_center(parts)
    offsets = {pair: parts[pair[2]]["xy"][:, pair[3] - 1] - parts[pair[0]]["xy"][:len(parts[pair[2]]["xy"]),
                                                                              pair[1] - 1]
               for pair in FLANGE_PAIRS}
    target_mp = np.array([plastic_moment(parts, s)[0] for s in range(len(xc))])
    joint_pts = {(pa, ka) for pa, ka, _, _ in FLANGE_PAIRS} | {(pb, kb) for _, _, pb, kb in FLANGE_PAIRS}
    ends = {(pid, k) for pid, p in enumerate(parts) for k in (1, p["xy"].shape[1])}

    patch2_anchor = parts[PATCH2]["xy"] - parts[PLATE]["xy"][:, 21:24]        # Patch2 ↔ Plate pt 22–24
    symmetrize(out, xc, sym_weight)
    smooth_along_height(out, z_window, z_order)
    smooth_in_section(out, joint_pts | ends, lam, lam_iters)
    out[PATCH2]["xy"] = out[PLATE]["xy"][:, 21:24] + patch2_anchor           # Patch2는 Plate를 따라감
    restore_joints(out, offsets)
    scales = correct_mp(out, target_mp, offsets) if mp_correct else np.ones(len(xc))
    return out, xc, scales, joint_pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="GNN 설계 단면 CSV")
    ap.add_argument("--out-dir", default=str(HERE), help="출력 폴더 (기본: 이 스크립트 폴더)")
    ap.add_argument("--sym-weight", type=float, default=1.0, help="대칭화 강도 0–1 (1 = 완전 대칭)")
    ap.add_argument("--z-window", type=int, default=5, help="높이 방향 스무딩 창 (섹션 수, 홀수)")
    ap.add_argument("--lam", type=float, default=0.25, help="단면 안 스무딩 강도 (0 = 끔)")
    ap.add_argument("--no-mp-correct", action="store_true", help="Mp 보정 끄기")
    a = ap.parse_args()

    src = Path(a.csv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = read_parts(src)
    post, xc, scales, joint_pts = post_process(parts, a.sym_weight, a.z_window, 2, a.lam, 1,
                                               not a.no_mp_correct)

    stem = src.stem
    write_parts(post, out_dir / f"{stem}_post.csv")
    rep = report(parts, post, xc, joint_pts)
    rep["mp_scale_c"] = scales
    rep.to_csv(out_dir / f"{stem}_post_mp.csv", index=False, float_format="%.6g")
    plot_compare(parts, post, out_dir / f"{stem}_post_compare.png", stem)
    plot_mp(rep, out_dir / f"{stem}_post_mp.png", f"{stem}: Mp before/after post-processing")

    pd.set_option("display.width", 200)
    view = rep[["sec", "Mp_before_Nmm", "Mp_after_Nmm", "Mp_change_pct", "area_change_pct",
                "asym_before_mm", "asym_after_mm", "max_shift_mm", "clearance_before_mm", "clearance_after_mm"]]
    print(view.to_string(index=False, float_format=lambda v: f"{v:10.3f}" if abs(v) < 1e5 else f"{v:.4e}"))
    print(f"\nMp 변화: 최대 |{rep['Mp_change_pct'].abs().max():.4f}| %   "
          f"비대칭: {rep['asym_before_mm'].max():.2f} → {rep['asym_after_mm'].max():.2f} mm   "
          f"최대 이동: {rep['max_shift_mm'].max():.2f} mm")
    if (rep["clearance_after_mm"] < np.minimum(rep["clearance_before_mm"], 0) - 1e-6).any():
        print("[경고] 후처리 후 파트 간 겹침이 생긴 섹션이 있습니다 (clearance_after_mm < 0).")
    print(f"→ {out_dir / (stem + '_post.csv')}")


if __name__ == "__main__":
    main()
