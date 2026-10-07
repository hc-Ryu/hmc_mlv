"""UQ 샘플 단면별 섹션 소성 모멘트 Mp 계산 (v23) — 샘플마다 CSV 하나.

사용 (conda env fem):
    python section_mp_per_sample.py                        # ../../../sample/v23/bpillar_sample_*.csv 전체
    python section_mp_per_sample.py --sample-dir 폴더 --out-dir 폴더

Mp 계산 (CGNN 리포트·post_processing과 같은 방법)
    섹션마다 그 섹션에 있는 파트의 폴리라인 선분을 면적 A = L·t, 항복응력 fy인 점으로 보고,
    인장·압축 합력이 같아지는 소성 중립축 y_p를 찾은 뒤  Mp = Σ fy·A·|y − y_p|  (충돌 방향 y 굽힘).
    fy (MPa): Outer 1470, Plate 980, Inner 1470, Patch1 980, Patch2 440, Patch3 440, Patch4 980
    일부 섹션에만 있는 파트(Patch1–3: sec 11–14, Patch4: sec 0,1,5,6,8–14)는 있는 섹션에서만 더한다.

출력 (이 스크립트 폴더)
    bpillar_sample_001_mp.csv …   열: sec, Mp_Nmm, y_p_mm (소성 중립축), area_mm2 (단면적)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_SAMPLE_DIR = HERE.parents[2] / "sample" / "v23"
PART_FY = [1470.0, 980.0, 1470.0, 980.0, 440.0, 440.0, 980.0]   # Outer, Plate, Inner, Patch1–4


def read_parts(path) -> list[dict]:
    """단면 CSV → [{t, fy, xy (n_sec, n_pt, 2)}]. 파트 블록 끝 구분 행(point_idx = 0)에 두께.
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
        parts.append(dict(t=float(blk.loc[blk["point_idx"] == 0, "t_mm"].iloc[0]), fy=PART_FY[pid], xy=xy))
    return parts


def plastic_moment(parts, s) -> tuple[float, float, float]:
    """섹션 s의 (Mp [N·mm], 소성 중립축 y_p [mm], 단면적 [mm²])."""
    y, force, area = [], [], 0.0
    for p in parts:
        xy = p["xy"][s]
        if np.isnan(xy[0, 0]):                        # 이 섹션에 없는 파트
            continue
        L = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        y.append(0.5 * (xy[1:, 1] + xy[:-1, 1]))
        force.append(L * p["t"] * p["fy"])
        area += float((L * p["t"]).sum())
    y, force = np.concatenate(y), np.concatenate(force)
    o = np.argsort(y)
    y_p = float(y[o][np.searchsorted(np.cumsum(force[o]), 0.5 * force.sum())])
    return float(np.sum(force * np.abs(y - y_p))), y_p, area


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR), help="샘플 단면 CSV 폴더")
    ap.add_argument("--pattern", default="bpillar_sample_[0-9]*.csv", help="샘플 파일 패턴")
    ap.add_argument("--out-dir", default=str(HERE), help="출력 폴더")
    a = ap.parse_args()

    files = sorted(Path(a.sample_dir).glob(a.pattern))
    if not files:
        raise SystemExit(f"샘플 파일이 없습니다: {Path(a.sample_dir) / a.pattern}")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp_all = []
    for f in files:
        parts = read_parts(f)
        n_sec = parts[0]["xy"].shape[0]
        rows = [dict(sec=s, **dict(zip(("Mp_Nmm", "y_p_mm", "area_mm2"), plastic_moment(parts, s))))
                for s in range(n_sec)]
        pd.DataFrame(rows).to_csv(out_dir / f"{f.stem}_mp.csv", index=False, float_format="%.6f")
        mp_all.append([r["Mp_Nmm"] for r in rows])

    mp_all = np.array(mp_all) / 1e6
    print(f"[done] 샘플 {len(files)}개 → {out_dir} (각 {files[0].stem}_mp.csv 형식)")
    print("섹션별 Mp (kN·m) 샘플 분포:  sec    최소     평균     최대   변동계수(%)")
    for s in range(mp_all.shape[1]):
        v = mp_all[:, s]
        print(f"{'':30s}{s:3d}  {v.min():7.2f}  {v.mean():7.2f}  {v.max():7.2f}  {100 * v.std() / v.mean():6.2f}")


if __name__ == "__main__":
    main()
