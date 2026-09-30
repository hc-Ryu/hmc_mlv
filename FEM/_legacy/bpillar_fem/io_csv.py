"""final_section_*.csv 파서 (idea_v1.md §1.1).

CSV는 5개 파트 블록이 차례로 나오고, 각 블록 끝에 구분 행(point_idx=0, t_mm=파트 두께)이 붙는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import N_SECTIONS

PART_NAMES = ["Outer", "Plate", "Inner", "Patch1", "Patch2"]
PART_FY = [1470.0, 980.0, 1470.0, 980.0, 440.0]   # 리포트 §4.1
OUTER, PLATE, INNER, PATCH1, PATCH2 = range(5)   # 파트 id = CSV 블록 순서

# Outer 몸체(플랜지 제외) point 번호 — pt 1–3, 28–30은 Plate와 접합된 플랜지
OUTER_BODY_PTS = tuple(range(4, 28))

# (part_a, point_idx_a, part_b, point_idx_b) — 01. CGNN/AI_design_v1_21_0_x1.3.py FLANGE_PAIR_SPEC
FLANGE_PAIR_SPEC = [
    (OUTER, 1, PLATE, 1), (OUTER, 2, PLATE, 2), (OUTER, 3, PLATE, 3),
    (OUTER, 28, PLATE, 28), (OUTER, 29, PLATE, 29), (OUTER, 30, PLATE, 30),
    (PLATE, 7, INNER, 1), (PLATE, 8, INNER, 2), (PLATE, 9, INNER, 3),
    (PLATE, 22, INNER, 16), (PLATE, 23, INNER, 17), (PLATE, 24, INNER, 18),
    (PLATE, 10, PATCH1, 1), (PLATE, 21, PATCH1, 12),
    (PLATE, 22, PATCH2, 1), (PLATE, 23, PATCH2, 2), (PLATE, 24, PATCH2, 3),
]


@dataclass
class Part:
    pid: int
    name: str
    t: float
    fy: float
    secs: np.ndarray        # 존재하는 섹션 번호 (0부터 연속)
    xy: np.ndarray          # (n_sec, n_pt, 2), point_idx 1..n_pt 순서

    @property
    def n_pt(self) -> int:
        return self.xy.shape[1]

    @property
    def last_sec(self) -> int:
        return int(self.secs[-1])


def read_parts(path: str) -> list[Part]:
    df = pd.read_csv(path)
    is_sep = df["point_idx"] == 0
    df["block"] = is_sep.cumsum().shift(fill_value=0).astype(int)

    parts = []
    for pid, blk in df.groupby("block"):
        sep = blk[blk["point_idx"] == 0]
        data = blk[blk["point_idx"] > 0]
        if len(sep) != 1:
            raise ValueError(f"block {pid}: 구분 행이 {len(sep)}개")
        secs = np.sort(data["floor_idx"].astype(int).unique())
        if not np.array_equal(secs, np.arange(secs[0], secs[0] + len(secs))) or secs[0] != 0:
            raise ValueError(f"block {pid}: 섹션이 0부터 연속이 아님 {secs}")
        counts = data.groupby("floor_idx").size().unique()
        if len(counts) != 1:
            # C3(호길이 재매개화)가 필요한 경우 — v1_21_0_x1.3에는 해당 없음
            raise NotImplementedError(f"block {pid}: 섹션마다 point 수가 다름 {counts}")
        data = data.sort_values(["floor_idx", "point_idx"])
        n_pt = int(counts[0])
        xy = data[["x_mm", "y_mm"]].to_numpy().reshape(len(secs), n_pt, 2)
        parts.append(Part(pid=int(pid), name=PART_NAMES[pid], t=float(sep["t_mm"].iloc[0]),
                          fy=PART_FY[pid], secs=secs, xy=xy))
    if len(parts) != len(PART_NAMES):
        raise ValueError(f"파트 수 {len(parts)} != {len(PART_NAMES)}")
    n_sec = max(p.last_sec for p in parts) + 1
    if n_sec != N_SECTIONS:
        raise ValueError(f"섹션 수 {n_sec} != {N_SECTIONS}")
    return parts
