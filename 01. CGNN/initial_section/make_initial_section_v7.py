#!/usr/bin/env python
# coding: utf-8
"""make_initial_section_v7.py — initial_section_v6.csv -> initial_section_v7.csv 변환

변경 사항 (v6 -> v7):
  1. 단면 깊이(y 방향 폭) 확대: floor별 배율 s(f)를 sill에서 SY0배, 최상단에서 1배로 선형 감소.
       s(f) = 1 + (SY0 - 1) * (1 - f / f_max)   -> 기본 SY0=1.5: 1.500, 1.464, 1.429, ..., 1.036, 1.000
     x는 그대로 (v6 flare 유지).
  2. 접합(CONTACT) 유지: CSV 좌표는 판재 중심선이라 단순 y 확대 시 겹침 접합 간격 (t_i+t_j)/2도 s배로 벌어진다.
     Inner Plate를 기준(root)으로 파트별 y map  y' = p + s*(y - p) + c_part  (p = floor별 Plate 최소 y)를 쓰고,
     평탄 접합부 간격 g가 유지되도록 c_part = (1 - s) * g_signed 를 준다 (Plate 기준 위 +, 아래 -):
       Outer Hat  : Plate 플랜지 위 +1.95   Inner Hat : Plate 위 +1.6   Patch 2/3 : Plate 아래 -1.6
     경사 구간에도 붙는 Patch 1(Plate 위), Patch 4(Outer crown 아래)는 v6와 같은 법선 offset으로 다시 만든다.
     -> Plate 바닥면(p)과 그 아래 Patch 2/3 위치는 v6와 동일하고, 단면은 +y 쪽으로 깊어진다.

사용법: py make_initial_section_v7.py [--src v6.csv] [--dst v7.csv] [--html v7.html] [--sy0 1.5]
"""

import argparse
import os

from make_initial_section_v5 import T_PATCH4, read_blocks, write_csv, make_patch4
from make_initial_section_v6 import DZ, reoffset_patch1
from visualize_initial_section_v5 import load_blocks, plot_sections_3d

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_PARTS = 7
PID_OUTER, PID_PLATE, PID_INNER_HAT, PID_PATCH1, PID_PATCH2, PID_PATCH3, PID_PATCH4 = range(7)

# Plate 기준 평탄 접합부의 부호 있는 중심선 간격 (위 +, 아래 -). 0 = Plate 자신.
# Patch 1/4는 이후 법선 offset으로 재생성하므로 임시로 0.
CONTACT_GAP = {PID_OUTER: +1.95, PID_PLATE: 0.0, PID_INNER_HAT: +1.6, PID_PATCH1: 0.0,
               PID_PATCH2: -1.6, PID_PATCH3: -1.6, PID_PATCH4: 0.0}


def depth_scale(floor, sy0, f_max):
    """y 방향 폭 배율: floor 0에서 sy0, 최상단 f_max에서 1로 선형 감소."""
    return 1.0 + (sy0 - 1.0) * (1.0 - floor / f_max)


def remap_y(blocks, sy0):
    """모든 파트에 floor별 y' = p + s*(y - p) + (1 - s)*gap 적용 (x 불변)."""
    plate = blocks[PID_PLATE][1]
    pivot = plate.groupby('floor_idx')['y_mm'].min()
    f_max = plate['floor_idx'].max()
    out = []
    for pid, (t, b) in enumerate(blocks):
        b = b.copy()
        for floor in b['floor_idx'].unique():
            m = b['floor_idx'] == floor
            s = depth_scale(floor, sy0, f_max)
            b.loc[m, 'y_mm'] = (pivot[floor] + s * (b.loc[m, 'y_mm'] - pivot[floor])
                                + (1.0 - s) * CONTACT_GAP[pid])
        out.append((t, b))
    return out


def section_depth(blocks, floor):
    """단면 y 방향 전체 폭 (모든 파트 중심선 기준)."""
    ys = [b.loc[b['floor_idx'] == floor, 'y_mm'] for _, b in blocks]
    return max(y.max() for y in ys) - min(y.min() for y in ys)


def main(src, dst, html, sy0):
    blocks = read_blocks(src)
    assert len(blocks) == NUM_PARTS, f'v6 파트 블록 수가 {NUM_PARTS}이 아닙니다: {len(blocks)}'
    src_blocks = blocks

    blocks = remap_y(blocks, sy0)

    (t_plate, plate), (t_p1, patch1) = blocks[PID_PLATE], blocks[PID_PATCH1]
    blocks[PID_PATCH1] = (t_p1, reoffset_patch1(patch1, plate, (t_plate + t_p1) / 2.0))
    blocks[PID_PATCH4] = (T_PATCH4, make_patch4(blocks[PID_OUTER][1]))

    write_csv(blocks, dst)
    f_max = blocks[PID_PLATE][1]['floor_idx'].max()
    for f in sorted(blocks[PID_OUTER][1]['floor_idx'].unique()):
        d0, d1 = section_depth(src_blocks, f), section_depth(blocks, f)
        print(f'floor {int(f):2d}: s={depth_scale(f, sy0, f_max):.3f}  '
              f'depth {d0:6.2f} -> {d1:6.2f} mm  (x{d1 / d0:.3f})')
    print(f'[v7] 저장: {dst}')

    n_floors = int(f_max) + 1
    title = (f'{os.path.basename(dst)} — {n_floors}-Floor, {len(blocks)}-Part B-Pillar, '
             f'H={(n_floors - 1) * DZ:.1f} mm (dz={DZ:.2f} mm)')
    plot_sections_3d(load_blocks(dst), html, title, z_coef=DZ)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(_THIS_DIR, 'initial_section_v6.csv'))
    ap.add_argument('--dst', default=os.path.join(_THIS_DIR, 'initial_section_v7.csv'))
    ap.add_argument('--html', default=os.path.join(_THIS_DIR, 'initial_section_v7.html'))
    ap.add_argument('--sy0', type=float, default=1.5, help='floor 0(sill) y 방향 폭 배율 (최상단은 1.0)')
    a = ap.parse_args()
    main(a.src, a.dst, a.html, a.sy0)
