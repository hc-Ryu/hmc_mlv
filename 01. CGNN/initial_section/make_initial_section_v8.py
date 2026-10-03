#!/usr/bin/env python
# coding: utf-8
"""make_initial_section_v8.py — initial_section_v7.csv -> initial_section_v8.csv 변환

변경 사항 (v7 -> v8):
  1. 최하단(floor 0, sill)만 x, y 방향 폭을 모두 SCALE(기본 1.3)배 확대. floor 1~14는 형상 불변.
       x: floor 0 Outer Hat x 범위 중심 cx 기준  x' = cx + S*(x - cx)  (좌우 대칭 확대)
       y: v7과 같은 규칙 — Inner Plate 바닥 p 고정,  y' = p + S*(y - p) + (1 - S)*gap_part
          (gap_part = Plate 기준 평탄 접합 간격: Outer +1.95, Inner Hat +1.6, Patch2/3 -1.6)
     x 확대는 수평 접합부의 법선 간격을 바꾸지 않고, y 보정으로 판재 두께 방향 접합 간격((t_i+t_j)/2)을
     유지한다. 경사 구간에도 붙는 Patch1(Plate 위), Patch4(Outer crown 아래)는 법선 offset으로 재생성.
  2. 전체(모든 floor)를 x로 (S - 1)*(cx - x_left)만큼 평행이동해 floor 0 좌측 끝을 다시 x=0에 둔다
     (v6와 같은 규약, 평행이동은 Mp와 무관).

사용법: py make_initial_section_v8.py [--src v7.csv] [--dst v8.csv] [--html v8.html] [--scale 1.3]
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '_lecacy'))   # v5/v6 헬퍼는 _lecacy/로 이동됨

from make_initial_section_v5 import T_PATCH4, read_blocks, write_csv, make_patch4
from make_initial_section_v6 import DZ, reoffset_patch1
from visualize_initial_section_v5 import load_blocks, plot_sections_3d

NUM_PARTS = 7
PID_OUTER, PID_PLATE, PID_INNER_HAT, PID_PATCH1, PID_PATCH2, PID_PATCH3, PID_PATCH4 = range(7)

# Plate 기준 평탄 접합부의 부호 있는 중심선 간격 (위 +, 아래 -) — make_initial_section_v7.py와 동일.
CONTACT_GAP = {PID_OUTER: +1.95, PID_PLATE: 0.0, PID_INNER_HAT: +1.6, PID_PATCH1: 0.0,
               PID_PATCH2: -1.6, PID_PATCH3: -1.6, PID_PATCH4: 0.0}

TARGET_FLOOR = 0.0


def scale_floor0(blocks, s):
    """floor 0만 x(중심 기준)·y(Plate 바닥 기준, 접합 간격 보정) s배, 이후 전체 x 평행이동."""
    outer = blocks[PID_OUTER][1]
    plate = blocks[PID_PLATE][1]
    g0 = outer.loc[outer['floor_idx'] == TARGET_FLOOR, 'x_mm']
    cx = 0.5 * (g0.min() + g0.max())
    p = plate.loc[plate['floor_idx'] == TARGET_FLOOR, 'y_mm'].min()
    shift = (s - 1.0) * (cx - g0.min())   # 확대 후 floor 0 좌측 끝을 원래 x(=0)로 되돌림
    out = []
    for pid, (t, b) in enumerate(blocks):
        b = b.copy()
        m = b['floor_idx'] == TARGET_FLOOR
        b.loc[m, 'x_mm'] = cx + s * (b.loc[m, 'x_mm'] - cx)
        b.loc[m, 'y_mm'] = p + s * (b.loc[m, 'y_mm'] - p) + (1.0 - s) * CONTACT_GAP[pid]
        b['x_mm'] += shift
        out.append((t, b))
    return out


def extent(blocks, floor):
    """floor의 (x 폭, y 폭) — 모든 파트 중심선 기준."""
    sel = [b[b['floor_idx'] == floor] for _, b in blocks]
    xs = [g['x_mm'] for g in sel]
    ys = [g['y_mm'] for g in sel]
    return (max(x.max() for x in xs) - min(x.min() for x in xs),
            max(y.max() for y in ys) - min(y.min() for y in ys))


def main(src, dst, html, s):
    blocks = read_blocks(src)
    assert len(blocks) == NUM_PARTS, f'v7 파트 블록 수가 {NUM_PARTS}이 아닙니다: {len(blocks)}'
    src_blocks = blocks

    blocks = scale_floor0(blocks, s)

    (t_plate, plate), (t_p1, patch1) = blocks[PID_PLATE], blocks[PID_PATCH1]
    blocks[PID_PATCH1] = (t_p1, reoffset_patch1(patch1, plate, (t_plate + t_p1) / 2.0))
    blocks[PID_PATCH4] = (T_PATCH4, make_patch4(blocks[PID_OUTER][1]))

    write_csv(blocks, dst)
    for f in sorted(blocks[PID_OUTER][1]['floor_idx'].unique()):
        (w0, d0), (w1, d1) = extent(src_blocks, f), extent(blocks, f)
        print(f'floor {int(f):2d}: x {w0:6.1f} -> {w1:6.1f} (x{w1 / w0:.3f})  '
              f'y {d0:6.2f} -> {d1:6.2f} (x{d1 / d0:.3f})')
    print(f'[v8] 저장: {dst}')

    n_floors = int(blocks[PID_OUTER][1]['floor_idx'].max()) + 1
    title = (f'{os.path.basename(dst)} — {n_floors}-Floor, {len(blocks)}-Part B-Pillar, '
             f'H={(n_floors - 1) * DZ:.1f} mm (dz={DZ:.2f} mm)')
    plot_sections_3d(load_blocks(dst), html, title, z_coef=DZ)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(_THIS_DIR, 'initial_section_v7.csv'))
    ap.add_argument('--dst', default=os.path.join(_THIS_DIR, 'initial_section_v8.csv'))
    ap.add_argument('--html', default=os.path.join(_THIS_DIR, 'initial_section_v8.html'))
    ap.add_argument('--scale', type=float, default=1.3, help='floor 0(sill) x, y 폭 배율')
    a = ap.parse_args()
    main(a.src, a.dst, a.html, a.scale)
