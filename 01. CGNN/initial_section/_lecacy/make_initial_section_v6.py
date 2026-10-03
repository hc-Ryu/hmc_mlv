#!/usr/bin/env python
# coding: utf-8
"""make_initial_section_v6.py — initial_section_v5.csv -> initial_section_v6.csv 변환

변경 사항 (v5 -> v6):
  0. 높이 기준: FEM/bpillar_fem_demo.py와 같은 실제 B-pillar 높이 — 17섹션(sec 0~16) 1051.7 mm,
     층간 DZ = 1051.7/16 = 65.73 mm. v6는 15층(상단 2층 삭제)이므로 z = f*DZ, 0 ~ 920.2 mm.
     CSV에는 z 열이 없으므로(floor_profiles_template.md 규격) 높이는 flare 형상 정의에만 쓰인다.
  1. 하단 flare: sill에서 높이 L_MM 구간의 단면 폭을 s(z)배로 넓힌다 (실제 B-pillar의 sill 쪽 나팔형 확장).
       s(z) = 1 + A * (1 - z/L_MM)^2   (z < L_MM),   s(z) = 1   (z >= L_MM)
     기본값 A=1.0, L_MM=6*DZ=394.4 mm -> floor별 s = 2.00, 1.69, 1.44, 1.25, 1.11, 1.03, 1.00, ...
     (1 - z/L)^2 형태라 z=L_MM에서 기울기 0으로 윗 섹션과 C1 연속, sill 쪽으로 갈수록 급격히 벌어진다.
  2. 좌측 정렬 해제(하단부만): 늘어난 폭 dW(f) = (s-1)*W(f) 중 BIAS 비율은 +x 쪽(v5에서 이미 기울어 있던
     우측)으로, 나머지는 -x 쪽(v5에서 x=0으로 정렬돼 있던 좌측)으로 나눠 준다 -> 하단에서 양쪽 끝이 모두
     비스듬하게 벌어진다. floor L 이상은 v5 배치 그대로(좌측 x 정렬 유지)이고, 전체를 X0만큼 평행이동해
     floor 0 좌측 끝이 x=0이 되게 한다 (평행이동은 Mp와 무관).
  변환은 floor별 x 방향 affine map  x' = X0 + xl + s(f)*(x - xl) - (1-BIAS)*dW(f)  (xl = v5 좌측 끝),
  y는 그대로 — 7개 파트 모두에 같은 map을 쓰므로 point_idx 대응(층간 3D 연결)과 파트 간 접합 관계가 유지된다.
  x만 늘리면 경사 벽이 눕기만 하므로 평행 판재 사이 법선 거리는 줄지 않는다(간섭 불가). 다만 법선 offset으로
  만든 Patch 1(Plate 위), Patch 4(Outer crown 아래)는 간격이 (t1+t2)/2보다 벌어지므로 v5와 같은 방법으로
  다시 offset해 정확히 맞춘다.

사용법: py make_initial_section_v6.py [--src v5.csv] [--dst v6.csv] [--html v6.html]
                                     [--amp 1.0] [--span-mm 394.4] [--bias 0.75]
"""

import argparse
import os

import numpy as np
import pandas as pd

from make_initial_section_v5 import (T_PATCH4, read_blocks, write_csv, offset_inward,
                                     make_patch4)
from visualize_initial_section_v5 import load_blocks, plot_sections_3d

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_PARTS = 7
PID_OUTER, PID_PLATE, PID_PATCH1, PID_PATCH4 = 0, 1, 3, 6

HEIGHT_FULL = 1051.7   # FEM/bpillar_fem_demo.py HEIGHT: sec 0(sill) ~ sec 16(roof rail), 17섹션
DZ = HEIGHT_FULL / 16  # 층간 65.73 mm (v6 15층 -> 최상단 z = 14*DZ = 920.2 mm)


def flare_scale(floor, amp, span_mm):
    """하단 flare 폭 배율 s(z) = 1 + amp*(1 - z/span_mm)^2 (z < span_mm), 이후 1. z = floor*DZ."""
    r = max(0.0, 1.0 - floor * DZ / span_mm)
    return 1.0 + amp * r * r


def remap_x(blocks, outer, amp, span, bias):
    """모든 파트에 floor별 x' = X0 + xl + s*(x - xl) - (1-bias)*(s-1)*W 적용 (y 불변)."""
    xl = outer.groupby('floor_idx')['x_mm'].min()
    wid = outer.groupby('floor_idx')['x_mm'].max() - xl
    x0 = (1.0 - bias) * (flare_scale(0, amp, span) - 1.0) * wid[0.0] - xl[0.0]
    out = []
    for t, b in blocks:
        b = b.copy()
        for floor in b['floor_idx'].unique():
            m = b['floor_idx'] == floor
            s = flare_scale(floor, amp, span)
            b.loc[m, 'x_mm'] = (x0 + xl[floor] + s * (b.loc[m, 'x_mm'] - xl[floor])
                                - (1.0 - bias) * (s - 1.0) * wid[floor])
        out.append((t, b))
    return out


def reoffset_patch1(patch1, plate, gap):
    """Patch 1을 새 Plate에서 다시 법선 offset (make_initial_section_v5.fix_patch1과 같은 규칙).
    v5 Patch 1 내부점은 miter offset이라 Plate 노드 x와 다르므로, 양 끝점 x(= Plate 노드 x)로
    Plate 노드 구간을 찾아 그 구간 전체를 offset한다."""
    rows = []
    for floor, g in patch1.groupby('floor_idx'):
        g = g.sort_values('point_idx')
        pl = plate[plate['floor_idx'] == floor].sort_values('point_idx')[['x_mm', 'y_mm']].to_numpy()
        px = g['x_mm'].to_numpy()
        i0, i1 = (int(np.argmin(np.abs(pl[:, 0] - x))) for x in (px[0], px[-1]))
        assert np.isclose(pl[i0, 0], px[0]) and np.isclose(pl[i1, 0], px[-1]) and i1 - i0 + 1 == len(px), \
            f'floor {floor}: Patch 1 끝점이 Plate 노드와 대응하지 않습니다'
        off = offset_inward(pl[i0:i1 + 1], -gap)
        for k in (0, -1):
            j = 1 if k == 0 else -2
            (x0, y0), (x1, y1) = off[k], off[j]
            off[k] = (px[k], y0 + (y1 - y0) * (px[k] - x0) / (x1 - x0))
        for k, (x, y) in enumerate(off, start=1):
            rows.append((floor, float(k), x, y))
    return pd.DataFrame(rows, columns=['floor_idx', 'point_idx', 'x_mm', 'y_mm'])


def main(src, dst, html, amp, span, bias):
    blocks = read_blocks(src)
    assert len(blocks) == NUM_PARTS, f'v5 파트 블록 수가 {NUM_PARTS}이 아닙니다: {len(blocks)}'

    blocks = remap_x(blocks, blocks[PID_OUTER][1], amp, span, bias)

    (t_plate, plate), (t_p1, patch1) = blocks[PID_PLATE], blocks[PID_PATCH1]
    blocks[PID_PATCH1] = (t_p1, reoffset_patch1(patch1, plate, (t_plate + t_p1) / 2.0))
    blocks[PID_PATCH4] = (T_PATCH4, make_patch4(blocks[PID_OUTER][1]))

    write_csv(blocks, dst)
    outer = blocks[PID_OUTER][1]
    for f in sorted(outer['floor_idx'].unique()):
        g = outer[outer['floor_idx'] == f]['x_mm']
        print(f'floor {int(f):2d}: z={f * DZ:6.1f} mm  s={flare_scale(f, amp, span):.3f}  '
              f'x=[{g.min():7.2f}, {g.max():7.2f}]  W={g.max() - g.min():6.1f}')
    print(f'[v6] 저장: {dst}')

    # 3D HTML: z = floor*DZ (실제 높이, aspectmode='data'라 flare 경사가 실제 비율로 보인다)
    n_floors = int(outer['floor_idx'].max()) + 1
    title = (f'{os.path.basename(dst)} — {n_floors}-Floor, {len(blocks)}-Part B-Pillar, '
             f'H={(n_floors - 1) * DZ:.1f} mm (dz={DZ:.2f} mm)')
    plot_sections_3d(load_blocks(dst), html, title, z_coef=DZ)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(_THIS_DIR, 'initial_section_v5.csv'))
    ap.add_argument('--dst', default=os.path.join(_THIS_DIR, 'initial_section_v6.csv'))
    ap.add_argument('--html', default=os.path.join(_THIS_DIR, 'initial_section_v6.html'))
    ap.add_argument('--amp', type=float, default=1.0, help='floor 0(sill) 폭 증가량 (s = 1 + amp)')
    ap.add_argument('--span-mm', type=float, default=6 * DZ,
                    help='flare가 끝나는 sill 기준 높이 mm (C1 연속 지점, 기본 6층 = 394.4 mm)')
    ap.add_argument('--bias', type=float, default=0.75,
                    help='늘어난 폭 중 +x 쪽 비율 (0.5 = 좌우 대칭, 1.0 = 좌측 x 정렬 유지)')
    a = ap.parse_args()
    main(a.src, a.dst, a.html, a.amp, a.span_mm, a.bias)
