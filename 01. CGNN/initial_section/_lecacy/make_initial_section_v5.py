#!/usr/bin/env python
# coding: utf-8
"""make_initial_section_v5.py — initial_section_v4.csv -> initial_section_v5.csv 변환

변경 사항 (v4 -> v5):
  1. 17층 -> 15층: 상단 2개 섹션(floor 15, 16) 삭제.
     floor 0 = 최하단(sill, 폭 256mm), floor 16 = 최상단(roof rail, 폭 160mm)이므로
     floor 0~14만 남긴다. floor 인덱스 재번호 없음(0~14 그대로).
  2. 5파트 -> 7파트: Patch 3(part_id 5), Patch 4(part_id 6) 추가.
     - Patch 3: 기존 Patch 2(Inner Hat 우측 foot 아래, Plate 외측 덧판)를 단면 중심선
       x = W/2 기준으로 좌우 대칭 복제 -> Inner Hat 좌측 foot 접합부 보강.
       y는 Patch 2와 동일(Plate 바닥 - 1.6mm), t = 1.6mm(Patch 2와 동일).
     - Patch 4: Outer Hat crown 안쪽 덧판(crown doubler). 측면 충돌(-y)을 직접 받는
       Outer crown 평탄부(point 11~20)에서 양쪽 벽으로 2노드씩 더 덮는다(point 9~22, 14점).
       x는 Outer 노드 x를 그대로 쓰고, y는 Outer 프로파일을 안쪽 법선 방향으로
       (t_outer + t_patch4)/2 만큼 offset한 곡선(miter join)에서 읽는다 — 경사 벽 구간에서도
       판 간 수직 거리가 v4의 파트 간 간격 규칙과 같게 유지된다. t = 1.4mm.
  3. Patch 1 수정: v4는 Plate를 y로만 +1.5 올려 양 끝 경사 구간에서 Plate와 0.567mm 겹쳤다
     (check_initial_section_interference.py로 확인). Plate 법선 방향 offset으로 재배치 -> fix_patch1().
  파트 간 mid-surface 간격은 v4 규칙((t1+t2)/2)을 그대로 따른다.

CSV 규격: floor_profiles_template.md (데이터 행 먼저, 파트 종료 직후 구분 행으로 두께 선언).

사용법: py make_initial_section_v5.py [--src v4.csv] [--dst v5.csv]
"""

import argparse
import os

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_FLOORS_V5 = 15
T_OUTER = 2.3
T_PATCH3 = 1.6
T_PATCH4 = 1.4
PATCH4_POINTS = range(9, 23)  # Outer Hat point_idx 9~22 = crown 평탄부(11~20) + 양쪽 벽 2노드


def read_blocks(csv_path):
    """[(t, DataFrame[floor_idx, point_idx, x_mm, y_mm]), ...] — CSV 블록 순서 = part_id."""
    df = pd.read_csv(csv_path)
    blocks, start = [], 0
    for i in df.index[df['point_idx'] == 0.0]:
        blocks.append((float(df.at[i, 't_mm']), df.iloc[start:i].reset_index(drop=True)))
        start = i + 1
    return blocks


def floor_width(outer, floor):
    """해당 floor의 단면 전체 폭 W (Outer Hat 마지막 점 x)."""
    return outer.loc[outer['floor_idx'] == floor, 'x_mm'].max()


def make_patch3(patch2, outer):
    """Patch 2를 x -> W - x 로 좌우 대칭 복제 (point 순서는 x 오름차순 유지)."""
    rows = []
    for floor, g in patch2.groupby('floor_idx'):
        w = floor_width(outer, floor)
        xs = np.sort(w - g['x_mm'].to_numpy())
        ys = g['y_mm'].to_numpy()[::-1]
        for k, (x, y) in enumerate(zip(xs, ys), start=1):
            rows.append((floor, float(k), x, y))
    return pd.DataFrame(rows, columns=['floor_idx', 'point_idx', 'x_mm', 'y_mm'])


def offset_inward(xy, gap):
    """Outer 폴리라인을 안쪽(진행 방향 기준 오른쪽 = crown 아래쪽) 법선으로 gap만큼 offset (miter join)."""
    d = np.diff(xy, axis=0)
    n = np.stack([d[:, 1], -d[:, 0]], axis=1) / np.linalg.norm(d, axis=1, keepdims=True)
    out = np.empty_like(xy)
    out[0] = xy[0] + gap * n[0]
    out[-1] = xy[-1] + gap * n[-1]
    for i in range(1, len(xy) - 1):
        n1, n2 = n[i - 1], n[i]
        out[i] = xy[i] + gap * (n1 + n2) / (1.0 + n1 @ n2)
    return out


def make_patch4(outer):
    """Outer crown + 양쪽 벽 2노드 안쪽에 붙는 doubler. x = Outer 노드 x, y = offset 곡선에서 보간."""
    gap = (T_OUTER + T_PATCH4) / 2.0
    rows = []
    for floor, g in outer.groupby('floor_idx'):
        g = g.sort_values('point_idx')
        off = offset_inward(g[['x_mm', 'y_mm']].to_numpy(), gap)
        sel = g[g['point_idx'].isin(PATCH4_POINTS)]
        ys = np.interp(sel['x_mm'].to_numpy(), off[:, 0], off[:, 1])
        for k, (x, y) in enumerate(zip(sel['x_mm'], ys), start=1):
            rows.append((floor, float(k), x, y))
    return pd.DataFrame(rows, columns=['floor_idx', 'point_idx', 'x_mm', 'y_mm'])


def fix_patch1(patch1, plate, gap):
    """v4 Patch 1은 Plate를 y로만 +gap 평행이동해서 경사 구간(끝단 램프)에서 법선 거리가
    gap·cosθ로 줄어 판재가 겹친다(-0.567mm). Plate 윗면 쪽(진행 방향 기준 왼쪽) 법선 offset으로 재배치:
      - 내부 점: 대응 Plate 노드의 miter offset 꼭짓점 (볼록 모서리에서 x가 약간 이동)
      - 양 끝점: x는 유지, y는 첫/마지막 offset 세그먼트 직선 위에서 읽음
    -> 모든 Patch 1 세그먼트가 대응 Plate 세그먼트와 정확히 gap 거리로 평행."""
    rows = []
    for floor, g in patch1.groupby('floor_idx'):
        g = g.sort_values('point_idx')
        pl = plate[plate['floor_idx'] == floor].sort_values('point_idx')[['x_mm', 'y_mm']].to_numpy()
        px = g['x_mm'].to_numpy()
        idx = np.array([np.argmin(np.abs(pl[:, 0] - x)) for x in px])
        assert np.allclose(pl[idx, 0], px) and np.all(np.diff(idx) == 1), \
            f'floor {floor}: Patch 1 x가 Plate 노드 x와 일치하지 않습니다'
        off = offset_inward(pl[idx], -gap)
        for k in (0, -1):
            j = 1 if k == 0 else -2
            (x0, y0), (x1, y1) = off[k], off[j]
            off[k] = (px[k], y0 + (y1 - y0) * (px[k] - x0) / (x1 - x0))
        for k, (x, y) in enumerate(off, start=1):
            rows.append((floor, float(k), x, y))
    return pd.DataFrame(rows, columns=['floor_idx', 'point_idx', 'x_mm', 'y_mm'])


def write_csv(blocks, path):
    """floor_profiles_template.md 규격으로 기록: 파트 데이터 -> 구분 행(두께)."""
    out = []
    for t, blk in blocks:
        for r in blk.itertuples(index=False):
            out.append((float(r.floor_idx), float(r.point_idx), r.x_mm, r.y_mm, 0.0, 0.0))
        out.append((0.0, 0.0, 0.0, 0.0, 0.0, t))
    pd.DataFrame(out, columns=['floor_idx', 'point_idx', 'x_mm', 'y_mm', 'r_mm', 't_mm']) \
        .to_csv(path, index=False, float_format=None)


def main(src, dst):
    blocks = read_blocks(src)
    assert len(blocks) == 5, f'v4 파트 블록 수가 5가 아닙니다: {len(blocks)}'

    # 1) 상단 2개 섹션 삭제
    blocks = [(t, b[b['floor_idx'] < NUM_FLOORS_V5].reset_index(drop=True)) for t, b in blocks]

    # 2) Patch 1 경사 구간 Plate 간섭 수정
    (t_plate, plate), (t_p1, patch1) = blocks[1], blocks[3]
    blocks[3] = (t_p1, fix_patch1(patch1, plate, (t_plate + t_p1) / 2.0))

    # 3) Patch 3, Patch 4 추가
    outer = blocks[0][1]
    patch2 = blocks[4][1]
    blocks.append((T_PATCH3, make_patch3(patch2, outer)))
    blocks.append((T_PATCH4, make_patch4(outer)))

    write_csv(blocks, dst)
    for pid, (t, b) in enumerate(blocks):
        n = b.groupby('floor_idx').size()
        print(f'part {pid}: t={t}, floors={b.floor_idx.nunique()}, pts/floor={sorted(n.unique())}')
    print(f'[v5] 저장: {dst}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(_THIS_DIR, 'initial_section_v4.csv'))
    ap.add_argument('--dst', default=os.path.join(_THIS_DIR, 'initial_section_v5.csv'))
    a = ap.parse_args()
    main(a.src, a.dst)
