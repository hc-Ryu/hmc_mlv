#!/usr/bin/env python
# coding: utf-8
"""check_initial_section_interference.py — 초기 단면 CSV의 파트 간 간섭(겹침)·공차 검사

CSV 좌표는 판재의 중심선(mid-surface)이므로, 두 파트 i, j의 실제 판재 간 간격은
    clearance = d_min(중심선 i, 중심선 j) - (t_i + t_j) / 2
로 계산한다. d_min은 floor별로 두 폴리라인의 모든 세그먼트 쌍 간 최소 거리이다.
(판재를 중심선 ± t/2 로 두께를 준 띠로 보고, 끝단은 반원 캡으로 근사하므로 끝단에서는 보수적이다.)

판정 (tol = 공차, gap_min = 최소 이격 거리):
    OVERLAP : clearance < -tol                → 판재가 파고듦 (간섭)
    CONTACT : |clearance| <= tol             → 의도된 겹침 접합 (공칭 간격 일치)
    NEAR    : tol < clearance < gap_min       → 붙지도 떨어지지도 않은 애매한 틈 (조립/용접 불가 수준)
    CLEAR   : clearance >= gap_min            → 충분히 이격

사용법: py check_initial_section_interference.py [--csv v5.csv] [--tol 0.05] [--gap-min 1.0]
                                                 [--report out.csv]
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

PART_NAMES = ['Outer Hat', 'Inner Plate', 'Inner Hat', 'Patch 1', 'Patch 2', 'Patch 3', 'Patch 4']


def load_blocks(csv_path):
    """[(t, {floor: (N,2) ndarray}), ...] — 블록 순서 = part_id."""
    df = pd.read_csv(csv_path)
    blocks, start = [], 0
    for i in df.index[df['point_idx'] == 0.0]:
        blk = df.iloc[start:i]
        floors = {int(f): g[['x_mm', 'y_mm']].to_numpy() for f, g in blk.groupby('floor_idx')}
        blocks.append((float(df.at[i, 't_mm']), floors))
        start = i + 1
    return blocks


def _point_seg_dist(p, a, b):
    """점 p(...,2)와 세그먼트 a-b(...,2) 간 거리 및 최근접점 (브로드캐스팅)."""
    ab = b - a
    denom = np.maximum((ab * ab).sum(-1), 1e-12)
    s = np.clip(((p - a) * ab).sum(-1) / denom, 0.0, 1.0)
    q = a + s[..., None] * ab
    return np.linalg.norm(p - q, axis=-1), q


def _cross(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def polyline_min_dist(P, Q):
    """두 폴리라인 P(n,2), Q(m,2) 간 최소 거리와 그 위치(두 최근접점의 중점)."""
    a, b = P[:-1][:, None, :], P[1:][:, None, :]
    c, d = Q[:-1][None, :, :], Q[1:][None, :, :]

    # 세그먼트 교차 여부 (진교차: 부호가 엄격히 반대)
    d1, d2 = _cross(b - a, c - a), _cross(b - a, d - a)
    d3, d4 = _cross(d - c, a - c), _cross(d - c, b - c)
    inter = (d1 * d2 < 0) & (d3 * d4 < 0)

    cands = [(_point_seg_dist(a, c, d), a), (_point_seg_dist(b, c, d), b),
             (_point_seg_dist(c, a, b), c), (_point_seg_dist(d, a, b), d)]
    dists = np.stack([np.broadcast_to(r[0][0], inter.shape) for r in cands])
    dist = np.where(inter, 0.0, dists.min(0))

    k = np.unravel_index(np.argmin(dist), dist.shape)
    which = int(np.argmin(dists[:, k[0], k[1]]))
    (_, q), p = cands[which]
    p = np.broadcast_to(p, inter.shape + (2,))[k]
    q = np.broadcast_to(q, inter.shape + (2,))[k]
    loc = (p + q) / 2.0
    return float(dist[k]), loc, bool(inter.any())


def classify(c, tol, gap_min):
    if c < -tol:
        return 'OVERLAP'
    if c <= tol:
        return 'CONTACT'
    if c < gap_min:
        return 'NEAR'
    return 'CLEAR'


def check(csv_path, tol, gap_min):
    blocks = load_blocks(csv_path)
    floors = sorted(set().union(*[f.keys() for _, f in blocks]))
    rows = []
    for (i, (ti, fi)), (j, (tj, fj)) in itertools.combinations(enumerate(blocks), 2):
        nominal = (ti + tj) / 2.0
        for fl in floors:
            if fl not in fi or fl not in fj:
                continue
            d, loc, crossed = polyline_min_dist(fi[fl], fj[fl])
            c = d - nominal
            rows.append(dict(part_i=i, part_j=j, floor=fl, d_center=d, nominal=nominal,
                             clearance=c, x=loc[0], y=loc[1], crossed=crossed,
                             status=classify(c, tol, gap_min)))
    return pd.DataFrame(rows)


def summarize(df):
    """파트 쌍별 최악(최소 clearance) floor 기준 요약."""
    order = {'OVERLAP': 0, 'NEAR': 1, 'CONTACT': 2, 'CLEAR': 3}
    out = []
    for (i, j), g in df.groupby(['part_i', 'part_j']):
        w = g.loc[g['clearance'].idxmin()]
        counts = g['status'].value_counts().to_dict()
        out.append(dict(
            pair=f'{PART_NAMES[i]} ↔ {PART_NAMES[j]}',
            nominal=w['nominal'], min_clear=w['clearance'], max_clear=g['clearance'].max(),
            worst_floor=int(w['floor']), at=f"({w['x']:.1f}, {w['y']:.1f})",
            status=min(counts, key=order.get),
            floors=' '.join(f'{k}:{v}' for k, v in sorted(counts.items(), key=lambda kv: order[kv[0]])),
        ))
    s = pd.DataFrame(out)
    return s.sort_values('status', key=lambda c: c.map(order), kind='stable')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(_THIS_DIR, 'initial_section_v5.csv'))
    ap.add_argument('--tol', type=float, default=0.05, help='접촉 판정 공차 (mm)')
    ap.add_argument('--gap-min', type=float, default=1.0, help='비접촉 파트 간 최소 이격 (mm)')
    ap.add_argument('--report', default=None, help='floor별 상세 결과 CSV 저장 경로')
    a = ap.parse_args()

    detail = check(a.csv, a.tol, a.gap_min)
    summary = summarize(detail)

    pd.set_option('display.width', 200)
    print(f'[check] {os.path.basename(a.csv)}  tol=±{a.tol} mm, gap_min={a.gap_min} mm')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    bad = detail[detail['status'].isin(['OVERLAP', 'NEAR'])]
    if len(bad):
        print('\n[check] OVERLAP/NEAR 상세 (floor별)')
        print(bad.assign(pair=bad.apply(lambda r: f'{PART_NAMES[r.part_i]} ↔ {PART_NAMES[r.part_j]}', axis=1))
              [['pair', 'floor', 'd_center', 'nominal', 'clearance', 'x', 'y', 'crossed', 'status']]
              .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    else:
        print('\n[check] OVERLAP/NEAR 없음')

    if a.report:
        detail.to_csv(a.report, index=False)
        print(f'[check] 상세 저장: {a.report}')
