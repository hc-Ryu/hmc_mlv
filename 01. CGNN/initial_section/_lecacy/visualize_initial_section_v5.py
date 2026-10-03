#!/usr/bin/env python
# coding: utf-8
"""visualize_initial_section_v5.py — initial_section_v5.csv(15층, 7파트)를 3D HTML로 시각화

csv_to_3d_html.py와 같은 스타일이며, 파트 구분은 point_idx 리셋 추정이 아니라
floor_profiles_template.md 규칙(구분 행 point_idx=0 = 파트 종료)으로 한다.
z축 라벨은 실제 기하(floor 0 = 최하단 sill, 마지막 floor = 최상단 roof rail)를 따른다.

사용법: py visualize_initial_section_v5.py [--csv v5.csv] [--output v5.html] [--z-coef 15]
"""

import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

PART_NAMES = ['Outer Hat', 'Inner Plate', 'Inner Hat', 'Patch 1', 'Patch 2', 'Patch 3', 'Patch 4']
PART_COLORS = ['#FF5722', '#FFEE00', '#4CAF50', '#2196F3', '#9C27B0', '#CE93D8', '#E91E63']


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


def plot_sections_3d(blocks, out_html, title, z_coef=15.0):
    fig = go.Figure()
    for pid, (t, floors) in enumerate(blocks):
        name = PART_NAMES[pid] if pid < len(PART_NAMES) else f'Part {pid}'
        color = PART_COLORS[pid] if pid < len(PART_COLORS) else '#888888'
        for k, floor in enumerate(sorted(floors)):
            pts = floors[floor]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=np.full(len(pts), floor * z_coef),
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=3, color=color, line=dict(color='black', width=0.5)),
                name=f'{name} (t={t:g})', legendgroup=name, showlegend=(k == 0),
                hovertemplate=f'Floor {floor}, {name} (t={t:g} mm)'
                              '<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>',
            ))

    n_floors = max(max(f) for _, f in blocks) + 1
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)',
                   zaxis_title=f'Z (floor x {z_coef:g}; floor 0=bottom/sill ~ {n_floors - 1}=top)',
                   aspectmode='data'),
        width=1100, height=850,
    )
    fig.write_html(out_html)
    print(f'[viz] 인터랙티브 3D 저장: {out_html}')
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(_THIS_DIR, 'initial_section_v5.csv'))
    ap.add_argument('--output', default=os.path.join(_THIS_DIR, 'initial_section_v5.html'))
    ap.add_argument('--z-coef', type=float, default=15.0, help='floor 간 z 간격(시각화용 스케일)')
    a = ap.parse_args()

    blocks = load_blocks(a.csv)
    n_floors = max(max(f) for _, f in blocks) + 1
    title = (f'{os.path.basename(a.csv)} — {n_floors}-Floor, {len(blocks)}-Part B-Pillar '
             '(Drag to rotate, scroll to zoom)')
    plot_sections_3d(blocks, a.output, title, z_coef=a.z_coef)
