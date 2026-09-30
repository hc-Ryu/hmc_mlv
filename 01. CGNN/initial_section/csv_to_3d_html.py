#!/usr/bin/env python
# coding: utf-8
"""floor_idx,point_idx,x_mm,y_mm,r_mm,t_mm 형식 단면 CSV를 initial_section.py의
plot_sections_3d_plotly() 스타일 인터랙티브 3D HTML로 변환한다.

각 floor 안에서 point_idx가 이전 값보다 작아지는(=1로 리셋되는) 지점을 새 파트의 시작으로
간주해 그룹을 나누고, 그룹 등장 순서를 part_id(0=Outer Hat .. 4=Patch 2)로 매핑한다.
floor 0에는 그룹 종료를 표시하는 point_idx=0 구분 행(두께 정보)이 섞여 있어 건너뛴다.

사용법: py csv_to_3d_html.py <input.csv> [output.html]
"""

import sys
import numpy as np
import pandas as pd

PART_NAMES = {0: 'Outer Hat', 1: 'Inner Plate', 2: 'Inner Hat', 3: 'Patch 1', 4: 'Patch 2'}
PART_COLORS = {0: '#FF5722', 1: '#FFEE00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}


def load_sections(csv_path):
    """{floor: {part_id: (N,2) ndarray}} 반환. point_idx=0 구분 행은 제외."""
    df = pd.read_csv(csv_path, encoding='utf-8')
    df = df[df['point_idx'] != 0.0]

    sections = {}
    for floor, fsub in df.groupby('floor_idx'):
        fsub = fsub.reset_index(drop=True)
        groups = []
        cur = []
        prev_idx = None
        for _, row in fsub.iterrows():
            idx = row['point_idx']
            if prev_idx is not None and idx <= prev_idx:
                groups.append(cur)
                cur = []
            cur.append((row['x_mm'], row['y_mm']))
            prev_idx = idx
        if cur:
            groups.append(cur)

        sections[int(floor)] = {
            pid: np.array(pts, dtype=float) for pid, pts in enumerate(groups)
        }
    return sections


def plot_sections_3d_plotly(sections, z_coef=15.0, out_html='sections_3d.html',
                             title='B-Pillar Cross Sections (Drag to rotate, scroll to zoom)'):
    import plotly.graph_objects as go

    fig = go.Figure()
    shown = set()
    for floor in sorted(sections.keys()):
        z = floor * z_coef
        for pid, pts in sections[floor].items():
            name = PART_NAMES.get(pid, f'Part {pid}')
            color = PART_COLORS.get(pid, '#888888')
            show_legend = pid not in shown
            shown.add(pid)
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=np.full(len(pts), z),
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=3, color=color, line=dict(color='black', width=0.5)),
                name=name,
                legendgroup=name,
                showlegend=show_legend,
                hovertemplate=f'Floor {floor}, {name}<br>x=%{{x:.1f}}, y=%{{y:.1f}}<extra></extra>',
            ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)',
                   zaxis_title='Floor (0=top/original ~ N=bottom)', aspectmode='data'),
        width=1100, height=850,
    )
    fig.write_html(out_html)
    print(f"[viz] 인터랙티브 3D 저장: {out_html}")
    return fig


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: py csv_to_3d_html.py <input.csv> [output.html]")
        sys.exit(1)
    in_csv = sys.argv[1]
    out_html = sys.argv[2] if len(sys.argv) > 2 else in_csv.rsplit('.', 1)[0] + '.html'

    sections = load_sections(in_csv)
    plot_sections_3d_plotly(sections, out_html=out_html)
