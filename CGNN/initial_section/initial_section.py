#!/usr/bin/env python
# coding: utf-8
"""
uni_section_v18.py 17-Section 확장 시각화
─────────────────────────────────────────
`build_bpillar_section()`(단일 섹션, 5-part hat 단면)을 17층으로 등방(x,y) 스케일링하여
확장한다. section 17(최상단)이 원본 형상(s=1.0)이고, section 1(최하단)로 갈수록
s가 선형으로 1.6까지 커진다. 파트별 두께(t)/항복강도(fy)는 섹션에 무관하게 고정.

산출:
  1) 섹션별 2D 단면 plot (17개)
  2) 17개 섹션을 종합한 3D plot (matplotlib — 회전/확대 가능)
  3) 17개 섹션을 종합한 3D plot (plotly — 브라우저 인터랙티브, HTML 저장)
  4) area/Mp 단조성 검증 (assert)
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
# 원본 단일 섹션 정의 (uni_section_v18.py build_bpillar_section 그대로)
# ══════════════════════════════════════════════════════════════════

part_configs = [
    # (part_id, y_base(미사용, 참고용), t, fy, is_hat)
    (0, 30.0, 2.30, 1470.0, True),   # Outer Hat
    (1, 28.05, 1.60, 980.0, False),  # Inner Plate
    (2, 29.0, 1.60, 1470.0, True),   # Inner Hat
    (3, 24.0, 1.40, 980.0, False),   # Patch 1
    (4, 22.0, 1.60, 440.0, False),   # Patch 2
]

part_name = {0: 'Outer Hat', 1: 'Inner Plate', 2: 'Inner Hat', 3: 'Patch 1', 4: 'Patch 2'}
color_map = {0: '#FF5722', 1: '#FFEE00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}

NUM_NODES = 30
TOTAL_WIDTH = 160.0
NUM_SECTIONS = 17
S_MAX = 1.6   # section 1 (bottom) scale
S_MIN = 1.0   # section 17 (top) scale


def section_scale(k):
    """k: 1(bottom) .. 17(top, original). Returns isotropic scale factor s(k)."""
    return S_MAX - (S_MAX - S_MIN) * (k - 1) / (NUM_SECTIONS - 1)


def _hat_profile(x_ratio, flange_y, crown_y, ramp_lo, ramp_hi, plateau_lo, plateau_hi):
    """flange -> ramp -> crown -> ramp -> flange 트라페조이드 보간 (uni_section_v18.py 그대로)."""
    if x_ratio < plateau_lo:
        frac = (x_ratio - ramp_lo) / (plateau_lo - ramp_lo)
        frac = min(max(frac, 0.0), 1.0)
        return flange_y + frac * (crown_y - flange_y)
    elif x_ratio > plateau_hi:
        frac = (ramp_hi - x_ratio) / (ramp_hi - plateau_hi)
        frac = min(max(frac, 0.0), 1.0)
        return flange_y + frac * (crown_y - flange_y)
    else:
        return crown_y


def build_base_part_points(part_id):
    """원본(section 17, s=1.0) 기준 (x, y) 노드 좌표 리스트를 생성 (트림 적용됨)."""
    eps = 1e-3
    pts = []
    for i in range(NUM_NODES):
        x_ratio = i / (NUM_NODES - 1)
        x_coord = x_ratio * TOTAL_WIDTH

        if part_id == 2 and (x_ratio < 0.2 - eps or x_ratio > 0.8 + eps):
            continue
        if part_id == 3 and (x_ratio < 0.3 - eps or x_ratio > 0.7 + eps):
            continue
        if part_id == 4 and (x_ratio < 0.7 - eps or x_ratio > 0.8 + eps):
            continue

        if part_id == 0:
            if x_ratio <= 0.1667 + eps or x_ratio >= 0.8333 - eps:
                y = 30.0
            else:
                y = _hat_profile(x_ratio, 30.0, 65.0, 0.1667, 0.8333, 0.3334, 0.6666)
        elif part_id == 1:
            if x_ratio <= 0.0833 + eps or x_ratio >= 0.9167 - eps:
                y = 28.05
            elif (0.0833 + eps <= x_ratio < 0.3334 - eps) or (0.6666 + eps < x_ratio <= 0.9167 - eps):
                y = 8.05
            elif 0.3334 - eps <= x_ratio <= 0.6666 + eps:
                y = 15.0
            else:
                y = 8.05
        elif part_id == 2:
            if (x_ratio <= 0.3 - eps and x_ratio > 0.2 + eps) or (x_ratio >= 0.7 + eps and x_ratio < 0.8 - eps):
                y = 9.65
            else:
                y = _hat_profile(x_ratio, 9.65, 45.0, 0.2, 0.8, 0.4, 0.6)
        elif part_id == 3:
            if (x_ratio <= 0.33 - eps and x_ratio > 0.3 + eps) or (x_ratio >= 0.67 + eps and x_ratio < 0.7 - eps):
                y = 9.55
            else:
                y = 16.5
        elif part_id == 4:
            y = 7.45
        else:
            raise ValueError(part_id)

        pts.append((x_coord, y))
    return np.array(pts, dtype=float)


BASE_PARTS = {pid: build_base_part_points(pid) for pid, *_ in part_configs}


def build_section_coords(k):
    """섹션 k(1..17)의 파트별 좌표: 등방 스케일 s(k) 적용."""
    s = section_scale(k)
    return {pid: BASE_PARTS[pid] * s for pid in BASE_PARTS}


coords_by_section = {k: build_section_coords(k) for k in range(1, NUM_SECTIONS + 1)}


# ══════════════════════════════════════════════════════════════════
# area / Mp 계산 (uni_section_v18.py compute_edge_mp_pna 재현, numpy)
# ══════════════════════════════════════════════════════════════════

def compute_area_mp(section_coords, n_iter=200):
    edges = []  # (x_u, y_u, x_v, y_v, t, fy)
    t_by_part = {pid: t for pid, _, t, _, _ in part_configs}
    fy_by_part = {pid: fy for pid, _, _, fy, _ in part_configs}
    for pid, pts in section_coords.items():
        t, fy = t_by_part[pid], fy_by_part[pid]
        for i in range(len(pts) - 1):
            x_u, y_u = pts[i]
            x_v, y_v = pts[i + 1]
            edges.append((x_u, y_u, x_v, y_v, t, fy))

    xu = np.array([e[0] for e in edges]); yu = np.array([e[1] for e in edges])
    xv = np.array([e[2] for e in edges]); yv = np.array([e[3] for e in edges])
    t  = np.array([e[4] for e in edges]); fy = np.array([e[5] for e in edges])

    L = np.sqrt((xu - xv) ** 2 + (yu - yv) ** 2)
    area_total = np.sum(L * t)

    dx = np.abs(xu - xv)
    t_y = t * (dx / (L + 1e-12))
    y_max, y_min = np.maximum(yu, yv), np.minimum(yu, yv)
    y_top, y_bot = y_max + t_y / 2.0, y_min - t_y / 2.0
    H = np.clip(y_top - y_bot, 1e-12, None)
    Area_fy = L * t * fy

    y_lo = min(yu.min(), yv.min()) - 5.0
    y_hi = max(yu.max(), yv.max()) + 5.0
    for _ in range(n_iter):
        y_mid = 0.5 * (y_lo + y_hi)
        alpha = np.clip((y_top - y_mid) / H, 0.0, 1.0)
        net_force = np.sum(Area_fy * (2.0 * alpha - 1.0))
        if net_force > 0:
            y_lo = y_mid
        else:
            y_hi = y_mid
    y_pna = 0.5 * (y_lo + y_hi)

    alpha = np.clip((y_top - y_pna) / H, 0.0, 1.0)
    centroid_top = y_top - (alpha * H) / 2.0
    centroid_bot = y_bot + ((1.0 - alpha) * H) / 2.0
    m_top = alpha * (centroid_top - y_pna)
    m_bot = (1.0 - alpha) * (y_pna - centroid_bot)
    mp_total = np.sum(Area_fy * (m_top + m_bot))
    return area_total, mp_total


def validate_monotonicity():
    prev_area, prev_mp = None, None
    rows = []
    for k in range(NUM_SECTIONS, 0, -1):  # 17(top) -> 1(bottom)
        area, mp = compute_area_mp(coords_by_section[k])
        if prev_area is not None:
            assert area > prev_area, f"Area not increasing at section {k}"
            assert mp > prev_mp, f"Mp not increasing at section {k}"
        rows.append((k, section_scale(k), area, mp))
        prev_area, prev_mp = area, mp
    print("[validate] Area, Mp 모두 section 17->1 방향 단조증가 확인 완료.")
    print(f"{'section':>7}  {'scale':>7}  {'area':>10}  {'mp':>14}")
    for k, s, area, mp in rows:
        print(f"{k:7d}  {s:7.4f}  {area:10.2f}  {mp:14.1f}")
    return rows


# ══════════════════════════════════════════════════════════════════
# 1) 섹션별 2D 단면 plot
# ══════════════════════════════════════════════════════════════════

def plot_sections_2d():
    n_cols = 4
    n_rows = math.ceil(NUM_SECTIONS / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for idx, k in enumerate(range(NUM_SECTIONS, 0, -1)):  # 17(top) first -> 1(bottom) last
        ax = axes[idx]
        section_coords = coords_by_section[k]
        for pid, pts in section_coords.items():
            ax.plot(pts[:, 0], pts[:, 1], '-o', ms=3, color=color_map[pid], label=part_name[pid])
        tag = ' (TOP, orig.)' if k == 17 else (' (BOTTOM)' if k == 1 else '')
        ax.set_title(f'Section {k}{tag}  s={section_scale(k):.3f}', fontsize=9, fontweight='bold')
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Y (mm)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='upper right')

    for j in range(NUM_SECTIONS, len(axes)):
        axes[j].axis('off')

    fig.suptitle('uni_section_v18 — 17-Section B-Pillar Initial Cross Sections', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# 2) 17개 섹션 종합 3D plot (matplotlib — 회전/확대 가능)
# ══════════════════════════════════════════════════════════════════

def plot_sections_3d_matplotlib(z_coef=15.0):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    for k in range(1, NUM_SECTIONS + 1):
        z = (k - 1) * z_coef
        for pid, pts in coords_by_section[k].items():
            ax.plot(pts[:, 0], pts[:, 1], zs=z, zdir='z', color=color_map[pid], linewidth=1.3, alpha=0.85)
            ax.scatter(pts[:, 0], pts[:, 1], zs=z, zdir='z', c=color_map[pid], s=10,
                       edgecolors='k', linewidths=0.3)

    handles = [plt.Line2D([0], [0], color=color_map[p], lw=2, label=part_name[p]) for p in color_map]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5))

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Section (1=bottom .. 17=top)')
    ax.set_title('uni_section_v18 — 17-Section B-Pillar 3D Initial Shape', fontsize=13, fontweight='bold')
    ax.view_init(elev=22, azim=-60)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════
# 3) 17개 섹션 종합 3D plot (plotly — 브라우저 인터랙티브, HTML 저장)
# ══════════════════════════════════════════════════════════════════

def plot_sections_3d_plotly(z_coef=15.0, out_html='initial_section_3d.html'):
    import plotly.graph_objects as go

    fig = go.Figure()
    shown = set()
    for k in range(1, NUM_SECTIONS + 1):
        z = (k - 1) * z_coef
        for pid, pts in coords_by_section[k].items():
            show_legend = pid not in shown
            shown.add(pid)
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=np.full(len(pts), z),
                mode='lines+markers',
                line=dict(color=color_map[pid], width=4),
                marker=dict(size=3, color=color_map[pid], line=dict(color='black', width=0.5)),
                name=part_name[pid],
                legendgroup=part_name[pid],
                showlegend=show_legend,
                hovertemplate=f'Section {k}, {part_name[pid]}<br>x=%{{x:.1f}}, y=%{{y:.1f}}<extra></extra>',
            ))

    fig.update_layout(
        title='uni_section_v18 — 17-Section B-Pillar (Drag to rotate, scroll to zoom)',
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)',
                   zaxis_title='Section (1=bottom .. 17=top)', aspectmode='data'),
        width=1100, height=850,
    )
    fig.write_html(out_html)
    print(f"Saved interactive 3D plot to: {out_html}")
    return fig


if __name__ == "__main__":
    validate_monotonicity()

    fig_2d = plot_sections_2d()
    fig_3d_mpl = plot_sections_3d_matplotlib()

    try:
        fig_3d_plotly = plot_sections_3d_plotly()
        fig_3d_plotly.show()
    except ImportError:
        print("plotly가 설치되어 있지 않아 3번(plotly) plot은 건너뜁니다. `pip install plotly`로 설치 가능합니다.")

    plt.show()
