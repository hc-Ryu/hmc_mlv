"""bpillar_fem_demo.py 격자 모델 시각화 (메쉬 + 경계조건 + 하중) → 인터랙티브 HTML.

사용 (conda env fem, plotly 필요):
    python FEM_visualization.py                                 # 기본: 후처리 단면, 해석과 같은 4분할 격자
    python FEM_visualization.py --csv 설계.csv --n-sub 1 --out mesh_quick.html

보이는 것 (범례 클릭으로 켜고 끔)
    - 파트별 요소: 원주 방향(섹션 안 단면 선분, 굵은 선)과 종방향(섹션 사이 선분, 가는 선)
    - 고정 경계조건: sec 0(sill), sec 16(roof rail) 전 절점 6자유도 고정
    - 플랜지 접합: 여러 파트가 하나로 병합된 절점
    - 하중: sec 5–11 Outer 윗면 절점의 −y 방향 화살표 (크기 = 그 절점이 받는 하중 비율)
    - 변위제어 절점: sec 8 Outer 윗면 (해석에서 이 절점을 −y로 눌러 나간다)
    - 섹션 번호 표시
    접촉(gap 요소)은 해석 중 변형에 따라 생기므로 표시하지 않는다.

좌표: x = 단면 폭 방향, y = 충돌 방향(−y가 차 안쪽), z = 기둥 길이 방향(sec 0 = 0, sec 16 = 1051.7 mm)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import bpillar_fem_demo as fem

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "02. post_processing" / "results" / "final_section_v1_21_0_x1.3_post.csv"
PART_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]    # Outer, Plate, Inner, Patch1, Patch2
LOAD_ARROW_SIZE = 5                 # 하중 화살표 크기 (plotly cone sizeref, 클수록 큼)


def line_trace(mesh, idx, name, color, width, group, show=True):
    """요소 목록 idx를 한 개의 선 trace로 (선분 사이 None으로 끊음)."""
    xyz = np.full((3 * len(idx), 3), np.nan)
    for r, e in enumerate(idx):
        i, j = mesh["elems"][e][:2]
        xyz[3 * r], xyz[3 * r + 1] = mesh["coords"][i], mesh["coords"][j]
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=name,
                        line=dict(color=color, width=width), legendgroup=group, hoverinfo="skip",
                        visible=True if show else "legendonly", connectgaps=False)


def node_trace(mesh, nodes, name, color, size, symbol, text=None, show=True):
    p = np.array([mesh["coords"][n] for n in nodes])
    return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
                        marker=dict(color=color, size=size, symbol=symbol, line=dict(width=0)),
                        text=text, hovertemplate="%{text}<extra></extra>" if text else None,
                        visible=True if show else "legendonly")


def build_figure(mesh, parts, n_sub, csv_name) -> go.Figure:
    fig = go.Figure()

    # 1. 파트별 요소
    for pid, p in enumerate(parts):
        for kind, label, width in (("circ", "section (circumferential)", 3), ("long", "longitudinal", 1)):
            idx = [e for e, (q, k) in enumerate(mesh["elem_info"]) if q == pid and k == kind]
            fig.add_trace(line_trace(mesh, idx, f"{p['name']} — {label} ({len(idx)})", PART_COLORS[pid],
                                     width, f"part{pid}"))

    # 2. 고정 경계조건
    ends = sorted({nd for s in (0, fem.N_SEC - 1) for nodes in mesh["sections"][s].values() for nd in nodes})
    fig.add_trace(node_trace(mesh, ends, f"Fixed BC: sec 0 & 16, 6 DOF ({len(ends)} nodes)", "black", 4,
                             "square", [f"fixed node {n}" for n in ends]))

    # 3. 플랜지 병합 절점
    joints = sorted(mesh["shared"])
    fig.add_trace(node_trace(mesh, joints, f"Flange joints: merged nodes ({len(joints)})", "#555555", 2,
                             "circle", [f"merged node {n}" for n in joints], show=False))

    # 4. 하중 화살표 (−y, 크기 = 하중 비율)
    ln = np.array(list(mesh["load"]))
    lf = np.array([mesh["load"][n] for n in ln])
    lp = np.array([mesh["coords"][n] for n in ln])
    fig.add_trace(go.Cone(x=lp[:, 0], y=lp[:, 1], z=lp[:, 2], u=np.zeros(len(ln)), v=-lf / lf.max(),
                          w=np.zeros(len(ln)), anchor="tip", sizemode="absolute", sizeref=LOAD_ARROW_SIZE,
                          colorscale=[[0, "#ff9999"], [1, "#cc0000"]], showscale=False, opacity=0.7,
                          name=f"Load: −y on Outer top, sec {fem.LOAD_SECS[0]}–{fem.LOAD_SECS[1]} "
                               f"({len(ln)} nodes)",
                          showlegend=True, customdata=lf * 100,
                          hovertemplate="load share %{customdata:.3f} %<extra></extra>"))

    # 5. 변위제어 절점
    fig.add_trace(node_trace(mesh, [mesh["control"]],
                             f"Control node: sec {fem.CONTROL_SEC} Outer pt {fem.CONTROL_PT} (−y)",
                             "gold", 9, "diamond", ["displacement-control node"]))

    # 6. 섹션 번호
    dz = fem.HEIGHT / (fem.N_SEC - 1)
    sec_x = [min(mesh["coords"][n][0] for n in mesh["sections"][s][fem.OUTER]) - 30 for s in range(fem.N_SEC)]
    fig.add_trace(go.Scatter3d(x=sec_x, y=[0] * fem.N_SEC, z=[s * dz for s in range(fem.N_SEC)],
                               mode="text", name="section labels",
                               text=[f"sec {s}" + (" ◀load" if fem.LOAD_SECS[0] <= s <= fem.LOAD_SECS[1] else "")
                                     for s in range(fem.N_SEC)],
                               textfont=dict(size=10, color="#333333"), hoverinfo="skip"))

    n_nodes, n_elem = len(mesh["coords"]), len(mesh["elems"])
    t = ", ".join(f"{p['name']} {p['t']:.3f}" for p in parts)
    fig.update_layout(
        title=dict(text=f"B-pillar lattice FEM model — {csv_name}<br>"
                        f"<sup>nodes {n_nodes}, beam elements {n_elem} (fiber strip, Corotational), "
                        f"n_sub = {n_sub}, height {fem.HEIGHT} mm | thickness (mm): {t}</sup>",
                   x=0.01),
        scene=dict(xaxis_title="x: width (mm)", yaxis_title="y: impact dir. (mm)",
                   zaxis_title="z: pillar axis (mm)", aspectmode="data",
                   camera=dict(eye=dict(x=1.6, y=1.1, z=0.6), up=dict(x=0, y=0, z=1))),
        legend=dict(x=0.0, y=0.93, bgcolor="rgba(255,255,255,0.8)", font=dict(size=11)),
        margin=dict(l=0, r=0, t=70, b=0), height=900,
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="단면 CSV")
    ap.add_argument("--n-sub", type=int, default=4, help="섹션 사이 분할 수 (해석 기본값 4)")
    ap.add_argument("--out", default=str(HERE / "FEM_visualization.html"), help="출력 HTML")
    a = ap.parse_args()

    parts = fem.read_parts(a.csv)
    mesh = fem.build_mesh(parts, a.n_sub, fem.HEIGHT / (fem.N_SEC - 1))
    fig = build_figure(mesh, parts, a.n_sub, Path(a.csv).name)
    fig.write_html(a.out, include_plotlyjs=True, full_html=True)       # 인터넷 없이 열리는 단독 HTML
    print(f"[done] 절점 {len(mesh['coords'])}, 요소 {len(mesh['elems'])}, 하중 절점 {len(mesh['load'])}, "
          f"고정 절점 {sum(len(set(v)) for s in (0, fem.N_SEC - 1) for v in mesh['sections'][s].values())} → {a.out}")


if __name__ == "__main__":
    main()
