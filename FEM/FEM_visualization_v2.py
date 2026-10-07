"""bpillar_fem_v2.py 격자 모델 시각화 (메쉬 + 경계조건 + 하중 + 초기 접촉) → 인터랙티브 HTML.

FEM_visualization.py(bpillar_fem_demo.py용)를 v2 메쉬에 맞춘 판. 바뀐 점:
    - 파트 수·섹션 수를 CSV에서 읽는다 (v1.22·v23: 7파트 × 15섹션).
    - 고정 절점 = v2 메쉬의 양 끝 레벨 전 절점 (일부 섹션에만 있는 덧판도 끝 레벨에 닿으면 포함).
    - 플랜지 접합 중 Patch4 양 끝–Outer 접합(FEM 전용)을 따로 표시한다.
    - 해석 시작 시점의 접촉 gap 요소(위 파트 절점 → 아래 파트 절점, y 방향)를 표시한다 (기본 꺼짐).
      해석 중에는 변형에 따라 쌍이 더 생기므로, 이것은 시작 시점의 쌍만이다.

사용 (conda env fem, plotly 필요):
    python FEM_visualization_v2.py                                   # 기본: v23 후처리 단면, 해석과 같은 4분할 격자
    python FEM_visualization_v2.py --csv 설계.csv --n-sub 2 --out mesh_quick.html

보이는 것 (범례 클릭으로 켜고 끔)
    - 파트별 요소: 원주 방향(섹션 안 단면 선분, 굵은 선)과 종방향(레벨 사이 선분, 가는 선)
    - 고정 경계조건: 첫 섹션(sill), 마지막 섹션 전 절점 6자유도 고정
    - 플랜지 접합: 여러 파트가 하나로 병합된 절점 (Patch4 양 끝 접합은 따로)
    - 하중: 하중 구간 Outer 윗면 절점의 −y 방향 화살표 (크기 = 그 절점이 받는 하중 비율)
    - 변위제어 절점, 섹션 번호, 초기 접촉 gap 요소

좌표: x = 단면 폭 방향, y = 충돌 방향(−y가 차 안쪽), z = 기둥 길이 방향(sec 0 = 0, 섹션 피치 65.73 mm)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import openseespy.opensees as ops
import plotly.graph_objects as go

import bpillar_fem_v2 as fem

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "02. post_processing" / "results" / "v23" / "final_section_v23_ref_post.csv"
LOAD_ARROW_SIZE = 5                 # 하중 화살표 크기 (plotly cone sizeref, 클수록 큼)


def line_trace(segs, name, color, width, group, show=True):
    """선분 목록 [(p_i, p_j)]를 한 개의 선 trace로 (선분 사이 None으로 끊음)."""
    xyz = np.full((3 * len(segs), 3), np.nan)
    for r, (a, b) in enumerate(segs):
        xyz[3 * r], xyz[3 * r + 1] = a, b
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=name,
                        line=dict(color=color, width=width), legendgroup=group, hoverinfo="skip",
                        visible=True if show else "legendonly", connectgaps=False)


def node_trace(mesh, nodes, name, color, size, symbol, text=None, show=True):
    p = np.array([mesh["coords"][n] for n in nodes])
    return go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers", name=name,
                        marker=dict(color=color, size=size, symbol=symbol, line=dict(width=0)),
                        text=text, hovertemplate="%{text}<extra></extra>" if text else None,
                        visible=True if show else "legendonly")


def initial_contacts(mesh) -> list[tuple[int, int]]:
    """해석 시작 시점(변형 0)에 만들어지는 접촉 쌍 (위 절점, 아래 절점) — bpillar_fem_v2.Contact와 같은 규칙."""
    ops.logFile(os.devnull, "-noEcho")                                 # zeroLength 길이 경고 숨김 (해석에서도 무해)
    fem.build_model(mesh)
    contact = fem.Contact(mesh)
    contact.update()
    pairs = list(contact.pairs)
    ops.wipe()
    return pairs


def build_figure(mesh, parts, n_sub, csv_name, contacts) -> go.Figure:
    fig = go.Figure()
    coords, n_sec = mesh["coords"], mesh["n_sec"]

    # 1. 파트별 요소
    for pid, p in enumerate(parts):
        for kind, label, width in (("circ", "section (circumferential)", 3), ("long", "longitudinal", 1)):
            segs = [(coords[e[0]], coords[e[1]]) for e, (q, k) in zip(mesh["elems"], mesh["elem_info"])
                    if q == pid and k == kind]
            if segs:
                fig.add_trace(line_trace(segs, f"{p['name']} — {label} ({len(segs)})", fem.PART_COLORS[pid],
                                         width, f"part{pid}"))

    # 2. 고정 경계조건
    ends = sorted(mesh["ends"])
    fig.add_trace(node_trace(mesh, ends, f"Fixed BC: sec 0 & {n_sec - 1}, 6 DOF ({len(ends)} nodes)", "black", 4,
                             "square", [f"fixed node {n}" for n in ends]))

    # 3. 플랜지 병합 절점 — Patch4 양 끝 접합(FEM 전용)은 따로
    #    병합 대표는 Outer 절점(번호 = pid·100000 + lev·100 + point). Outer pt 9/22는 Patch4 접합으로만 병합된다.
    joints = sorted(mesh["shared"])
    p4_joint = [r for r in joints if r // 100000 == fem.OUTER and r % 100 in (9, 22)]
    others = sorted(set(joints) - set(p4_joint))
    fig.add_trace(node_trace(mesh, others, f"Flange joints: merged nodes ({len(others)})", "#555555", 2,
                             "circle", [f"merged node {n}" for n in others], show=False))
    if p4_joint:
        fig.add_trace(node_trace(mesh, p4_joint, f"Patch4 edge ↔ Outer pt 9/22 joints, FEM only ({len(p4_joint)})",
                                 fem.PART_COLORS[fem.PATCH4], 3, "diamond",
                                 [f"Patch4 edge joint {n}" for n in p4_joint], show=False))

    # 4. 하중 화살표 (−y, 크기 = 하중 비율)
    ln = np.array(list(mesh["load"]))
    lf = np.array([mesh["load"][n] for n in ln])
    lp = np.array([coords[n] for n in ln])
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

    # 6. 초기 접촉 gap 요소 (위 절점 → 아래 절점)
    if contacts:
        segs = [(coords[u], coords[l]) for u, l in contacts]
        fig.add_trace(line_trace(segs, f"Contact gaps at start, y-dir compression only ({len(segs)} pairs)",
                                 "#00bcd4", 4, "contact", show=False))

    # 7. 섹션 번호
    sec_x = [min(coords[n][0] for n in mesh["sections"][s][fem.OUTER]) - 30 for s in range(n_sec)]
    fig.add_trace(go.Scatter3d(x=sec_x, y=[0] * n_sec, z=[s * fem.DZ for s in range(n_sec)],
                               mode="text", name="section labels",
                               text=[f"sec {s}" + (" ◀load" if fem.LOAD_SECS[0] <= s <= fem.LOAD_SECS[1] else "")
                                     for s in range(n_sec)],
                               textfont=dict(size=10, color="#333333"), hoverinfo="skip"))

    t = ", ".join(f"{p['name']} {p['t']:.2f}" for p in parts)
    fig.update_layout(
        title=dict(text=f"B-pillar lattice FEM model v2 — {csv_name}<br>"
                        f"<sup>nodes {len(coords)}, beam elements {len(mesh['elems'])} (fiber strip, Corotational), "
                        f"n_sub = {n_sub}, height {(n_sec - 1) * fem.DZ:.1f} mm | thickness (mm): {t}</sup>",
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
    ap.add_argument("--n-sub", type=int, default=4, help="섹션 사이 분할 수 (해석 기본값 4, 짝수)")
    ap.add_argument("--out", default=str(HERE / "FEM_visualization_v2.html"), help="출력 HTML")
    a = ap.parse_args()

    parts = fem.read_parts(a.csv)
    mesh = fem.build_mesh(parts, a.n_sub)
    contacts = initial_contacts(mesh)
    fig = build_figure(mesh, parts, a.n_sub, Path(a.csv).name, contacts)
    fig.write_html(a.out, include_plotlyjs=True, full_html=True)       # 인터넷 없이 열리는 단독 HTML
    print(f"[done] 절점 {len(mesh['coords'])}, 요소 {len(mesh['elems'])}, 하중 절점 {len(mesh['load'])}, "
          f"고정 절점 {len(mesh['ends'])}, 초기 접촉 {len(contacts)}쌍 → {a.out}")


if __name__ == "__main__":
    main()
