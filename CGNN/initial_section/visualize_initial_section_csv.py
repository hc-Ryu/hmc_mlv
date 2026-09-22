"""
visualize_initial_section_csv.py — 학습 없이 원본 단면 CSV를 3D HTML로 시각화 (/synod design 세션 산출물)
──────────────────────────────────────────────────────────────
[배경] AI_design_v1_15_4.py의 plot_sections_3d_plotly_multi()는 Initial/Final 두 상태를
버튼으로 토글하는 3D HTML(reports/figures/AI_design_v1_15_4_3d.html)을 만들지만, final_coords/
final_z_gate 등 "학습된 모델의 결과"를 필수 인자로 요구한다. 아직 학습되지 않은 원본 CSV
(예: initial_section/initial_section_v2.csv) 하나만 보고 싶을 때는 이 함수를 그대로 쓸 수 없다.

[/synod design 세션 합의] gemini-3(flash, conf95)는 plot_sections_3d_plotly_multi()의
final_coords/final_z_gate를 Optional로 바꿔 재사용할 것을 제안했으나, 그 예시 시그니처가
기존 호출부(AI_design_v1_15_4.py의 positional call, part_ids/section_ids 인자 순서)를 깨뜨릴
위험이 있었다. openai-cli(o3, conf78)가 지적한 대로, 학습 파이프라인의 이미 동작하는 함수는
건드리지 않고(이 저장소가 uni_section_v20.py 등에서 계속 지켜온 "동작 코드 불변" 관례와 동일),
독립 스크립트에서 "Initial 상태 traces" 블록(AI_design_v1_15_4.py L1595-1621, BC 고정 노드
네모 오버레이 포함)만 최소 재구현하는 쪽을 Judge가 채택했다. 토글 버튼은 Final 상태가 없으므로
생략(Initial 하나만 항상 visible).

사용법:
    python visualize_initial_section_csv.py
    python visualize_initial_section_csv.py --csv <path> --output <path.html>
"""

import argparse
import os
import sys

import numpy as np
import plotly.graph_objects as go

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
# [위치 변경] 이 스크립트가 CGNN/initial_section/ 안으로 옮겨졌으므로, uni-section/code와
# reports/는 한 단계 위(CGNN 루트)를 기준으로 찾는다.
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'uni-section', 'code'))
import uni_section_v23 as usv18  # noqa: E402  (AI_design_v1_15_4.py와 동일한 별칭 관례)

# AI_design_v1_15_4.py와 완전히 동일한 값 — 그 파일을 import하지 않고 리터럴로 복제한 이유는
# import 시 torch/torch_geometric/matplotlib 전체 로드 + 학습 파이프라인 관련 assert가 함께
# 실행되는 것을 피하기 위함이다(이 스크립트는 시각화 전용, 학습 의존성 없음).
NUM_SECTIONS = 17
PART_NAMES = ['Outer', 'Plate', 'Inner', 'Patch1', 'Patch2']
PART_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

DEFAULT_CSV = os.path.join(_THIS_DIR, 'initial_section_v2.csv')
DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, 'reports', 'figures', 'initial_section_v2_3d.html')


def visualize_section_csv(csv_path, save_path, z_coef=15.0):
    """csv_path(floor_profiles_template.md 규격)를 로드해 Initial 상태 3D HTML 1개를 만든다.
    plot_sections_3d_plotly_multi()의 Initial-state 트레이스 로직(색상/선굵기/BC 고정 노드
    네모 오버레이)만 재사용한다 — Final state/토글 버튼 없음(학습 결과가 없으므로)."""
    data = usv18.build_bpillar_from_csv(csv_path, num_sections=NUM_SECTIONS)

    x = data.x.cpu().numpy()
    coords = x[:, 0:2]
    fix_np = ((x[:, 2] != 0) | (x[:, 3] != 0))
    part_ids_np = x[:, 4].astype(int)
    section_ids_np = x[:, 5].astype(int)

    n_sections = int(section_ids_np.max()) + 1 if len(section_ids_np) else 0
    if n_sections != NUM_SECTIONS:
        print(f"[warn] CSV의 실제 section 수({n_sections})가 NUM_SECTIONS({NUM_SECTIONS})와 "
              f"다릅니다 — 있는 그대로 시각화합니다.")

    fig = go.Figure()
    shown = set()
    for sec in range(n_sections):
        z = sec * z_coef
        for pid in range(len(PART_NAMES)):
            mask = (section_ids_np == sec) & (part_ids_np == pid)
            if not mask.any():
                continue
            pts = coords[mask]
            show_legend = pid not in shown
            shown.add(pid)
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=np.full(len(pts), z),
                mode='lines+markers',
                line=dict(color=PART_COLORS[pid], width=3),
                marker=dict(size=2, color=PART_COLORS[pid]),
                name=PART_NAMES[pid], legendgroup=PART_NAMES[pid], showlegend=show_legend,
                hovertemplate=f'Section {sec}, {PART_NAMES[pid]}<br>x=%{{x:.1f}}, y=%{{y:.1f}}<extra></extra>',
            ))

    if fix_np.any():
        idx = np.nonzero(fix_np)[0]
        zs = section_ids_np[idx] * z_coef
        fig.add_trace(go.Scatter3d(
            x=coords[idx, 0], y=coords[idx, 1], z=zs,
            mode='markers',
            marker=dict(size=5, symbol='square',
                        color=[PART_COLORS[p] for p in part_ids_np[idx]],
                        opacity=1.0),
            name='BC 고정 노드', legendgroup='fixed_nodes', showlegend=True,
            customdata=np.stack([section_ids_np[idx], part_ids_np[idx]], axis=1),
            hovertemplate=('BC 고정 노드<br>Section %{customdata[0]}, part_id=%{customdata[1]}'
                           '<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>'),
        ))

    fig.update_layout(
        title=f'{os.path.basename(csv_path)} — {n_sections}-Section B-Pillar (Drag to rotate, scroll to zoom)',
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)',
                   zaxis_title='Section (0=top/original ~ N-1=bottom)', aspectmode='data'),
        width=1100, height=850,
    )
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.write_html(save_path)
    print(f"[viz] 인터랙티브 3D 저장: {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="원본 단면 CSV를 학습 없이 3D HTML로 시각화")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                         help="floor_profiles_template.md 규격 CSV 경로")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                         help="저장할 HTML 경로 (기본: reports/figures/AI_design_v1_15_4_3d.html과 "
                              "겹치지 않도록 initial_section_v2_3d.html)")
    parser.add_argument("--z-coef", type=float, default=15.0,
                         help="섹션 간 z축 간격 배율 (mm 단위 아님, 시각화용 임의 스케일)")
    args = parser.parse_args()

    visualize_section_csv(args.csv, args.output, z_coef=args.z_coef)
