#!/usr/bin/env python
# coding: utf-8

# ⚠ uni_section_v26.py는 uni_section_v25.py의 사본이며, initial_section_v5.csv(15층, 7파트)를
# 읽기 위한 CSV 로더 경로만 확장했다(AI_design_v1_22_0_ref.py 전용). 변경 지점:
#   - CANDIDATE_PARTS [3, 4] -> [3, 4, 5, 6]  (Patch3 = Patch2 좌우 대칭, Patch4 = Outer crown doubler)
#   - FY_BY_PART / EXPECTED_NODE_COUNT_BY_PART 에 part 5, 6 추가
#   - _build_fix_mask_table(): part 5(전 노드 고정, Patch2와 동일 규칙), part 6(전 노드 자유) 추가
#   - _parse_floor_profiles_csv()/build_bpillar_from_csv(): 파트 수·floor 수 하드코딩(5/17) 제거,
#     NUM_SECTIONS_CSV(=15) 기본값 + 인자로 검증
# 물리/손실/게이트 함수는 v25와 완전히 동일하다. build_bpillar_section()·run_training() 등 CSV 경로가
# 아닌 레거시(5파트 x_ratio 생성) 함수는 참고용으로 그대로 두었다 — AI_design_v1_22_0_ref.py는 쓰지 않는다.
# [후속] initial_section_v6.csv(v5 + 하단 flare, x만 floor별 affine 변환)도 토폴로지가 v5와 같아 이
# 로더를 그대로 쓴다 — fix 마스크는 point_idx 기반 정적 테이블이라 좌표 변화에 영향받지 않는다.

# ⚠ uni_section_v25.py는 uni_section_v24.py의 사본이며, 수정 없이 그대로 재사용한다 — 이미 v24가
# _build_fix_mask_table()에 Plate(#03) point_idx {4,5,6,25,26,27} 고정 해제(원칙 3)를 구현해
# 두고 있고, 사용자가 v1.16의 export 시 강체 정렬(align_flange_pairs_rigid(), 원칙 1/2)만 폐기하고
# 이 고정 해제 한 가지만 AI_design_v1_15_5.py 기반(v1.15.6)에 적용하길 원했기 때문이다. v24와
# 완전히 동일한 코드이며 버전 번호만 v1.15.6 계열에 맞춰 새로 붙였다(docs/review/review_v1_16.md
# 권장 A/B 격리 테스트의 "원칙 3만" 조합에 해당).

# ⚠ uni_section_v24.py는 uni_section_v23.py의 사본이며, _build_fix_mask_table() 1개 함수만
# 수정했다(Plate(#03) point_idx {4,5,6,25,26,27} 고정 해제, docs/command/command_v1_16.md §2).
# 그 외 모든 함수는 v23과 완전히 동일하다.

# ⚠ uni_section_v23.py는 uni_section_v22.py의 사본이며, build_collision_spec()과
# compute_collision_loss_v5() 2개 함수만 수정했다(fixed-fixed 항목 제외,
# docs/command/command_v1_15.md §7). 그 외 모든 함수는 v22와 완전히 동일하다.

# ⚠ uni_section_v22.py는 uni_section_v21.py의 사본이며, compute_collision_loss_v5() 1개 함수만
# 수정했다(direction-레벨 Top-K 풀링, docs/command/command_v1_14.md §1). build_bpillar_from_csv()를
# 포함한 그 외 모든 함수는 v21과 완전히 동일하다.

# ⚠ uni_section_v21.py는 uni_section_v20.py의 사본이며, 초기 형상 생성 방식만 다르다.
# build_bpillar_section()(x_ratio 규칙 기반 하드코딩 생성)은 그대로 남겨 참고/회귀비교용으로
# 보존하고, 신규 함수 build_bpillar_from_csv(csv_path)가 floor_profiles_template.md 규격 CSV
# (경로는 호출부 인자, 현재 AI_design_v1_13.py는 initial_section/initial_section.csv를 전달)를
# 로드해 17-section Data를 직접 반환한다 — build_bpillar_17section()은 더 이상 build_bpillar_
# section()을 17회 호출+scale_factor()로 스케일링하지 않는다(CSV가 이미 floor별 최종 실좌표를
# 담고 있음). 그 외 모든 함수는 uni_section_v20.py와 동일(fy/target_mp 등 학습 로직은 v20 그대로,
# 이 파일은 건드리지 않았다).

"""
uni_section_v23.py
───────────────────────────────────
[v23 변경, docs/command/command_v1_15.md §7 실행]
  build_collision_spec(): fix_mask 인자 추가. 각 direction에 로컬 인덱스 공간의
  ff_mask (n_pts, n_segs) 를 함께 저장한다 — "점도 고정이고 투영 대상 세그먼트의 양 끝도
  모두 고정"인 조합(AND)만 True다.
  compute_collision_loss_v5(): 그 ff_mask 항목을 valid에서 제외한다.

  근거: v1.14 잔여 겹침(Plate-Inner 0.4119mm, Outer-Plate 0.0620mm)은 용접 플랜지가
  맞닿도록 설계된 상태에서 두께만 자라고 중심선은 벌어지지 못해 생긴 모델링
  아티팩트다(review_v1_13.md / idea_v1_15.md §0, 예측값이 실측과 4자리 일치).
  양쪽 노드가 모두 BC 고정이라 좌표 자유도가 0이므로 collision loss는 만족
  불가능한 제약을 강요해 왔다 — 이제 AI_design_v1_15의 두께 연동 오프셋 레이어가
  구조적으로 처리한다.
───────────────────────────────────
uni_section_v22.py
─────────────────────────────────────
[v22 변경, docs/command/command_v1_14.md §1 실행]
  compute_collision_loss_v5(): direction(섹션×파트쌍×방향) 레벨 집계를 단순 평균(total_loss /
  n_dirs)에서 Top-K 평균으로 교체. 각 direction 내부의 Top-K(top_k_fraction=0.3)는 v20부터 이미
  있었으나, direction들 "사이"의 합산은 여전히 전체 평균이라 실제 위반이 발생하는 극소수
  파트쌍(v1.13 결과 기준 Part0-1/1-2/1-4의 일부 floor)의 그래디언트가 희석됐다
  (review_v1_13.md #1 — 근본 원인 1위). compute_mesh_order_loss()(v19)·compute_smoothness_lse_v3()
  (v1.10)에서 이미 두 번 적용된 것과 동일한 "mean 희석" 해소 패턴을 이 함수에도 뒤늦게 적용한다.

  [실측 정정 1] review_v1_13.md/command_v1_14.md는 direction 수를 153개로 적었으나, 153은
  collision_spec의 (sec, a, b) '키' 개수이고 실제 direction(각 키가 갖는 방향 dict) 수는 289개다
  (initial_section_v2.csv 기준 실측). 즉 희석 배수는 1/153이 아니라 1/289였다.

  [실측 정정 2] 같은 두 문서는 "희석된 l_collision이 tau_col(0.02) 미만이라 g_col<0 → relu가 죽는
  dead zone에 빠진다"고 설명했으나, 실측(전 파트 t=2.3mm 균일, 초기 좌표 기준)에서는 v21 방식도
  l_collision=0.0901 > tau_col이라 dead zone에 있지 않았다. 이 수정의 실제 효과는 "dead zone
  탈출"이 아니라 g_col 자체의 증폭이다: 0.0901 -> 0.3214 (3.57배), 즉 term_col의 그래디언트
  (∂term_col/∂l_collision = rho_col * relu(g_col + mu_col/rho_col))가 그만큼 커진다.
  물리(Mp) 손실과의 경쟁력이 3.6배 올라가는 것이 이 패치의 실질적 기여다.

  top_k_dir_fraction<=0 또는 >=1.0을 넘기면 v21과 동일한 전체 평균으로 폴백한다(ablation 대조용,
  실측 상대오차 8e-08 — float32 합산 순서 차이만 남는다).
─────────────────────────────────────
uni_section_v21.py
─────────────────────────────────────
[v21 변경, /synod design 세션(Gemini flash 95 / OpenAI o3 93, 둘 다 can_exit=true) 반영]
  floor_profiles_template.md 규격 CSV(예: initial_section/initial_section.csv)를 로드하는
  build_bpillar_from_csv(csv_path)를 추가. 설계 결정:
    - CSV 파싱·BC(fix_x/fix_y) 복원·재료(fy) 복원·엣지 재구성은 전부 이 물리 정의 계층
      (uni_section)이 책임진다 — AI_design은 완성된 PyG Data만 받는다.
    - fix_x/fix_y는 좌표 역산이 아니라 (part_id, point_idx) 기반 정적 룩업 테이블로 복원한다
      (파트별 floor당 점 개수가 30/30/18/12/3으로 항상 고정이고, 원본 build_bpillar_section()의
      x_ratio 조건과 1:1 대응되므로 좌표 무관하게 안전).
    - fy는 part_id별 고정 상수(FY_BY_PART, build_bpillar_section()의 part_configs와 동일값).
  그 외 모든 함수는 uni_section_v20.py와 동일.
─────────────────────────────────────
uni_section_v20.py
─────────────────────────────────────
[v20 변경, docs/command/command_v1_9.md 실행]
  compute_collision_loss_v5(): 방향별 위반량 sum()/valid.sum() 평균 → Top-K(PHYS_TOPK_FRACTION=0.3)
  평균 풀링. 한두 노드의 깊은 국소 관통(송곳형)이 다수의 정상 노드에 의해 평균에서 희석되던 문제를
  compute_mesh_order_loss()(v19)와 동일한 패턴으로 해소(review_v1_8.md §9).
  그 외 모든 함수는 uni_section_v19.py와 동일.
─────────────────────────────────────
uni_section_v19.py
─────────────────────────────────────
[v19 변경, docs/command/command_v1_8.md 실행]
  compute_mesh_order_loss(): torch.mean() → Top-K(top_k_fraction=0.02) 평균 풀링.
  v1.4~v1.7 내내 이 함수의 mean 희석이 "Order 손실 무한정 증가"의 실제 원인이었으나
  불변 파일 내부에 있어 발견되지 못했다(AI_design_v1_8.py /synod idea 세션에서 최초 확인).
  그 외 모든 함수는 uni_section_v18.py와 동일.
─────────────────────────────────────
uni_section_v18.py
─────────────────────────────────────
uni_section_v17_01.py를 docs/command/command_v18.md 지시 사항에 따라 재작성한 버전.
설계 근거: /synod idea 세션(idea_v18 요지: AI_design_v0.py의 B-pillar 3-part hat 프로파일에서
          비율적 영감만 얻어 uni_section의 5-part 초기 좌표를 더 현실적인 hat 형상으로 바꾸는
          방법 평가, conf 88 — Gemini avg 93.5/OpenAI avg 77.5, Critic 라운드에서 non-penetration
          순서 보존과 트리밍 범위 정합성을 핵심 리스크로 확정) → command_v18.md(구현 지시서) →
          /synod design 세션(v18 파일 패키징 방식 결정: 헤더 docstring/콘솔 배너/출력 PNG 파일명은
          v18로 갱신하고, build_bpillar_section() 이외의 모든 함수/클래스/로직은 v17_01.py와 완전히
          동일하게 유지 — Critic 라운드에서 "everything else unchanged"는 build_bpillar_section()의
          토폴로지/BC/노드-엣지 구조 보존을 의미하며, 파일 전체의 바이트 단위 diff 제약이 아님을
          확인. 이 저장소의 기존 관례(v15→v16→v17→v17_paratune)도 매 버전마다 헤더 docstring이
          해당 버전의 변경점을 서술해왔음).

v17_01 대비 변경점(command_v18.md 전체 반영, cosmetic-only):
  build_bpillar_section() 안에서 Part 0(Outer Hat)과 Part 2(Inner Hat)의 y_coord_node 대입값만
  수정. 두 part 모두 fix=1인 flange 분기(if)는 그대로 두고, else 분기의 상수 대입을 신규 헬퍼
  `_hat_profile()`(flange→ramp→crown→ramp→flange 트라페조이드 선형 보간) 호출로 교체해, 기존의
  평평한 계단형 초기 형상을 AI_design_v0.py의 hat 실루엣에 비율적으로 가까운 형태로 바꿨다.
  Part 1(Inner Plate)/3(Patch1)/4(Patch2)의 y_coord_node 대입식은 변경하지 않았다(command_v18.md
  §2: 이미 3단 이상으로 나뉘어 있어 collision 리스크 대비 추가 개선 이득이 작다고 판단).
  crown 값은 Part 0(65.0) > Part 2(45.0) 순서를 유지해 기존 collision loss가 가정하는 비관통
  순서(outer > inner hat > inner plate)를 그대로 보존한다.

v17_01에서 그대로 유지되는 부분(build_bpillar_section() 이외 전부):
  node 개수/노드 x-spacing(30개, dx=160/29), part 2/3/4 트리밍 범위, fix_x/fix_y 배정 로직,
  t_val/fy_val, add_edge() 엣지 연결, join_pairs(빈 텐서), CGDN 모델, HardConcrete 게이팅/적응형
  TAU_GATE, ALDA, collision v5, train_step/run_training 전체, 시각화 함수. TARGET_MP/weights 등
  __main__ 하이퍼파라미터도 동일하게 유지했다(command_v18.md 범위 밖).
"""

import argparse
import csv
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Gulim'  # Windows 한글 폰트

from torch_geometric.nn import GATv2Conv, LayerNorm
from torch_geometric.data import Data


# ══════════════════════════════════════════════════════════════════
# SECTION 0: Mp 계산 — native autograd (v8/v10/v15/v16/v17/v17_01 §3.1 유지)
# ══════════════════════════════════════════════════════════════════

def compute_edge_mp_pna(coords, t, fy, edge_index, n_iter=50):
    """Thick Edge (2D Plate) PNA 이분탐색 + Mp 계산 (v17 그대로). Returns: (mp_total, y_pna)"""
    mask = edge_index[0] < edge_index[1]
    u, v = edge_index[0][mask], edge_index[1][mask]

    y_u, y_v = coords[u, 1], coords[v, 1]
    x_u, x_v = coords[u, 0], coords[v, 0]
    L = torch.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2)
    t_e  = t[u].squeeze(-1)
    fy_e = fy[u].squeeze(-1)

    dx   = torch.abs(x_u - x_v)
    t_y  = t_e * (dx / (L + 1e-12))
    y_max = torch.maximum(y_u, y_v)
    y_min = torch.minimum(y_u, y_v)
    y_top = y_max + t_y / 2.0
    y_bot = y_min - t_y / 2.0
    H     = torch.clamp(y_top - y_bot, min=1e-12)

    Area_fy = L * t_e * fy_e

    with torch.no_grad():
        y_lo = coords[:, 1].min().clone() - 5.0
        y_hi = coords[:, 1].max().clone() + 5.0
        for _ in range(n_iter):
            y_mid = 0.5 * (y_lo + y_hi)
            alpha = torch.clamp((y_top - y_mid) / H, 0.0, 1.0)
            net_force = torch.sum(Area_fy * (2.0 * alpha - 1.0))
            if net_force > 0:
                y_lo = y_mid
            else:
                y_hi = y_mid
        y_pna = 0.5 * (y_lo + y_hi)

    # 중앙부 측면 충돌 기준: y > y_pna = 압축부(comp), y < y_pna = 인장부(tens).
    # 현재는 fy가 인장/압축 대칭이라 이 라벨이 y_pna/Mp/기울기 수치에 영향을 주지 않으나,
    # 좌굴(유효폭) 항이나 비대칭 재료를 도입하면 압축부 = y > y_pna 구역이 기준이 된다.
    alpha         = torch.clamp((y_top - y_pna) / H, 0.0, 1.0)   # y_pna 위쪽(압축부) 비율
    centroid_comp = y_top - (alpha * H) / 2.0
    centroid_tens = y_bot + ((1.0 - alpha) * H) / 2.0
    m_comp        = alpha * (centroid_comp - y_pna)
    m_tens        = (1.0 - alpha) * (y_pna - centroid_tens)
    mp_total      = torch.sum(Area_fy * (m_comp + m_tens))

    return mp_total, y_pna


def calculate_mpl(coords, t, fy, edge_index):
    mp_total, _ = compute_edge_mp_pna(coords, t, fy, edge_index)
    return mp_total


def verify_thickness_gradient(coords, t, fy, edge_index, eps=1e-4):
    """[v8/v10/v15/v16/v17/v17_01 §3.1 유지] 학습 전 유한차분으로 ∂Mp/∂t 검증 — 실패 시 학습 중단."""
    coords = coords.detach()
    fy = fy.detach()

    t_leaf = t.detach().clone().requires_grad_(True)
    mp, _ = compute_edge_mp_pna(coords, t_leaf, fy, edge_index)
    mp.backward()
    if t_leaf.grad is None:
        raise RuntimeError("[gradcheck] dMp/dt 가 None!")
    grad_sum = t_leaf.grad.sum().item()

    with torch.no_grad():
        mp_p, _ = compute_edge_mp_pna(coords, t + eps, fy, edge_index)
        mp_m, _ = compute_edge_mp_pna(coords, t - eps, fy, edge_index)
    fd = (mp_p.item() - mp_m.item()) / (2.0 * eps)

    rel_err = abs(fd - grad_sum) / (abs(fd) + 1e-8)
    if fd <= 0:
        raise RuntimeError(f"[gradcheck] dMp/dt = {fd:.4e} <= 0 -- 물리적으로 비정상!")
    if rel_err > 1e-3:
        raise RuntimeError(f"[gradcheck] autograd({grad_sum:.6e}) vs FD({fd:.6e}) "
                           f"상대오차 {rel_err:.2e} > 1e-3")
    print(f"[gradcheck] dMp/dt OK -- autograd={grad_sum:.4e}, FD={fd:.4e}, "
          f"rel_err={rel_err:.2e}, dMp/dt>0 확인")
    return grad_sum, fd, rel_err


class FiLMGenerator(nn.Module):
    """target_mp [B, 1] → (gamma, beta) [B, hidden] (v17 그대로)"""
    MP_SCALE = 1e6

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, hidden_channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, target_mp):
        target_mp_norm = target_mp / self.MP_SCALE
        out = self.net(target_mp_norm)
        delta_gamma, beta = torch.chunk(out, 2, dim=-1)
        gamma = 1.0 + delta_gamma
        return gamma, beta


class CGDNBlock(nn.Module):
    """GATv2Conv → LayerNorm → FiLM → GELU → Residual (v17 그대로)"""
    def __init__(self, hidden_channels: int, heads: int = 4, edge_dim: int = 4):
        super().__init__()
        assert hidden_channels % heads == 0
        self.conv = GATv2Conv(
            hidden_channels,
            hidden_channels // heads,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
        )
        self.norm = LayerNorm(hidden_channels)

    def forward(self, h, edge_index, edge_attr, gamma, beta):
        h_res = h
        h = self.conv(h, edge_index, edge_attr)
        h = self.norm(h)
        h = gamma * h + beta
        h = F.gelu(h)
        h = h + h_res
        return h


# ══════════════════════════════════════════════════════════════════
# [command_v16.md §1 유지] 파트 존재 게이트 — 이원화(z_gate/z_open)
# ══════════════════════════════════════════════════════════════════

S_PROTECT = [0, 1, 2]        # [command_v17.md §1] Inner Hat(#06, part_id=2) 추가 보호
CANDIDATE_PARTS = [3, 4, 5, 6]  # [v26] Patch1(#07), Patch2(#08), Patch3, Patch4 — 모두 삭제 후보
                                 # (CGDN17의 cand_col = part_id - CANDIDATE_PARTS[0] 산술 때문에 연속 id여야 함)
                              # [command_v17_paratune.md 범위 밖] target별 동적화는 이번 범위 제외


def compute_gates(log_alpha, training=True, temperature=0.5, gamma=-0.1, zeta=1.1, s_protect=None):
    """[v16/v17/v17_01 §1 그대로] HardConcrete relaxation, 게이트 이원화.

    temperature: [command_v17_paratune.md Phase 3] 호출부(CGDN.forward)가 이제 실제로 동적
    온도를 전달한다 — v17까지는 이 인자가 기본값 0.5로 고정된 채 한 번도 오버라이드되지 않았다.
    """
    s_det = torch.sigmoid(log_alpha)
    z_gate = torch.clamp(s_det * (zeta - gamma) + gamma, 0.0, 1.0)

    if training:
        u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
        logistic_noise = torch.log(u) - torch.log(1.0 - u)
        s_stoch = torch.sigmoid((log_alpha + logistic_noise) / temperature)
        z_open = torch.clamp(s_stoch * (zeta - gamma) + gamma, 0.0, 1.0)
    else:
        z_open = z_gate

    if s_protect:
        idx = torch.tensor(s_protect, dtype=torch.long, device=log_alpha.device)
        z_gate = z_gate.clone()
        z_open = z_open.clone()
        z_gate[idx] = 1.0
        z_open[idx] = 1.0

    return z_gate, z_open


def make_pruning_state(num_parts=5, candidate_parts=None):
    """[v16/v17/v17_01 §3 유지 + command_v17_paratune.md Phase 1] EWMA+히스테리시스 프루닝 상태
    + 쿨다운 카운터 + 적응형 TAU_GATE 상태(best-so-far 단조 추적 + EMA + 현재 임계값)."""
    if candidate_parts is None:
        candidate_parts = CANDIDATE_PARTS
    return {
        'ewma_z': torch.ones(num_parts),
        'state': ['ALIVE'] * num_parts,          # 'ALIVE' | 'PENDING_DELETE' | 'DELETED'
        'pending_duration': torch.zeros(num_parts),
        'ewma_alpha': 0.15,
        'thresh_low': 0.08,     # [command_v17_paratune.md Phase 2] diagnostic 실측 없이는 변경 금지
        'thresh_high': 0.15,    # 상동
        'confirm_epochs': 20,
        's_protect': S_PROTECT,
        'candidate_parts': candidate_parts,
        'cooldown_remaining': 0,   # [command_v17.md §4] State-Only Reset 이후 쿨다운
        # ── [command_v17_paratune.md Phase 1] 적응형 TAU_GATE 상태 (신규) ──
        'best_so_far_mp_rel_err': float('inf'),   # 단조 비증가 running minimum
        'ema_best_so_far': TAU_CEILING,           # 초기값: 가장 관대한 상한에서 시작
        'current_tau_gate': TAU_CEILING,
    }


@torch.no_grad()
def update_pruning_state(pruning_state, z_gate_per_part):
    """[v16/v17/v17_01 §3.2 그대로] EWMA+히스테리시스 상태 갱신. Returns: 새로 DELETED 확정된 파트 id 리스트."""
    newly_deleted = []
    a = pruning_state['ewma_alpha']
    for i in pruning_state['candidate_parts']:
        if pruning_state['state'][i] == 'DELETED':
            continue

        z_i = float(z_gate_per_part[i])
        pruning_state['ewma_z'][i] = a * z_i + (1 - a) * pruning_state['ewma_z'][i]
        ez = pruning_state['ewma_z'][i].item()

        if pruning_state['state'][i] == 'ALIVE':
            if ez < pruning_state['thresh_low']:
                pruning_state['state'][i] = 'PENDING_DELETE'
                pruning_state['pending_duration'][i] = 0
        elif pruning_state['state'][i] == 'PENDING_DELETE':
            if ez > pruning_state['thresh_high']:
                pruning_state['state'][i] = 'ALIVE'
                pruning_state['pending_duration'][i] = 0
            else:
                pruning_state['pending_duration'][i] += 1
                if pruning_state['pending_duration'][i] >= pruning_state['confirm_epochs']:
                    pruning_state['state'][i] = 'DELETED'
                    newly_deleted.append(i)
    return newly_deleted


class CGDN(nn.Module):
    """
    Constraint-aware Graph Deformation Network v18 (백본은 v17/v17_01과 동일)
    forward() 반환값: (new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open)
    """

    DELTA_SCALE = 1.35
    T_MIN       = 0.1    # mm
    T_MAX       = 2.5    # mm — 관찰만, 수정은 범위 밖

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        edge_dim: int = 4,
        max_displacement: float = 50.0,
        num_parts: int = 5,
        init_log_alpha: float = 2.0,
    ):
        super().__init__()
        self.max_displacement = max_displacement
        self.num_layers = num_layers
        self.num_parts = num_parts
        self.s_protect = S_PROTECT

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            LayerNorm(hidden_channels),
            nn.GELU(),
        )
        self.film_generators = nn.ModuleList([
            FiLMGenerator(hidden_channels) for _ in range(num_layers)
        ])
        self.blocks = nn.ModuleList([
            CGDNBlock(hidden_channels, heads=heads, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        self.coord_decoder = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        self.thickness_decoder = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.thickness_decoder[-1].bias, 0.03)

        self.log_alpha = nn.Parameter(torch.full((num_parts,), init_log_alpha))

    @staticmethod
    def leaky_tanh(x):
        """0.95·tanh(x) + 0.05·x/3 — gradient 하한 확보 (v8~v17_01 §3.4 유지)"""
        return 0.95 * torch.tanh(x) + 0.05 * x / 3.0

    def forward(self, x, edge_index, edge_attr, target_mp,
                fix_x_mask, fix_y_mask, join_pairs=None, thickness_gate=1.0,
                gate_active=True, temperature=0.5):
        """[command_v17_paratune.md Phase 3] temperature 인자 추가 — compute_gates로 실제 전달.
        v17까지는 이 인자 자체가 없어 compute_gates가 항상 기본값 0.5로 호출됐다(배관 누락)."""
        h = self.node_encoder(x)

        for i, block in enumerate(self.blocks):
            gamma, beta = self.film_generators[i](target_mp)
            h = block(h, edge_index, edge_attr, gamma, beta)

        # ── 좌표 예측 ──
        delta_coords = self.coord_decoder(h)
        delta_coords = torch.clamp(delta_coords, -self.max_displacement, self.max_displacement)
        delta_x = delta_coords[:, 0:1] * (~fix_x_mask).float()
        delta_y = delta_coords[:, 1:2] * (~fix_y_mask).float()
        delta_coords = torch.cat([delta_x, delta_y], dim=1)
        new_coords = x[:, :2] + delta_coords

        if join_pairs is not None and join_pairs.shape[0] > 0:
            u_idx = join_pairs[:, 0]
            v_idx = join_pairs[:, 1]
            mid = (new_coords[u_idx] + new_coords[v_idx]) * 0.5
            new_coords = new_coords.clone()
            new_coords[u_idx] = mid
            new_coords[v_idx] = mid

        # ── 두께 예측 ──
        delta_t_raw = self.thickness_decoder(h)
        delta_t_raw = self.leaky_tanh(delta_t_raw) * self.DELTA_SCALE

        part_ids_local    = x[:, 4].long()
        section_ids_local = x[:, 5].long()
        t_initial         = x[:, 6].unsqueeze(1)

        max_parts = int(part_ids_local.max().item()) + 1
        composite_key = section_ids_local * max_parts + part_ids_local
        _, inverse = torch.unique(composite_key, return_inverse=True)
        num_groups = int(inverse.max().item()) + 1

        delta_t_1d  = delta_t_raw.squeeze(-1)
        group_sum   = torch.zeros(num_groups, device=x.device).scatter_add_(0, inverse, delta_t_1d)
        group_count = torch.zeros(num_groups, device=x.device).scatter_add_(0, inverse, torch.ones_like(delta_t_1d))
        group_mean  = group_sum / group_count.clamp(min=1)
        delta_t_part = group_mean[inverse].unsqueeze(-1)

        delta_t_part = delta_t_part * thickness_gate

        t_min, t_max = self.T_MIN, self.T_MAX
        t_initial_frac = (t_initial - t_min) / (t_max - t_min)
        t_initial_frac = torch.clamp(t_initial_frac, 1e-4, 1.0 - 1e-4)
        t_initial_logit = torch.logit(t_initial_frac)
        t_raw = t_min + (t_max - t_min) * torch.sigmoid(t_initial_logit + delta_t_part)

        if gate_active:
            z_gate, z_open = compute_gates(self.log_alpha, training=self.training,
                                            temperature=temperature, s_protect=self.s_protect)
        else:
            z_gate = torch.ones(self.num_parts, device=x.device)
            z_open = torch.ones(self.num_parts, device=x.device)

        part_gate_node = z_gate[part_ids_local].unsqueeze(-1)
        t_final = t_raw * part_gate_node

        return new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open


# ══════════════════════════════════════════════════════════════════
# SECTION 1: Loss Functions
# ══════════════════════════════════════════════════════════════════

def asymmetric_huber_phys(pred_mp, target_mp, delta=0.05, under_w=2.0):
    """[v10~v17_01 §5 유지] 비대칭 Huber phys loss."""
    err = (pred_mp - target_mp) / target_mp
    abs_err = err.abs()
    huber = torch.where(abs_err <= delta,
                        0.5 * abs_err ** 2 / delta,
                        abs_err - 0.5 * delta)
    w = torch.where(err < 0,
                    torch.full_like(err, under_w),
                    torch.ones_like(err))
    return w * huber


def compute_smoothness_loss_angle(new_coords, edge_index, edge_attr):
    """노드별 좌우 엣지 각도 최소화 & 90도 미만 제한 (v17 그대로)"""
    src, dst = edge_index
    edge_type = edge_attr[:, 3]

    mask = (src < dst) & torch.isclose(edge_type, torch.zeros_like(edge_type))
    if not mask.any():
        return torch.tensor(0.0, device=new_coords.device)

    src = src[mask]
    dst = dst[mask]

    num_nodes = new_coords.shape[0]
    all_u = torch.cat([src, dst])
    all_v = torch.cat([dst, src])

    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(all_u.tolist(), all_v.tolist()):
        adj[u].append(v)

    left_angles = []
    right_angles = []

    for node, neighbors in enumerate(adj):
        if len(neighbors) < 2:
            continue

        node_x = new_coords[node, 0]
        left_nodes = [n for n in neighbors if new_coords[n, 0] < node_x]
        right_nodes = [n for n in neighbors if new_coords[n, 0] > node_x]

        if len(left_nodes) != 1 or len(right_nodes) != 1:
            continue

        left_node = left_nodes[0]
        right_node = right_nodes[0]

        left_vec = new_coords[node] - new_coords[left_node]
        right_vec = new_coords[right_node] - new_coords[node]

        left_angles.append(torch.atan2(left_vec[1], left_vec[0]))
        right_angles.append(torch.atan2(right_vec[1], right_vec[0]))

    if len(left_angles) == 0:
        return torch.tensor(0.0, device=new_coords.device)

    left_angles = torch.stack(left_angles)
    right_angles = torch.stack(right_angles)

    results = 0.0

    angle_diff = (left_angles - right_angles + math.pi) % (2.0 * math.pi) - math.pi
    results += torch.mean(angle_diff.pow(2))

    max_rad = math.pi / 2.0
    left_violation = torch.relu(left_angles.abs() - max_rad)
    right_violation = torch.relu(right_angles.abs() - max_rad)
    results += torch.mean(left_violation.pow(2) + right_violation.pow(2))

    return results


def compute_mass_loss(new_coords, t, edge_index, edge_attr, target_area):
    """[command_v15.md A-2 유지] 비대칭 mass loss. area는 게이팅된 t를 통해 자동 게이팅됨."""
    src, dst = edge_index
    edge_type = edge_attr[:, 3]

    mask = (src < dst) & torch.isclose(edge_type, torch.zeros_like(edge_type))
    src = src[mask]
    dst = dst[mask]

    seg_len = torch.norm(new_coords[src] - new_coords[dst], dim=1)
    t_src = t[src].squeeze(-1)
    area = torch.sum(seg_len * t_src)

    l_mass = torch.relu(area / (target_area + 1e-12) - 1.0) ** 2
    return area, l_mass


def compute_alda_loss(area, target_area, pred_mp_total, target_mp_total,
                       l_collision, alda_state):
    """[command_v15.md §1 유지] Augmented Lagrangian Dual-Ascent."""
    f = area / (target_area + 1e-12)

    g_Mp = (pred_mp_total - target_mp_total).abs() / (target_mp_total.abs() + 1e-12) - alda_state['tau_Mp']
    g_col = l_collision - alda_state['tau_col']

    term_Mp = (alda_state['rho_Mp'] / 2.0) * torch.relu(
        g_Mp + alda_state['mu_Mp'] / alda_state['rho_Mp']) ** 2
    term_col = (alda_state['rho_col'] / 2.0) * torch.relu(
        g_col + alda_state['mu_col'] / alda_state['rho_col']) ** 2

    L_alda = f + term_Mp + term_col
    return L_alda, g_Mp.detach().item(), g_col.detach().item()


def compute_anchor_loss(new_coords, base_coords, fix_x_mask, fix_y_mask):
    disp = new_coords - base_coords
    return torch.mean(disp[:, 0] ** 2 + disp[:, 1] ** 2)


def compute_saturation_loss(delta_t_part, delta_scale=1.35, knee=0.9):
    """L_sat = relu(|delta_t| - knee×scale)^2 (v17 그대로)"""
    threshold = knee * delta_scale
    return torch.mean(torch.relu(delta_t_part.abs() - threshold) ** 2)


def compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr, eps=0.5,
                             top_k_fraction=0.02):
    """[v19 §1] 2D Mesh Order Loss — Top-K 평균 풀링으로 교체.
    [정정 이력] v1.4~v1.7 내내 이 함수는 mean()으로 전체 물리 엣지(~2992개)에 걸쳐
    평균을 냈다 — compute_smoothness_loss_angle()에서 이미 발견·수정했던 것과 동일한
    희석 버그가 "Order 손실"이라는 이름으로 4개 버전 내내 계속 관찰되고도 원인 불명이었던
    것은 이 함수가 그동안 불변 파일 안에 있었기 때문이다(design 세션에서 최초 확인).
    LogSumExp가 아니라 Top-K를 쓰는 이유: v1.7이 온도 미보정(T=0.01)으로 겪은
    exp(deviation/T) 오버플로우 위험이 Top-K에는 구조적으로 없다(지수 연산 자체가 없음) —
    design 세션에서 수학적으로 검증(exp(10/0.01)=exp(1000)로 float32 확정적 오버플로우)."""
    src, dst = edge_index
    mask = (src < dst) & (edge_attr[:, 3] == 0.0)
    if not mask.any():
        return torch.tensor(0.0, device=new_coords.device)

    e0 = base_coords[dst[mask]] - base_coords[src[mask]]
    e0_hat = e0 / (e0.norm(dim=1, keepdim=True) + 1e-8)
    e_new = new_coords[dst[mask]] - new_coords[src[mask]]
    proj = (e_new * e0_hat).sum(dim=1)
    violation = torch.relu(eps - proj) ** 2
    violation = violation.view(-1)   # [v19, 구현 세션 방어 코드] 항상 1D이지만 안전을 위해 명시
    k = max(1, int(violation.numel() * top_k_fraction))   # 전체 ~2992개 중 약 2%(~60개)
    top_k_violations, _ = torch.topk(violation, k=k, largest=True)
    return torch.mean(top_k_violations)


# ── [v10~v17_01 §2] Collision v5 (기하 자체는 유지, z_gate 곱만 있음) ──────

def _signed_projections(coords_seg, coords_pts):
    """점→세그먼트 부호 있는 법선 투영 (v17 그대로)"""
    A = coords_seg[:-1]
    B = coords_seg[1:]
    AB = B - A

    P = coords_pts.unsqueeze(1)
    A_exp = A.unsqueeze(0)
    AB_exp = AB.unsqueeze(0)

    AB_squared = torch.sum(AB_exp ** 2, dim=-1) + 1e-8
    AP = P - A_exp
    t_proj = torch.sum(AP * AB_exp, dim=-1) / AB_squared
    valid_mask = (t_proj >= 0.0) & (t_proj <= 1.0)

    C = A_exp + t_proj.unsqueeze(-1) * AB_exp
    tangent = AB_exp / (torch.norm(AB_exp, dim=-1, keepdim=True) + 1e-8)
    normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)

    CP = P - C
    projection = torch.sum(CP * normal, dim=-1)
    return projection, valid_mask


def build_collision_spec(coords, t, part_ids, section_ids,
                          clearance_default=0.5, init_buffer=0.05, sign_eps=1e-3,
                          degenerate_clearance=-5.0, fix_mask=None):
    """[v10~v17_01 §2/D2 유지 + v23 §7] 초기 형상에서 전체 unordered 파트쌍에 대해 부호 앵커·
    clearance 산정.

    [v23 §7] fix_mask: (N,) bool — BC 고정 노드 표시. 주어지면 각 direction에
    `ff_mask` (n_pts, n_segs) bool 을 함께 저장한다. True인 항목은 "점도 고정이고 그 점이 투영되는
    세그먼트의 양 끝 노드도 모두 고정"인 경우, 즉 **양쪽 모두 움직일 수 없어 충돌을 해결하는 것이
    기하학적으로 불가능한** 조합이다(command_v1_15.md §7). 이런 항목은 두께 연동 플랜지 오프셋
    레이어가 구조적으로 처리하므로 collision loss에서 제외한다.

    ★ 조건은 반드시 AND다(양쪽 모두 고정). 한쪽만 고정이면 나머지 자유 노드를 움직여 간섭을 실제로
    해결할 수 있으므로 제외하면 안 된다 — design 세션에서 OR로 쓴 초안이 결함으로 판정됐다.
    ★ 인덱스 공간 주의: _signed_projections()는 (섹션, 파트) 부분집합으로 호출되므로 반환 행렬의
    인덱스는 로컬(0..n-1)이다. 따라서 ff_mask도 반드시 이 로컬 공간에서 조립해야 한다(전역 노드
    인덱스를 쓰면 IndexError이거나 엉뚱한 항목을 마스킹한다)."""
    spec = {}
    with torch.no_grad():
        for sec in torch.unique(section_ids):
            sec_int = int(sec.item())
            sec_mask = (section_ids == sec)
            parts = sorted(int(p.item()) for p in torch.unique(part_ids[sec_mask]))
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    a, b = parts[i], parts[j]
                    mask_a = sec_mask & (part_ids == a)
                    mask_b = sec_mask & (part_ids == b)
                    ca, cb = coords[mask_a], coords[mask_b]
                    t_a0 = t[mask_a].mean().item()
                    t_b0 = t[mask_b].mean().item()
                    if fix_mask is not None:
                        fa, fb = fix_mask[mask_a], fix_mask[mask_b]   # 로컬 순서 보존
                    else:
                        fa = fb = None

                    directions = []
                    for (c_seg, c_pt, roles, f_seg, f_pt) in [
                            (ca, cb, (a, b), fa, fb), (cb, ca, (b, a), fb, fa)]:
                        if c_seg.shape[0] < 2 or c_pt.shape[0] == 0:
                            continue
                        proj, valid = _signed_projections(c_seg, c_pt)
                        if valid.sum() == 0:
                            continue
                        vals = proj[valid]
                        m = vals.mean().item()
                        sigma = 1.0 if abs(m) < sign_eps else float(np.sign(m))
                        slack = (sigma * vals).min().item() - (t_a0 + t_b0) / 2.0
                        clearance = min(clearance_default, slack - init_buffer)
                        if clearance < degenerate_clearance:
                            continue
                        entry = {
                            'seg_part': roles[0], 'pt_part': roles[1],
                            'sigma': sigma, 'clearance': clearance,
                        }
                        if f_seg is not None:
                            # 세그먼트 s 는 seg 노드 s, s+1 을 잇는다 -> 양 끝이 모두 고정일 때만 고정 세그먼트
                            seg_fixed = f_seg[:-1] & f_seg[1:]                 # (n_segs,)
                            ff = f_pt.unsqueeze(1) & seg_fixed.unsqueeze(0)    # (n_pts, n_segs) AND
                            entry['ff_mask'] = ff
                        directions.append(entry)
                    if directions:
                        spec[(sec_int, a, b)] = directions
    return spec


# ══════════════════════════════════════════════════════════════════
# [v22] direction-레벨 Top-K 풀링 상수 (command_v1_14.md §1)
# ★ 배치 주의: 아래 두 상수는 compute_collision_loss_v5()의 "기본 인자값"으로 참조된다 —
# 기본 인자값은 함수 본문과 달리 def 시점에 즉시 평가되므로 반드시 함수 정의보다 앞에 있어야
# 한다(v1.10 구현 세션에서 Judge가 확인한 동일 주의사항).
# ══════════════════════════════════════════════════════════════════
COL_TOPK_DIR_FRACTION = 0.25   # 전체 direction(initial_section_v2.csv 기준 실측 289개 —
                                # 153은 (sec,a,b) 키 개수이지 direction 수가 아니다) 중 상위 25%
                                # (=72개)만 평균에 사용.
                                # 시작값 — command_v1_14.md §6 검증 계획에서 실측 후 재보정 필수.
                                # 너무 작으면 학습 초반 다발적 위반을 놓치고, 너무 크면 v21과
                                # 같은 희석으로 되돌아간다(design 세션 Critic 라운드 공통 지적).
COL_MIN_DIRS_TO_KEEP = 5        # Top-K 하한 — 위반 direction이 1~2개뿐일 때 손실이 그 소수에만
                                # 100% 집중돼 형상이 국소적으로 급변하는 것을 막는 안정성 장치.


def compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids, collision_spec, z_gate=None,
                               top_k_fraction=0.3,
                               top_k_dir_fraction=COL_TOPK_DIR_FRACTION,
                               min_dirs_to_keep=COL_MIN_DIRS_TO_KEEP):
    """[v17 유지 + v22 §1] Collision v5 — z_gate[seg]*z_gate[pt] 예외 곱셈 포함.

    [v22 §1] direction별 loss_dir을 곧바로 누적(total_loss += loss_dir)하지 않고 리스트에 모아
    두었다가, 마지막에 상위 K개만 평균한다(기존: 153개 전체 평균 → 위반 소수 direction 희석).
    top_k_fraction(각 direction '내부'의 노드 Top-K)과 top_k_dir_fraction(direction '사이'의
    Top-K)은 서로 다른 계층에 적용되는 별개 파라미터임에 주의 — 전자는 v20부터 존재했고
    후자가 v22의 신규 추가분이다.
    """
    dir_losses = []   # [v22 §1] direction별 손실(0-dim 텐서)을 모아 마지막에 Top-K 평균

    for (sec_int, a, b), directions in collision_spec.items():
        sec_mask = (section_ids == sec_int)
        part_coords = {
            a: new_coords[sec_mask & (part_ids == a)],
            b: new_coords[sec_mask & (part_ids == b)],
        }
        part_t = {
            a: t_final[sec_mask & (part_ids == a)].mean(),
            b: t_final[sec_mask & (part_ids == b)].mean(),
        }

        for d in directions:
            c_seg = part_coords[d['seg_part']]
            c_pt  = part_coords[d['pt_part']]
            if c_seg.shape[0] < 2 or c_pt.shape[0] == 0:
                continue
            proj, valid = _signed_projections(c_seg, c_pt)
            # [v23 §7] fixed-fixed 항목을 valid에서 제외한다 — proj에 큰 값을 심는(masked_fill)
            # 방식은 항목을 valid로 남겨 relu/topk 경로에 들여보내 그래디언트 스파이크를 유발할 수
            # 있어 채택하지 않았다(design 세션 양 Critic 공통 판정).
            ff = d.get('ff_mask')
            if ff is not None:
                valid = valid & (~ff.to(valid.device))
            if valid.sum() == 0:
                continue

            t_sum_half = (part_t[d['seg_part']] + part_t[d['pt_part']]) / 2.0
            gap = d['sigma'] * proj - t_sum_half - d['clearance']
            violation = torch.relu(-gap).clamp(max=1.0) * valid.float()
            # [v20 §4-C] mean → Top-K 풀링(review_v1_8.md §9). 한두 노드의 깊은 국소 관통이
            # 다수의 정상 노드에 의해 평균에서 희석되던 문제를 compute_mesh_order_loss()(v19)와
            # 동일한 패턴으로 해소한다. Huber는 여기서 쓰지 않는다(원본 그대로 제곱 페널티 —
            # Huber 적용은 AI_design_v1_9.compute_collision_penalty_unclamped()의 별도 역할).
            sq_v = (violation ** 2)[valid.bool()]
            if sq_v.numel() > 0:
                k_col = max(1, int(sq_v.numel() * top_k_fraction))
                top_sq_v, _ = torch.topk(sq_v, k=k_col, largest=True)
                loss_dir = top_sq_v.mean()
            else:
                loss_dir = violation.sum() * 0.0

            if z_gate is not None:
                pair_gate = z_gate[d['seg_part']] * z_gate[d['pt_part']]
                loss_dir = loss_dir * pair_gate

            dir_losses.append(loss_dir)   # [v22 §1] 즉시 누적 대신 수집

    # ── [v22 §1] direction-레벨 집계: 단순 평균 → Top-K 평균 ──
    if len(dir_losses) == 0:
        # v21과 동일한 폴백(유효 direction이 하나도 없는 경우) — 그래프 연결이 없는 0 텐서
        return torch.tensor(0.0, device=new_coords.device, requires_grad=True)

    dir_losses_t = torch.stack(dir_losses)          # (n_dirs,)
    if not (0.0 < top_k_dir_fraction < 1.0):
        # ablation/회귀 대조 경로 — v21과 완전히 동일한 전체 평균
        return dir_losses_t.mean()

    k_dir = max(min_dirs_to_keep, int(dir_losses_t.numel() * top_k_dir_fraction))
    k_dir = min(k_dir, dir_losses_t.numel())        # direction이 min_dirs_to_keep보다 적을 때 가드
    top_dir_losses, _ = torch.topk(dir_losses_t, k=k_dir, largest=True)
    return top_dir_losses.mean()


# ══════════════════════════════════════════════════════════════════
# SECTION 2: 커리큘럼 & 게이트 스케줄 & ALDA 상태 & 적응형 TAU_GATE/온도 상수
# ══════════════════════════════════════════════════════════════════

STAGE2_EPOCH = 10
GATE_STEEPNESS = 0.6
GATE_ACTIVE_EPOCH = STAGE2_EPOCH + 1   # [command_v17.md §2] 두께언락과 게이트활성화 시점 분리
INNER_HAT_FLOOR_EPOCHS = 40             # 참고용(현재 Inner Hat은 S_PROTECT라 프루닝 대상 아님)
INNER_HAT_T_MIN_FLOOR = 0.3

PRUNING_COOLDOWN_EPOCHS = 20            # [command_v17.md §4]

W_SPARSE = 0.5                         # [command_v17.md §1] 기존 0.05 → 상향
TAU_GATE = 0.05                         # [command_v17_paratune.md Phase 1] fallback/구버전 대조용으로 유지
                                         # (실제 사용처는 Phase 4에서 pruning_state['current_tau_gate']로 교체)
SPARSE_K = 15.0                         # [command_v17_paratune.md Phase 1] 50.0 → 15.0 완화
ALPHA_ENT = 0.01                        # 엔트로피 항 스케일
ENTROPY_EPS = 1e-7

# ── [command_v17_paratune.md Phase 1] 신규: 적응형 TAU_GATE 상수 ──
TAU_MIN = 0.05             # 기존 TAU_GATE와 동일한 하한(가장 엄격한 목표 달성 시)
TAU_CEILING = 0.10         # 상한(목표가 어려워 오차가 정체돼도 압력이 완전히 죽지 않도록)
TAU_EMA_ALPHA = 0.15       # best-so-far EMA 평활 계수(ewma_alpha와 동일 스케일)

# ── [command_v17_paratune.md Phase 3] 신규: HardConcrete 온도 어닐링 상수 ──
TEMP_INIT = 1.0
TEMP_MIN = 0.3
TEMP_WARMUP_EPOCHS = 30    # GATE_ACTIVE_EPOCH 기준 상대epoch로 이 기간 동안 TEMP_INIT→TEMP_MIN 선형 감쇠


def thickness_gate_value(epoch):
    """[v10~v17_01 §4 유지] gate = sigmoid(0.6×(epoch-128)) — 두께 언락(STAGE2_EPOCH 기준, 불변)"""
    return float(torch.sigmoid(torch.tensor(GATE_STEEPNESS * (epoch - STAGE2_EPOCH))).item())


def compute_gate_temperature(epoch):
    """[command_v17_paratune.md Phase 3] GATE_ACTIVE_EPOCH 기준 상대epoch로 TEMP_INIT→TEMP_MIN
    선형 어닐링. rel_epoch < 0(게이트 비활성 구간)에서는 TEMP_INIT 고정.
    warmup 기간을 max(1, TEMP_WARMUP_EPOCHS)로 가드해 0-division을 방지(Critic 라운드 지적 반영)."""
    rel_epoch = epoch - GATE_ACTIVE_EPOCH
    if rel_epoch < 0:
        return TEMP_INIT
    warmup = max(1, TEMP_WARMUP_EPOCHS)
    progress = min(1.0, rel_epoch / warmup)
    return TEMP_INIT + (TEMP_MIN - TEMP_INIT) * progress


def get_curriculum_weights_v10(epoch, total_epochs, curriculum_ratio):
    """[v10~v17_01 유지] 3-stage curriculum."""
    stage_a_end = int(total_epochs * curriculum_ratio[0])
    stage_b_end = int(total_epochs * curriculum_ratio[1])

    if epoch < stage_a_end:
        progress = 0.0
    elif epoch < stage_b_end:
        x = (epoch - stage_a_end) / max(stage_b_end - stage_a_end, 1)
        progress = 0.5 * (1 + math.sin(math.pi * (x - 0.5)))
    else:
        progress = 1.0

    s_phys   = 0.2 + 0.8 * progress
    s_smooth = progress
    return s_phys, s_smooth


def make_alda_state():
    """[command_v15.md §3 유지] ALDA 초기값/스케줄 상태 dict."""
    return {
        'mu_Mp': 1.0, 'mu_col': 1.0,
        'rho_Mp': 200.0, 'rho_col': 200.0,
        'rho_Mp0': 200.0, 'rho_col0': 200.0,
        'tau_Mp': 0.01,
        'tau_col': 0.02,
        'max_mu': 1e5,
        'rho_min': 200.0, 'rho_max': 2000.0,
        'update_every': 10,
        'slack_threshold': 0.015,
        'rho_freeze_epochs': 30,
    }


def compute_rho_adaptive(model, x, edge_index, edge_attr, target_mp_node,
                          fix_x_mask, fix_y_mask, join_pairs,
                          target_mps, section_ids, part_ids, target_area,
                          collision_spec, thickness_gate, alda_state, gate_active,
                          temperature=0.5):
    """[command_v15.md §2.3 유지] 10epoch마다 objective(f)/제약(g_Mp,g_col) 그래디언트 노름 계산."""
    params = [p for p in model.parameters() if p.requires_grad]

    def _fresh_forward():
        new_coords, _, t_final, _, z_gate, _ = model(
            x, edge_index, edge_attr, target_mp_node,
            fix_x_mask, fix_y_mask, join_pairs, thickness_gate=thickness_gate,
            gate_active=gate_active, temperature=temperature
        )
        area, _ = compute_mass_loss(new_coords, t_final, edge_index, edge_attr, target_area)

        pred_mp_tensors = []
        for section in torch.unique(section_ids):
            section_mask = (section_ids == section)
            src, dst = edge_index
            edge_mask = section_mask[src] & section_mask[dst]
            edge_type = edge_attr[:, 3]
            physical_mask = edge_mask & torch.isclose(edge_type, torch.zeros_like(edge_type))
            edge_index_section = edge_index[:, physical_mask]
            local_index = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
            local_index[section_mask] = torch.arange(section_mask.sum(), device=x.device)
            edge_index_section = local_index[edge_index_section]
            fy_section = x[:, 7:8][section_mask]
            pred_mp_section = calculate_mpl(new_coords[section_mask], t_final[section_mask],
                                             fy_section, edge_index_section)
            pred_mp_tensors.append(pred_mp_section)
        pred_mp_total = torch.stack(pred_mp_tensors).sum()
        target_mp_total = torch.tensor(sum(target_mps.values()), dtype=torch.float32, device=x.device)

        l_collision = compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids,
                                                  collision_spec, z_gate=z_gate)
        return area, pred_mp_total, target_mp_total, l_collision

    with torch.enable_grad():
        area_f, pred_mp_total_f, target_mp_total_f, l_col_f = _fresh_forward()
        f_val = area_f / (target_area + 1e-12)
        grad_f = torch.autograd.grad(f_val, params, retain_graph=False, allow_unused=True)
        norm_f = torch.cat([g.flatten() for g in grad_f if g is not None]).norm().item()

        area_g, pred_mp_total_g, target_mp_total_g, l_col_g = _fresh_forward()
        g_Mp = (pred_mp_total_g - target_mp_total_g).abs() / (target_mp_total_g.abs() + 1e-12) - alda_state['tau_Mp']
        grad_g_Mp = torch.autograd.grad(g_Mp, params, retain_graph=False, allow_unused=True)
        norm_g_Mp = torch.cat([g.flatten() for g in grad_g_Mp if g is not None]).norm().item()

        area_c, pred_mp_total_c, target_mp_total_c, l_col_c = _fresh_forward()
        g_col = l_col_c - alda_state['tau_col']
        grad_g_col = torch.autograd.grad(g_col, params, retain_graph=False, allow_unused=True)
        norm_g_col = torch.cat([g.flatten() for g in grad_g_col if g is not None]).norm().item()

    model.zero_grad(set_to_none=True)

    ratio_Mp = min(1.0, norm_f / (norm_g_Mp + 1e-9))
    ratio_col = min(1.0, norm_f / (norm_g_col + 1e-9))
    return ratio_Mp, ratio_col


# ══════════════════════════════════════════════════════════════════
# SECTION 3: Data Setup (build_bpillar_section — command_v18.md 반영)
# ══════════════════════════════════════════════════════════════════

def _hat_profile(x_ratio, flange_y, crown_y, ramp_lo, ramp_hi, plateau_lo, plateau_hi):
    """[command_v18.md §3.1 신규] flange(평탄) → shoulder(선형 램프) → crown(평탄) → shoulder →
    flange 형태의 대칭 사다리꼴(trapezoid) 프로파일. x_ratio는 이미 호출부의 if/elif 분기 조건을
    만족한 상태에서만 호출되므로, 이 함수 자체는 분기 조건이나 fix_x/fix_y 로직과 무관한 순수 함수다."""
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


def build_bpillar_section():
    """B-Pillar 5-Part 단면 (v8~v17_01과 노드/엣지/BC 토폴로지는 100% 동일).

    [command_v18.md 반영] Part 0(Outer Hat)/Part 2(Inner Hat)의 else 분기 y_coord_node 대입값만
    _hat_profile()로 교체 — 나머지 branch 조건식, fix_x/fix_y 로직, part 1/3/4의 y값, 노드 개수/
    x-spacing, 엣지 연결, join_pairs는 v17_01.py와 동일하다."""
    part_configs = [
        (0, 30.0, 2.30, 1470.0, True),   # #00 Outer Hat
        (1, 28.05, 1.60,  980.0, False), # #03 Inner Plate
        (2, 29.0, 1.60, 1470.0, True),   # #06 Inner Hat
        (3, 24.0, 1.40,  980.0, False),  # #07 Patch 1
        (4, 22.0, 1.60,  440.0, False),  # #08 Patch 2
    ]

    num_nodes   = 30
    total_width = 160.0
    dx          = total_width / (num_nodes - 1)

    nodes = []
    node_registry = {}
    idx = 0
    num_nodes_per_part = {}
    eps = 1e-3

    for part_id, y_base, t_val, fy_val, _ in part_configs:
        local_idx = 0
        for i in range(num_nodes):
            x_coord = i * dx
            x_ratio = x_coord / total_width

            if part_id == 2 and (x_ratio < 0.2 - eps or x_ratio > 0.8 + eps):
                continue
            if part_id == 3 and (x_ratio < 0.3 - eps or x_ratio > 0.7 + eps):
                continue
            if part_id == 4 and (x_ratio < 0.7 - eps or x_ratio > 0.8 + eps):
                continue

            fix = 0.0

            if part_id == 0:
                if x_ratio <= 0.1667 + eps or x_ratio >= 0.8333 - eps:
                    fix = 1.0
                    y_coord_node = 30.0
                else:
                    # [command_v18.md §3.2] flange=30.0 -> crown=65.0 트라페조이드 보간
                    y_coord_node = _hat_profile(
                        x_ratio, flange_y=30.0, crown_y=65.0,
                        ramp_lo=0.1667, ramp_hi=0.8333,
                        plateau_lo=0.3334, plateau_hi=0.6666,
                    )

            elif part_id == 1:
                if x_ratio <= 0.0833 + eps or x_ratio >= 0.9167 - eps:
                    fix = 1.0
                    y_coord_node = 28.05
                elif (x_ratio >= 0.0833 + eps and x_ratio < 0.3334 - eps) or (x_ratio > 0.6666 + eps and x_ratio <= 0.9167 - eps):
                    fix = 1.0
                    y_coord_node = 8.05
                elif 0.3334 - eps <= x_ratio <= 0.6666 + eps:
                    y_coord_node = 15.0
                else:
                    y_coord_node = 8.05

            elif part_id == 2:
                if (x_ratio <= 0.3 - eps and x_ratio > 0.2 + eps) or (x_ratio >= 0.7 + eps and x_ratio < 0.8 - eps):
                    fix = 1.0
                    y_coord_node = 9.65
                else:
                    # [command_v18.md §3.3] flange=9.65 -> crown=45.0 트라페조이드 보간
                    # crown(45.0) < Part 0 crown(65.0) 유지 -> outer/inner hat 비관통 순서 보존
                    y_coord_node = _hat_profile(
                        x_ratio, flange_y=9.65, crown_y=45.0,
                        ramp_lo=0.2, ramp_hi=0.8,
                        plateau_lo=0.4, plateau_hi=0.6,
                    )

            elif part_id == 3:
                if (x_ratio <= 0.33 - eps and x_ratio > 0.3 + eps) or (x_ratio >= 0.67 + eps and x_ratio < 0.7 - eps):
                    fix = 1.0
                    y_coord_node = 9.55
                else:
                    y_coord_node = 16.5

            elif part_id == 4:
                fix = 1.0
                y_coord_node = 7.45

            nodes.append([x_coord, y_coord_node, fix, fix, float(part_id), 0.0, t_val, fy_val])
            node_registry[(part_id, local_idx)] = idx
            local_idx += 1
            idx += 1

        num_nodes_per_part[part_id] = local_idx

    x = torch.tensor(nodes, dtype=torch.float32)

    src_list, dst_list, edge_attr_list = [], [], []

    def add_edge(u, v, part_id):
        dx_val = x[v, 0] - x[u, 0]
        dy_val = x[v, 1] - x[u, 1]
        length = math.sqrt(dx_val**2 + dy_val**2)
        angle  = math.atan2(dy_val, dx_val)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        edge_attr_list.extend([[length, angle, float(part_id), 0.0],
                               [length, -angle, float(part_id), 0.0]])

    for part_id, _, _, _, _ in part_configs:
        for i in range(num_nodes_per_part[part_id] - 1):
            u = node_registry[(part_id, i)]
            v = node_registry[(part_id, i + 1)]
            add_edge(u, v, part_id)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float32)
    join_pairs = torch.zeros((0, 2), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, join_pairs=join_pairs), node_registry


# ══════════════════════════════════════════════════════════════════
# [v21] floor_profiles_template.md 규격 CSV 로더 — 해당 규칙을 따르는 initial section 좌표를
# 17-section Data로 직접 구성한다(CSV 경로는 호출부 인자로 받음).
# ══════════════════════════════════════════════════════════════════

# part_id 순서(CSV 블록 순서와 반드시 일치): 0=Outer Hat, 1=Inner Plate, 2=Inner Hat,
# 3=Patch1, 4=Patch2, 5=Patch3, 6=Patch4 — 0~4는 build_bpillar_section()의 part_configs와 동일한 재료 상수.
# [v26] Patch3 = Patch2의 좌우 대칭 복제(같은 역할의 foot 보강) -> Patch2와 동일 440MPa.
#       Patch4 = Outer crown 안쪽 doubler -> Patch1과 같은 패치 등급 980MPa (임시값 — 실제 재질 확정 시 교체).
FY_BY_PART = {0: 1470.0, 1: 980.0, 2: 1470.0, 3: 980.0, 4: 440.0, 5: 440.0, 6: 980.0}

# 파트별 floor당 예상 점 개수(고정) — CSV 구조 검증용.
EXPECTED_NODE_COUNT_BY_PART = {0: 30, 1: 30, 2: 18, 3: 12, 4: 3, 5: 3, 6: 14}

# [v26] initial_section_v5/v6.csv floor 수(상단 2개 섹션 삭제: 17 -> 15)
NUM_SECTIONS_CSV = 15


# [v1.16 §2] 원칙 3 — Plate(#03) 전이 구간 노드 고정 해제. FLANGE_PAIR_SPEC(AI_design_v1_16.py)에
# 등록된 실제 플랜지 노드(7,8,9,10,21,22,23,24)는 여전히 고정 유지, 그 바로 옆 전이 구간만 해제한다.
# point_idx는 1-based(이 테이블의 키와 동일 기준).
PLATE_UNFIX_POINT_IDX = {4, 5, 6, 25, 26, 27}


def _build_fix_mask_table(enable_plate_unfix=True):
    """(part_id, point_idx 1-based) -> fix(0.0/1.0) 정적 룩업 테이블.
    build_bpillar_section()의 x_ratio 조건(플랜지=1, 램프/크라운=0)을 좌표와 무관하게
    좌표 생성 시점의 grid position만으로 그대로 재현한다(/synod design 세션 결론:
    좌표 역산이 아니라 point_idx 기반 정적 테이블이 국소 y-shift에 영향받지 않아 안전).

    [v1.16 §5] enable_plate_unfix=False면 PLATE_UNFIX_POINT_IDX 예외를 적용하지 않아 v1.15.5와
    동일한(전부 고정) 테이블을 반환한다 — --enable-plate-unfix CLI 플래그의 A/B 검증용."""
    eps = 1e-3
    num_nodes = 30
    total_width = 160.0
    dx = total_width / (num_nodes - 1)
    table = {}
    for part_id in range(5):
        point_idx = 0
        for i in range(num_nodes):
            x_ratio = (i * dx) / total_width
            if part_id == 2 and (x_ratio < 0.2 - eps or x_ratio > 0.8 + eps):
                continue
            if part_id == 3 and (x_ratio < 0.3 - eps or x_ratio > 0.7 + eps):
                continue
            if part_id == 4 and (x_ratio < 0.7 - eps or x_ratio > 0.8 + eps):
                continue

            fix = 0.0
            if part_id == 0:
                if x_ratio <= 0.1667 + eps or x_ratio >= 0.8333 - eps:
                    fix = 1.0
            elif part_id == 1:
                if x_ratio <= 0.0833 + eps or x_ratio >= 0.9167 - eps:
                    fix = 1.0
                elif (x_ratio >= 0.0833 + eps and x_ratio < 0.3334 - eps) or \
                     (x_ratio > 0.6666 + eps and x_ratio <= 0.9167 - eps):
                    fix = 1.0
                # [v1.16 §2] 원칙 3 — 전이 구간 6개 노드만 예외적으로 해제.
                # point_idx는 아래서 +1 되므로 이번 순회에서 확정될 point_idx는 point_idx+1.
                if enable_plate_unfix and (point_idx + 1) in PLATE_UNFIX_POINT_IDX:
                    fix = 0.0
            elif part_id == 2:
                if (x_ratio <= 0.3 - eps and x_ratio > 0.2 + eps) or \
                   (x_ratio >= 0.7 + eps and x_ratio < 0.8 - eps):
                    fix = 1.0
            elif part_id == 3:
                if (x_ratio <= 0.33 - eps and x_ratio > 0.3 + eps) or \
                   (x_ratio >= 0.67 + eps and x_ratio < 0.7 - eps):
                    fix = 1.0
            elif part_id == 4:
                fix = 1.0

            point_idx += 1
            table[(part_id, point_idx)] = fix

    # [v26] part 5, 6은 30-grid x_ratio 규칙으로 표현되지 않으므로 point_idx로 직접 등록한다.
    #   part 5(Patch3, 3점): Patch2(part 4)와 동일 규칙 — 전 노드 고정. Plate 바닥 외측 덧판이라
    #     접합 상대인 Plate pt 7/8/9(고정 플랜지 밴드)와 함께 움직이지 않는다.
    #   part 6(Patch4, 14점): 전 노드 자유 — 접합 상대인 Outer crown/벽(point 9~22)이 자유 노드라
    #     Patch4를 고정하면 Outer crown이 안쪽으로 이동할 때 collision으로 Outer를 붙잡는 제약이 된다.
    #     Outer와의 관계는 collision loss(부호 앵커 + clearance)가 유지한다(Patch1 중앙부와 같은 방식).
    for k in range(1, EXPECTED_NODE_COUNT_BY_PART[5] + 1):
        table[(5, k)] = 1.0
    for k in range(1, EXPECTED_NODE_COUNT_BY_PART[6] + 1):
        table[(6, k)] = 0.0
    return table


_FIX_MASK_TABLE = _build_fix_mask_table()


def _parse_floor_profiles_csv(csv_path, num_sections=NUM_SECTIONS_CSV):
    """floor_profiles_template.md 규칙(데이터 행 먼저, 파트 좌표 종료 직후 구분 행으로 두께 확정)을
    따르는 CSV를 파싱한다. 반환: [{'part_id':int, 't':float, 'floors': {floor_idx:int -> [(x,y),...]}}]
    (파트는 CSV에 나온 블록 순서 그대로 0,1,2,3,4에 대응된다고 가정)."""
    blocks = []
    cur_floors = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["floor_idx", "point_idx", "x_mm", "y_mm", "r_mm", "t_mm"], \
            f"[uni_section_v21] CSV 헤더가 floor_profiles_template.md 규격과 다릅니다: {header}"
        for row in reader:
            floor_idx_s, point_idx_s, x_s, y_s, r_s, t_s = row
            if point_idx_s == "0.0":
                part_id = len(blocks)
                blocks.append({"part_id": part_id, "t": float(t_s), "floors": cur_floors})
                cur_floors = {}
            else:
                fi = int(float(floor_idx_s))
                cur_floors.setdefault(fi, []).append((float(x_s), float(y_s)))

    n_parts = len(EXPECTED_NODE_COUNT_BY_PART)
    assert len(blocks) == n_parts, \
        f"[uni_section_v26] CSV 파트 블록 수가 {n_parts}가 아닙니다(got {len(blocks)}) — " \
        f"Outer/Plate/InnerHat/Patch1/Patch2/Patch3/Patch4 구조(initial_section_v5/v6.csv)를 확인하세요."
    for b in blocks:
        pid = b["part_id"]
        n_floors = len(b["floors"])
        assert n_floors == num_sections, \
            f"[uni_section_v26] part_id={pid} 의 floor 개수가 {num_sections}가 아닙니다(got {n_floors})."
        counts = {len(pts) for pts in b["floors"].values()}
        assert counts == {EXPECTED_NODE_COUNT_BY_PART[pid]}, \
            f"[uni_section_v21] part_id={pid} 의 floor당 점 개수가 예상({EXPECTED_NODE_COUNT_BY_PART[pid]})과 " \
            f"다릅니다(got {counts}) — CSV point_idx 순서가 원본 grid와 어긋났을 가능성이 있습니다."
    return blocks


def build_bpillar_from_csv(csv_path, num_sections=NUM_SECTIONS_CSV, enable_plate_unfix=True):
    """floor_profiles_template.md 규격 CSV(예: initial_section/initial_section.csv)를 로드해
    17-section 전체를 담은 단일
    Data(x[N,8], edge_index, edge_attr, join_pairs)를 직접 반환한다 — 기존
    build_bpillar_17section()(AI_design)이 하던 "단일 섹션 17회 생성 + scale_factor 곱셈" 대신,
    CSV가 이미 floor별 최종 실좌표(간섭 제거 보정 포함)를 담고 있으므로 그대로 읽어 조립만 한다.
    x 컬럼 의미는 build_bpillar_section()과 완전히 동일:
      [0]=x_mm, [1]=y_mm, [2]=fix_x, [3]=fix_y, [4]=part_id, [5]=section_id(=floor_idx),
      [6]=t_mm, [7]=fy.

    [v1.16 §5] enable_plate_unfix=True(기본값)면 모듈 로드 시 만들어진 _FIX_MASK_TABLE(이미
    PLATE_UNFIX_POINT_IDX 반영)을 그대로 쓰고, False면 v1.15.5와 동일한(전부 고정) 테이블을
    그때그때 새로 만들어 쓴다(모듈 상수는 건드리지 않음 — 학습 스크립트가 플래그를 바꿔도 안전)."""
    # 모듈 로드 시 만들어진 _FIX_MASK_TABLE은 이미 enable_plate_unfix=True(신규 기본값) 기준이다.
    fix_table = _FIX_MASK_TABLE if enable_plate_unfix else _build_fix_mask_table(enable_plate_unfix=False)
    blocks = _parse_floor_profiles_csv(csv_path, num_sections=num_sections)

    all_x_rows = []
    src_list, dst_list, edge_attr_list = [], [], []
    node_offset = 0

    for floor in range(num_sections):
        for b in blocks:
            part_id = b["part_id"]
            t_val = b["t"]
            fy_val = FY_BY_PART[part_id]
            pts = b["floors"][floor]

            local_idx_start = node_offset
            for point_idx, (x_mm, y_mm) in enumerate(pts, start=1):
                fix = fix_table[(part_id, point_idx)]
                all_x_rows.append([x_mm, y_mm, fix, fix, float(part_id), float(floor), t_val, fy_val])

            for i in range(len(pts) - 1):
                u = local_idx_start + i
                v = local_idx_start + i + 1
                dx_val = all_x_rows[v][0] - all_x_rows[u][0]
                dy_val = all_x_rows[v][1] - all_x_rows[u][1]
                length = math.sqrt(dx_val ** 2 + dy_val ** 2)
                angle = math.atan2(dy_val, dx_val)
                src_list.extend([u, v])
                dst_list.extend([v, u])
                edge_attr_list.extend([[length, angle, float(part_id), 0.0],
                                       [length, -angle, float(part_id), 0.0]])

            node_offset += len(pts)

    x = torch.tensor(all_x_rows, dtype=torch.float32)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    join_pairs = torch.zeros((0, 2), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, join_pairs=join_pairs)


def compute_y_pna_ref(coords, t, fy, edge_index, n_iter=50):
    with torch.no_grad():
        _, y_pna = compute_edge_mp_pna(coords, t, fy, edge_index, n_iter)
        return y_pna.item() if torch.is_tensor(y_pna) else y_pna


def compute_section_area(coords, t, edge_index, part_ids=None):
    """단면 면적 계산: A = Σ(L × t_e) [mm²] (v17 그대로)"""
    with torch.no_grad():
        mask = edge_index[0] < edge_index[1]
        u, v = edge_index[0][mask], edge_index[1][mask]
        x_u, y_u = coords[u, 0], coords[u, 1]
        x_v, y_v = coords[v, 0], coords[v, 1]
        L   = torch.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2)
        t_e = t[u].squeeze(-1)
        area_e = L * t_e

        total_area = area_e.sum().item()

        per_part = {}
        if part_ids is not None:
            pid_e = part_ids[u]
            for pid in torch.unique(pid_e):
                per_part[int(pid.item())] = area_e[pid_e == pid].sum().item()

    return total_area, per_part


# ══════════════════════════════════════════════════════════════════
# SECTION 4: Training Step
# ══════════════════════════════════════════════════════════════════

def train_step(model, data, optimizer, target_mps, target_area,
               epoch, max_epochs, weights, curriculum,
               curriculum_ratio, collision_spec, alda_state,
               pruning_state=None, diagnostic=False, diagnostic_part_id=4):
    """
    v18은 build_bpillar_section() 이외 로직을 v17_01.py와 동일하게 유지한다(command_v18.md §0:
    cosmetic-only 변경). train_step 자체의 시그니처/동작은 v17_01과 완전히 동일.
    """
    model.train()
    optimizer.zero_grad()

    x          = data.x
    edge_index = data.edge_index
    edge_attr  = data.edge_attr
    join_pairs = data.join_pairs if hasattr(data, 'join_pairs') else None
    base_coords = x[:, :2].detach()

    fix_x_mask  = x[:, 2].bool().unsqueeze(1)
    fix_y_mask  = x[:, 3].bool().unsqueeze(1)
    part_ids    = x[:, 4]
    section_ids = x[:, 5]
    fy          = x[:, 7].unsqueeze(1)

    unique_sections = torch.unique(section_ids)

    target_mp_node = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
    for section in unique_sections:
        section_mask = (section_ids == section)
        section_int = int(section.item())
        target_mp_node[section_mask] = target_mps[section_int]

    gate = thickness_gate_value(epoch)                # 두께 언락(STAGE2_EPOCH 기준, 불변)
    gate_active = epoch >= GATE_ACTIVE_EPOCH           # [command_v17.md §2] 분리된 게이트 활성화 시점

    # [command_v17_paratune.md Phase 3] HardConcrete temperature — 상대epoch 기준 선형 어닐링
    current_temp = compute_gate_temperature(epoch)

    # ── 진단 모드: GATE_ACTIVE_EPOCH 시점에 지정 파트의 log_alpha를 강제로 낮춤 ──
    # [command_v17_paratune.md Phase 2, /synod design 세션 인덱싱 확정]
    # log_alpha는 CANDIDATE_PARTS 위치가 아니라 part_id로 직접 인덱싱되는 길이-num_parts 텐서다
    # (S_PROTECT=[0,1,2]도 동일한 direct part-id 인덱싱 관례를 따름). 따라서
    # CANDIDATE_PARTS.index()로 변환하지 않고 diagnostic_part_id를 그대로 인덱스로 사용한다.
    if diagnostic and epoch == GATE_ACTIVE_EPOCH:
        with torch.no_grad():
            model.log_alpha.data[diagnostic_part_id] = -5.0
        print(f"[DIAGNOSTIC] epoch {epoch}: part {diagnostic_part_id} log_alpha 강제 -5.0 "
              f"(ewma_z 하강 궤적 관찰 시작)")

    new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open = model(
        x, edge_index, edge_attr, target_mp_node,
        fix_x_mask, fix_y_mask, join_pairs, thickness_gate=gate,
        gate_active=gate_active, temperature=current_temp
    )

    ## ── 물리 손실: 비대칭 Huber (w_phys 항) — coords·t 모두에 grad ──
    l_phys_terms = []
    pred_mp_tensors = []
    pred_mp_sections = []

    for section in unique_sections:
        section_mask = (section_ids == section)
        coords_section = new_coords[section_mask]
        t_section = t_final[section_mask]
        fy_section = fy[section_mask]

        src, dst = edge_index
        edge_mask = section_mask[src] & section_mask[dst]

        edge_type = edge_attr[:, 3]
        physical_mask = edge_mask & torch.isclose(edge_type, torch.zeros_like(edge_type))

        edge_index_section = edge_index[:, physical_mask]

        local_index = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        local_index[section_mask] = torch.arange(section_mask.sum(), device=x.device)
        edge_index_section = local_index[edge_index_section]

        pred_mp_section = calculate_mpl(coords_section, t_section, fy_section, edge_index_section)

        section_int = int(section.item())
        target_mp_section = torch.tensor(target_mps[section_int], dtype=torch.float32, device=x.device)

        l_phys_terms.append(asymmetric_huber_phys(pred_mp_section, target_mp_section))
        pred_mp_tensors.append(pred_mp_section)
        pred_mp_sections.append(pred_mp_section.item())

    l_phys_total = torch.stack(l_phys_terms).mean()
    pred_mp_total = torch.stack(pred_mp_tensors).sum()
    target_mp_total = torch.tensor(sum(target_mps.values()), dtype=torch.float32, device=x.device)

    pred_mp_sections = np.array(pred_mp_sections)
    mp_rel_err = float(np.abs(np.sum(pred_mp_sections) - sum(target_mps.values())) / sum(target_mps.values()))

    ## ── [command_v17_paratune.md Phase 4] 적응형 TAU_GATE 갱신 — mp_rel_err 계산 직후,
    ##    gate_multiplier 계산 이전. pruning_state가 없으면(하위호환) 정적 TAU_GATE로 폴백.
    if pruning_state is not None:
        if pruning_state['cooldown_remaining'] == 0:   # 하드 프리즈: 쿨다운 중엔 갱신 자체를 skip
            pruning_state['best_so_far_mp_rel_err'] = min(
                pruning_state['best_so_far_mp_rel_err'], mp_rel_err)
            pruning_state['ema_best_so_far'] = (
                TAU_EMA_ALPHA * pruning_state['best_so_far_mp_rel_err']
                + (1.0 - TAU_EMA_ALPHA) * pruning_state['ema_best_so_far']
            )
            pruning_state['current_tau_gate'] = min(
                TAU_CEILING, max(TAU_MIN, pruning_state['ema_best_so_far']))
        # 쿨다운 중에는 current_tau_gate를 이전 값 그대로 유지(재계산하지 않음)
        current_tau_gate = pruning_state['current_tau_gate']
    else:
        current_tau_gate = TAU_GATE   # 하위호환 폴백(구버전 정적 임계값)

    ## ── [command_v17.md §3 유지, Phase 4로 TAU_GATE만 교체] 희소화 게이팅 + 엔트로피 ──
    gate_multiplier = float(torch.sigmoid(torch.tensor(SPARSE_K * (current_tau_gate - mp_rel_err))).item())
    w_sparse_effective = weights['w_sparse'] * gate_multiplier

    l_sparse = z_open[CANDIDATE_PARTS].mean()
    contrib_sparse = w_sparse_effective * l_sparse

    contrib_entropy = torch.tensor(0.0, device=x.device)
    if gate_active:
        z_cand = z_open[CANDIDATE_PARTS]
        entropy = -torch.mean(z_cand * torch.log(z_cand + ENTROPY_EPS)
                               + (1.0 - z_cand) * torch.log(1.0 - z_cand + ENTROPY_EPS))
        contrib_entropy = (ALPHA_ENT * gate_multiplier) * entropy

    ## ── 커리큘럼 가중치 ──
    s_phys, s_smooth = 1.0, 1.0
    if curriculum:
        s_phys, s_smooth = get_curriculum_weights_v10(epoch, max_epochs, curriculum_ratio)

    ## ── 다목적 손실 계산 ──
    l_smooth       = compute_smoothness_loss_angle(new_coords, edge_index, edge_attr)
    area, l_mass   = compute_mass_loss(new_coords, t_final, edge_index, edge_attr, target_area)
    l_collision    = compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids,
                                                collision_spec, z_gate=z_gate)
    l_order        = compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr)
    l_anchor       = compute_anchor_loss(new_coords, base_coords, fix_x_mask, fix_y_mask)
    l_sat          = compute_saturation_loss(delta_t_part, delta_scale=model.DELTA_SCALE)

    ## ── ALDA: area objective + Mp/collision 하드 제약 ──
    L_alda, g_Mp_val, g_col_val = compute_alda_loss(
        area, target_area, pred_mp_total, target_mp_total, l_collision, alda_state)
    L_alda_effective = L_alda * max(gate, 0.05)

    ## ── 가중치 적용 후 항별 기여도 ──
    contrib_phys      = weights['w_phys']      * l_phys_total * s_phys
    contrib_smooth    = weights['w_smooth']    * l_smooth     * s_smooth
    contrib_order     = weights['w_order']     * l_order
    contrib_anchor    = weights['w_anchor']    * l_anchor
    contrib_sat       = weights['w_sat']       * l_sat

    loss = (contrib_phys + contrib_smooth + L_alda_effective
            + contrib_order + contrib_anchor + contrib_sat
            + contrib_sparse + contrib_entropy)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    with torch.no_grad():
        dt_sat_threshold = 0.9 * model.DELTA_SCALE
        saturated_mask = delta_t_part.abs().squeeze(-1) >= dt_sat_threshold
        saturated_parts = 0
        t_per_part = {}
        for pid in torch.unique(part_ids):
            pmask = (part_ids == pid)
            if saturated_mask[pmask].float().mean().item() > 0.5:
                saturated_parts += 1
            t_per_part[int(pid.item())] = t_final[pmask].mean().item()

        z_gate_np = z_gate.detach().cpu().numpy()
        z_open_np = z_open.detach().cpu().numpy()

    return {
        "loss":          loss.item(),
        "pred_mp":       pred_mp_sections,
        "mp_rel_err":    mp_rel_err,
        "l_phys":        l_phys_total.item(),
        "l_smooth":      l_smooth.item(),
        "area":          area.item(),
        "l_mass":        l_mass.item(),
        "l_collision":   l_collision.item(),
        "l_order":       l_order.item(),
        "l_anchor":      l_anchor.item(),
        "l_sat":         l_sat.item(),
        "l_sparse":      l_sparse.item(),
        "gate_multiplier": gate_multiplier,
        "new_coords":    new_coords.detach(),
        "thickness_gate": gate,
        "gate_active":   gate_active,
        "delta_t_mean":  delta_t_part.mean().item(),
        "saturated_parts": saturated_parts,
        "t_per_part":    t_per_part,
        "alda_g_Mp":     g_Mp_val,
        "alda_g_col":    g_col_val,
        "L_alda":        L_alda.item(),
        "z_gate":        z_gate_np,
        "z_open":        z_open_np,
        # ── [command_v17_paratune.md Phase 4] 적응형 상태 echo(로깅용, in-place mutation과 별개로
        #    run_training의 기존 "info dict = 유일한 출처" 계약을 유지하기 위해 값을 복사해 담는다) ──
        "temperature":       current_temp,
        "current_tau_gate":  current_tau_gate,
        "cooldown_remaining": (pruning_state['cooldown_remaining'] if pruning_state is not None else 0),
    }


def run_training(data, target_mps, target_area,
                  max_epochs=300, lr=1e-3, weights=None, curriculum=True,
                  curriculum_ratio=(0.2, 0.7), snapshot_interval=10,
                  feasibility_mp_err=0.02, feasibility_collision=0.05,
                  diagnostic=False, diagnostic_part_id=4):
    """
    v18의 run_training은 v17_01.py와 완전히 동일하다(command_v18.md §0: build_bpillar_section()
    이외 변경 없음). 초기 형상만 더 현실적인 hat 프로파일로 바뀌었을 뿐, 학습 절차/체크포인트/
    프루닝/ALDA/온도 어닐링 로직은 전부 동일하게 유지된다.
    """
    if diagnostic and diagnostic_part_id not in CANDIDATE_PARTS:
        raise ValueError(
            f"[diagnostic] diagnostic_part_id={diagnostic_part_id}는 CANDIDATE_PARTS={CANDIDATE_PARTS}에 "
            f"없습니다 — S_PROTECT 파트는 애초에 프루닝 대상이 아니므로 진단 대상이 될 수 없습니다."
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data   = data.to(device)

    model = CGDN(
        in_channels=8,
        hidden_channels=128,
        num_layers=4,
        heads=4,
        edge_dim=4,
        max_displacement=50.0,
        num_parts=5,
        init_log_alpha=2.0,   # [command_v17_paratune.md Phase 0] 0.0 오버라이드 제거, 생성자 기본값과 일치
    ).to(device)

    thick_param_ids = {id(p) for p in model.thickness_decoder.parameters()}
    gate_param_ids  = {id(model.log_alpha)}
    main_params  = [p for p in model.parameters()
                    if id(p) not in thick_param_ids and id(p) not in gate_param_ids]
    thick_params = list(model.thickness_decoder.parameters())
    gate_params  = [model.log_alpha]
    assert len(main_params) + len(thick_params) + len(gate_params) == len(list(model.parameters()))
    weight_decay_val = 1e-4
    optimizer = optim.AdamW(
        [{'params': main_params,  'name': 'main'},
         {'params': thick_params, 'name': 'thickness_decoder'},
         {'params': gate_params,  'name': 'gate_params', 'lr': 0.0}],
        lr=lr, weight_decay=weight_decay_val)

    if weights is None:
        weights = {
            'w_phys':      10.0,
            'w_collision':  5.0,   # ALDA로 대체되어 loss에 미사용(호환성 유지)
            'w_order':      1.0,
            'w_mass':       2.0,   # ALDA로 대체되어 loss에 미사용(호환성 유지)
            'w_smooth':     0.5,
            'w_anchor':     0.02,
            'w_sat':        0.01,
            'w_sparse':     W_SPARSE,
        }

    alda_state = make_alda_state()
    pruning_state = make_pruning_state(num_parts=5, candidate_parts=CANDIDATE_PARTS)

    x = data.x
    part_labels_t = x[:, 4].cpu().long()
    edge_index_cpu = data.edge_index.cpu()
    base_coords = x[:, :2].detach().cpu()
    base_t_cpu  = x[:, 6:7].cpu()
    fy_full     = x[:, 7:8].cpu()
    part_ids    = x[:, 4]
    section_ids = x[:, 5]

    verify_thickness_gradient(x[:, :2], x[:, 6:7], x[:, 7:8], data.edge_index)

    if target_area is None:
        target_area, _ = compute_section_area(x[:, :2].cpu(), base_t_cpu, edge_index_cpu)
    print(f"[mass] target_area = {target_area:.1f} mm² (초기 단면적 스냅샷)")

    collision_spec = build_collision_spec(x[:, :2], x[:, 6:7], part_ids, section_ids)
    print(f"[collision v5] 전쌍({len(collision_spec)}쌍) 부호 앵커 & clearance (초기 형상 기준):")
    for (sec, a, b), dirs in collision_spec.items():
        info = " | ".join(f"{d['seg_part']}seg/{d['pt_part']}pt σ={d['sigma']:+.0f} clr={d['clearance']:.2f}"
                          for d in dirs)
        print(f"    sec{sec} Part{a}-Part{b}: {info}")

    history = {
        'loss': [], 'pred_mp': [], 'mp_rel_err': [], 'l_phys': [], 'l_smooth': [],
        'area': [], 'l_mass': [], 'l_collision': [], 'l_order': [],
        'l_anchor': [], 'l_sat': [], 'l_sparse': [], 'gate_multiplier': [],
        'thickness_gate': [], 'delta_t_mean': [],
        'saturated_parts': [], 'snapshots': [], 't_per_part': [],
        'alda_g_Mp': [], 'alda_g_col': [], 'alda_mu_Mp': [], 'alda_mu_col': [],
        'alda_rho_Mp': [], 'alda_rho_col': [], 'L_alda': [],
        'z_gate': [], 'z_open': [],
        # ── [command_v17_paratune.md Phase 4] 신규 history 키 — 향후 시각화 확장 대비 키만 확보
        #    (이번 범위에서 visualize_training에 새 패널을 추가하지는 않음) ──
        'temperature': [], 'current_tau_gate': [],
    }

    best_feasible = {
        'found': False, 'epoch': None, 'mp_rel_err': None,
        'l_collision': None, 'state_dict': None, 'pruning_state': None,
    }
    best_mp = {
        'epoch': None, 'mp_rel_err': float('inf'),
        'l_collision': None, 'state_dict': None, 'pruning_state': None,
    }
    first_feasible_epoch = None
    stage2_done = False
    gate_stage_done = False   # [command_v17.md §2] GATE_ACTIVE_EPOCH 전환 플래그(STAGE2_EPOCH와 분리)

    print(f"\n{'=' * 78}")
    print(f"[ uni_section_v18 ] Training  |  Target Mp = {target_mps[0]:,.0f} N·mm  |  Epochs: {max_epochs}")
    print(f"  CGDN: hidden=128, layers=4, heads=4  |  Curriculum: {curriculum} {curriculum_ratio}")
    print(f"  (command_v18.md: build_bpillar_section() Part0/Part2 초기 좌표를 hat 프로파일로 갱신 — "
          f"그 외 학습 로직은 v17_01과 동일)")
    print(f"  T_MAX={CGDN.T_MAX}mm | DELTA_SCALE={CGDN.DELTA_SCALE} | w_sparse={weights['w_sparse']} | grad_clip=10.0")
    print(f"  S_protect={S_PROTECT} | Candidate={CANDIDATE_PARTS} | init_log_alpha=2.0")
    print(f"  Stage2(두께언락)@{STAGE2_EPOCH} | GATE_ACTIVE_EPOCH(게이트/프루닝 활성)@{GATE_ACTIVE_EPOCH}")
    print(f"  Pruning: ewma_alpha={pruning_state['ewma_alpha']} thresh={pruning_state['thresh_low']}/"
          f"{pruning_state['thresh_high']} confirm={pruning_state['confirm_epochs']}epoch "
          f"cooldown={PRUNING_COOLDOWN_EPOCHS}epoch")
    print(f"  Adaptive TAU_GATE: TAU_MIN={TAU_MIN} TAU_CEILING={TAU_CEILING} ema_alpha={TAU_EMA_ALPHA} "
          f"| SPARSE_K={SPARSE_K} | ALPHA_ENT={ALPHA_ENT}")
    print(f"  Temperature annealing: {TEMP_INIT}→{TEMP_MIN} over {TEMP_WARMUP_EPOCHS}epoch "
          f"(GATE_ACTIVE_EPOCH 기준 상대epoch)")
    if diagnostic:
        print(f"  [DIAGNOSTIC MODE] part {diagnostic_part_id}의 log_alpha를 GATE_ACTIVE_EPOCH에서 "
              f"강제로 -5.0 낮춰 ewma_z 하강 궤적을 관찰합니다. thresh_low/high는 변경하지 않습니다.")
    print(f"  Feasibility 기준: Mp err < {feasibility_mp_err*100:.1f}%  AND  l_collision < {feasibility_collision}")
    print(f"{'=' * 78}")
    print(f"Epoch ||  Loss  ||  MpErr% |  Smth  |  Area  |  Coll  | Sparse || tGate | gAct | mu_Mp | z_gate(3,4) | Temp | TauGate")

    new_coords = None
    for epoch in range(max_epochs):
        if (not stage2_done) and epoch == STAGE2_EPOCH:
            for group in optimizer.param_groups:
                if group.get('name') == 'thickness_decoder':
                    for p in group['params']:
                        optimizer.state.pop(p, None)
                    group['lr'] = 0.3 * lr
            stage2_done = True
            print(f"[Stage 2] epoch {epoch}: thickness_decoder 그룹 AdamW state 리셋(lr→{0.3*lr:.1e}) "
                  f"(좌표 헤드 모멘텀 보존)")

        # [command_v17.md §2] gate_params lr 활성화는 GATE_ACTIVE_EPOCH에서 별도로 수행
        if (not gate_stage_done) and epoch == GATE_ACTIVE_EPOCH:
            for group in optimizer.param_groups:
                if group.get('name') == 'gate_params':
                    group['lr'] = 1e-2
            gate_stage_done = True
            print(f"[Gate Active] epoch {epoch}: gate_params lr → 1e-2 (게이트/프루닝 트리거 활성화, "
                  f"두께 언락과 {GATE_ACTIVE_EPOCH - STAGE2_EPOCH}epoch 분리)")

        info = train_step(model, data, optimizer, target_mps, target_area,
                           epoch, max_epochs, weights, curriculum,
                           curriculum_ratio, collision_spec, alda_state,
                           pruning_state=pruning_state,
                           diagnostic=diagnostic, diagnostic_part_id=diagnostic_part_id)

        for key in ('loss', 'pred_mp', 'mp_rel_err', 'l_phys', 'l_smooth', 'area', 'l_mass',
                    'l_collision', 'l_order', 'l_anchor', 'l_sat', 'l_sparse', 'gate_multiplier',
                    'thickness_gate', 'delta_t_mean', 'saturated_parts',
                    'alda_g_Mp', 'alda_g_col', 'L_alda', 'z_gate', 'z_open',
                    'temperature', 'current_tau_gate'):
            history[key].append(info[key])
        history['t_per_part'].append(info['t_per_part'])
        new_coords = info['new_coords']

        # [command_v15.md §2.2] 10epoch마다 ALDA 승수 갱신
        if epoch > 0 and epoch % alda_state['update_every'] == 0:
            g_Mp_val = info['alda_g_Mp']
            g_col_val = info['alda_g_col']

            alda_state['mu_Mp'] = min(alda_state['max_mu'],
                                       max(0.0, alda_state['mu_Mp'] + alda_state['rho_Mp'] * g_Mp_val))
            alda_state['mu_col'] = min(alda_state['max_mu'],
                                        max(0.0, alda_state['mu_col'] + alda_state['rho_col'] * g_col_val))

            if epoch >= alda_state['rho_freeze_epochs']:
                x_full = data.x
                fix_x_mask_full = x_full[:, 2].bool().unsqueeze(1)
                fix_y_mask_full = x_full[:, 3].bool().unsqueeze(1)
                section_ids_full = x_full[:, 5]
                target_mp_node_full = torch.zeros((x_full.shape[0], 1), dtype=torch.float32, device=x_full.device)
                for section in torch.unique(section_ids_full):
                    section_mask = (section_ids_full == section)
                    target_mp_node_full[section_mask] = target_mps[int(section.item())]
                gate_now = thickness_gate_value(epoch)
                gate_active_now = epoch >= GATE_ACTIVE_EPOCH
                temp_now = compute_gate_temperature(epoch)

                if g_Mp_val > alda_state['slack_threshold'] or g_col_val > alda_state['slack_threshold']:
                    ratio_Mp, ratio_col = compute_rho_adaptive(
                        model, x_full, data.edge_index, data.edge_attr, target_mp_node_full,
                        fix_x_mask_full, fix_y_mask_full, data.join_pairs,
                        target_mps, section_ids_full, part_ids, target_area,
                        collision_spec, gate_now, alda_state, gate_active_now,
                        temperature=temp_now)

                    if g_Mp_val > alda_state['slack_threshold']:
                        alda_state['rho_Mp'] = min(alda_state['rho_max'],
                                                    max(alda_state['rho_min'],
                                                        alda_state['rho_Mp0'] * ratio_Mp))
                    if g_col_val > alda_state['slack_threshold']:
                        alda_state['rho_col'] = min(alda_state['rho_max'],
                                                     max(alda_state['rho_min'],
                                                         alda_state['rho_col0'] * ratio_col))

        history['alda_mu_Mp'].append(alda_state['mu_Mp'])
        history['alda_mu_col'].append(alda_state['mu_col'])
        history['alda_rho_Mp'].append(alda_state['rho_Mp'])
        history['alda_rho_col'].append(alda_state['rho_col'])

        # [command_v17.md §4] 프루닝 트리거 + State-Only Reset(쿨다운 포함)
        if info['gate_active']:
            if pruning_state['cooldown_remaining'] > 0:
                pruning_state['cooldown_remaining'] -= 1
                newly_deleted = []
            else:
                newly_deleted = update_pruning_state(pruning_state, info['z_gate'])

            for pid in newly_deleted:
                model.log_alpha.data[pid] = -10.0
                print(f"[Pruning] epoch {epoch}: part {pid} DELETED — triggering State-Only Reset")

                # (a) 옵티마이저 재인스턴스화 — 현재 lr을 동적으로 읽어와 보존(하드코딩 금지)
                current_lrs = {g['name']: g['lr'] for g in optimizer.param_groups}
                optimizer = optim.AdamW(
                    [{'params': main_params,  'name': 'main',              'lr': current_lrs['main']},
                     {'params': thick_params, 'name': 'thickness_decoder', 'lr': current_lrs['thickness_decoder']},
                     {'params': gate_params,  'name': 'gate_params',       'lr': current_lrs['gate_params']}],
                    weight_decay=weight_decay_val)

                # (b) ALDA 승수 초기화(rho는 유지)
                alda_state['mu_Mp'] = 1.0
                alda_state['mu_col'] = 1.0
                print(f"[Reset] optimizer 재인스턴스화 완료(모멘텀 flush), mu_Mp/mu_col → 1.0 "
                      f"(rho_Mp/rho_col 유지: {alda_state['rho_Mp']:.1f}/{alda_state['rho_col']:.1f})")

                # (c) 쿨다운 시작 — 다음 epoch의 train_step부터 best_so_far_mp_rel_err 갱신이
                #     자동으로 프리즈된다(Phase 4, /synod design 세션에서 off-by-one 없음 확인)
                pruning_state['cooldown_remaining'] = PRUNING_COOLDOWN_EPOCHS
                print(f"[Cooldown] 이후 {PRUNING_COOLDOWN_EPOCHS}epoch 동안 프루닝 트리거 판정 및 "
                      f"적응형 TAU_GATE best_so_far 갱신을 정지")

        is_feasible = (info['mp_rel_err'] < feasibility_mp_err) and (info['l_collision'] < feasibility_collision)
        if is_feasible:
            if first_feasible_epoch is None:
                first_feasible_epoch = epoch
            if (not best_feasible['found']) or (info['mp_rel_err'] < best_feasible['mp_rel_err']):
                best_feasible['found']       = True
                best_feasible['epoch']       = epoch
                best_feasible['mp_rel_err']  = info['mp_rel_err']
                best_feasible['l_collision'] = info['l_collision']
                best_feasible['state_dict']  = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_feasible['pruning_state'] = {
                    'ewma_z': pruning_state['ewma_z'].clone(),
                    'state': list(pruning_state['state']),
                    'pending_duration': pruning_state['pending_duration'].clone(),
                    # [command_v17_paratune.md Phase 5] 적응형 TAU_GATE 상태도 체크포인트에 포함
                    'best_so_far_mp_rel_err': pruning_state['best_so_far_mp_rel_err'],
                    'ema_best_so_far': pruning_state['ema_best_so_far'],
                    'current_tau_gate': pruning_state['current_tau_gate'],
                }

        if info['mp_rel_err'] < best_mp['mp_rel_err']:
            best_mp['epoch']       = epoch
            best_mp['mp_rel_err']  = info['mp_rel_err']
            best_mp['l_collision'] = info['l_collision']
            best_mp['state_dict']  = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_mp['pruning_state'] = {
                'ewma_z': pruning_state['ewma_z'].clone(),
                'state': list(pruning_state['state']),
                'pending_duration': pruning_state['pending_duration'].clone(),
                # [command_v17_paratune.md Phase 5]
                'best_so_far_mp_rel_err': pruning_state['best_so_far_mp_rel_err'],
                'ema_best_so_far': pruning_state['ema_best_so_far'],
                'current_tau_gate': pruning_state['current_tau_gate'],
            }

        if epoch == 0 or (epoch + 1) % 20 == 0:
            with torch.no_grad():
                snap_coords = new_coords.detach().cpu()
                snap_y_pna  = compute_y_pna_ref(snap_coords, base_t_cpu, fy_full, edge_index_cpu)
            history['snapshots'].append({
                'epoch':   epoch,
                'coords':  snap_coords,
                'y_pna':   snap_y_pna,
                'pred_mp': float(np.sum(info['pred_mp'])),
            })

        if (epoch + 1) % 20 == 0 or epoch == 0:
            flag = " [FEASIBLE]" if is_feasible else ""
            zg = info['z_gate']
            gact = "Y" if info['gate_active'] else "N"
            print(f"{epoch:05d} || {info['loss']:.4f} || {info['mp_rel_err']*100:6.2f}% | "
                  f"{info['l_smooth']:.4f} | {info['area']:6.1f} | {info['l_collision']:.4f} | "
                  f"{info['l_sparse']:.4f} || {info['thickness_gate']:.2f} | {gact:>4s} | "
                  f"{alda_state['mu_Mp']:.1e} | {zg[3]:.2f}/{zg[4]:.2f} | "
                  f"{info['temperature']:.3f} | {info['current_tau_gate']:.4f}{flag}")
            if diagnostic:
                print(f"  [DIAGNOSTIC] part {diagnostic_part_id} ewma_z = "
                      f"{pruning_state['ewma_z'][diagnostic_part_id]:.4f} "
                      f"(thresh_low={pruning_state['thresh_low']})")

    final_new_coords = new_coords.detach().cpu() if new_coords is not None else base_coords

    print(f"\n{'─' * 78}")
    if best_feasible['found']:
        print(f"[Feasibility] 첫 만족 epoch: {first_feasible_epoch}  |  "
              f"최고(Mp err 최소) epoch: {best_feasible['epoch']}  |  "
              f"Mp err: {best_feasible['mp_rel_err']*100:.2f}%  |  "
              f"l_collision: {best_feasible['l_collision']:.4f}")
    else:
        print(f"[Feasibility] 학습 전체(max_epochs={max_epochs}) 동안 "
              f"'Mp err < {feasibility_mp_err*100:.1f}% AND l_collision < {feasibility_collision}' "
              f"조건을 동시에 만족한 epoch이 없음.")
    print(f"[Best-Mp] epoch {best_mp['epoch']}: Mp err = {best_mp['mp_rel_err']*100:.2f}%, "
          f"l_collision = {best_mp['l_collision']:.4f}  (collision 무관 추적)")
    print(f"[ALDA final] mu_Mp={alda_state['mu_Mp']:.2e} mu_col={alda_state['mu_col']:.2e} "
          f"rho_Mp={alda_state['rho_Mp']:.1f} rho_col={alda_state['rho_col']:.1f}")
    print(f"[Pruning final] state={pruning_state['state']} "
          f"ewma_z={[round(v, 3) for v in pruning_state['ewma_z'].tolist()]}")
    print(f"[Adaptive TAU_GATE final] best_so_far_mp_rel_err={pruning_state['best_so_far_mp_rel_err']:.4f} "
          f"ema_best_so_far={pruning_state['ema_best_so_far']:.4f} "
          f"current_tau_gate={pruning_state['current_tau_gate']:.4f}")
    print(f"{'─' * 78}")

    return history, base_coords, final_new_coords, part_labels_t, best_feasible, best_mp, pruning_state


# ══════════════════════════════════════════════════════════════════
# SECTION 5: 시각화 (v17 골격 유지 — 이번 범위에서 신규 패널 추가하지 않음)
# ══════════════════════════════════════════════════════════════════

def pruning_state_deleted_ids(pruning_state):
    """pruning_state에서 state=='DELETED'인 part_id 리스트를 반환(None-safe)."""
    if pruning_state is None:
        return []
    return [i for i, s in enumerate(pruning_state.get('state', [])) if s == 'DELETED']


def visualize_training(history, base_coords, result_coords, target_mp_val, part_labels=None,
                        best_feasible=None, pruning_state=None):
    fig, axes = plt.subplots(2, 4, figsize=(26, 9))
    axes = axes.flatten()
    epochs = list(range(len(history['loss'])))

    ax = axes[0]
    ax.plot(epochs, history['loss'], color='#2196F3', linewidth=1.2, label='Total Loss')
    ax.axvline(STAGE2_EPOCH, color='gray', linestyle='--', linewidth=1.0, label=f'Stage2 ({STAGE2_EPOCH})')
    ax.axvline(GATE_ACTIVE_EPOCH, color='#E91E63', linestyle=':', linewidth=1.2, label=f'GateActive ({GATE_ACTIVE_EPOCH})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss 수렴', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[1]
    base_np = base_coords.numpy()
    result_np = result_coords.numpy()
    part_colors = {0: '#FF5722', 1: '#FFAA00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}
    part_names  = {0: '#00(Outer)', 1: '#03(Plate)', 2: '#06(Inner)', 3: '#07(Patch1)', 4: '#08(Patch2)'}
    pl = part_labels.numpy() if part_labels is not None else None
    deleted_parts = set(pruning_state_deleted_ids(pruning_state))
    for part_id in range(5):
        mask = (pl == part_id) if pl is not None else slice(None)
        c = part_colors[part_id]
        name = part_names[part_id]
        is_deleted = part_id in deleted_parts
        result_style = ':' if is_deleted else '-'
        result_alpha = 0.15 if is_deleted else 1.0
        result_linewidth = 1.0 if is_deleted else 1.8
        result_label = f'{name} Result [DELETED]' if is_deleted else f'{name} Result'
        ax.plot(base_np[mask, 0], base_np[mask, 1], 'o--', color=c, alpha=0.1, linewidth=1.2, label=f'{name} Base')
        ax.plot(result_np[mask, 0], result_np[mask, 1], 's' + result_style, color=c, alpha=result_alpha,
                linewidth=result_linewidth, label=result_label)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('단면 형상: Base vs Result(final epoch)  |  점선/반투명 = DELETED(z_gate≈0)', fontweight='bold')
    ax.legend(loc='best', fontsize=6.5, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    pred_mp_total = [float(np.sum(v)) for v in history['pred_mp']]
    ax.plot(epochs, [v / 1e6 for v in pred_mp_total], color='#2196F3', linewidth=1.2, label='Pred Mp')
    ax.axhline(target_mp_val / 1e6, color='#FF5722', linestyle=':', linewidth=2.0, label='Target Mp')
    if best_feasible is not None and best_feasible.get('found'):
        ax.axvline(best_feasible['epoch'], color='#4CAF50', linestyle='--', linewidth=1.5,
                   label=f"Best feasible (epoch {best_feasible['epoch']})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mp (MN·mm)')
    ax.set_title('Mp 수렴', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(epochs, history['l_smooth'], label='Smooth', linewidth=1.0)
    ax.plot(epochs, history['l_mass'], label='Mass(asym, monitor)', linewidth=1.0)
    ax.plot(epochs, history['l_collision'], label='Collision', linewidth=1.0)
    ax.plot(epochs, history['l_order'], label='Order', linewidth=1.2, color='#E91E63')
    ax.plot(epochs, history['l_sparse'], label='Sparse(z_open)', linewidth=1.2, color='#795548')
    ax.plot(epochs, history['l_anchor'], label='Anchor', linewidth=1.0)
    ax.plot(epochs, history['l_sat'], label='Sat', linewidth=1.0, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss term')
    ax.set_title('보조 손실 항 추이', fontweight='bold')
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(epochs, history['area'], color='#4CAF50', linewidth=1.5, label='Area (mm²)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Area (mm²)')
    ax.set_title('단면적(질량) 추이 — ALDA + 프루닝', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    ax.plot(epochs, [e * 100 for e in history['mp_rel_err']], color='#FF5722', linewidth=1.2, label='Mp rel err (%)')
    ax.axhline(2.0, color='gray', linestyle=':', linewidth=1.0, label='Feasibility 임계 (2%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mp 상대오차 (%)')
    ax.set_title('Mp 오차 추이 (Feasibility 판정용)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    ax = axes[6]
    z_gate_hist = np.array(history['z_gate'])  # [epoch, 5]
    part_colors5 = {0: '#FF5722', 1: '#FFAA00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}
    part_names5  = {0: '#00(Outer)', 1: '#03(Plate)', 2: '#06(Inner)', 3: '#07(Patch1)', 4: '#08(Patch2)'}
    for pid in range(5):
        ax.plot(epochs, z_gate_hist[:, pid], color=part_colors5[pid], linewidth=1.3, label=part_names5[pid])
    ax.axvline(GATE_ACTIVE_EPOCH, color='gray', linestyle='--', linewidth=1.0, label=f'GateActive ({GATE_ACTIVE_EPOCH})')
    ax.axhline(0.08, color='red', linestyle=':', linewidth=1.0, label='prune thresh (0.08)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('z_gate')
    ax.set_title('파트 존재 게이트(z_gate) 추이', fontweight='bold')
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    ax = axes[7]
    t_history = history['t_per_part']
    for part_id in range(5):
        if len(t_history) == 0 or part_id not in t_history[0]:
            continue
        t_series = [rec[part_id] for rec in t_history]
        ax.plot(epochs, t_series, color=part_colors5[part_id], linewidth=1.3, label=part_names5[part_id])
    ax.axvline(STAGE2_EPOCH, color='gray', linestyle='--', linewidth=1.0, label=f'Stage2 ({STAGE2_EPOCH})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Thickness t (mm)')
    ax.set_title('파트별 두께 변화 추이(게이팅 반영)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('uni_section_v18 학습 결과  |  build_bpillar_section() hat-profile 초기 형상 갱신',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    try:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'uni_section_v18_result.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"\n결과 저장: {out_path}")


def visualize_epoch_snapshots(history, base_coords, target_mp_val, part_labels=None, ncols=4,
                               pruning_state=None):
    """학습 epoch마다 저장된 단면 형상 스냅샷을 그리드로 나열 출력한다. (v17과 동일 로직)

    pruning_state가 주어지면 각 스냅샷 epoch 시점의 history['z_gate']를 thresh_low와 비교해
    그 시점에 이미 '사실상 삭제(z_gate < thresh_low)'로 볼 수 있는 파트를 점선/반투명으로 그린다.
    (최종 pruning_state['state']만 쓰면 모든 스냅샷에 동일하게 적용되어 부정확하므로, 스냅샷별로
    그 epoch의 z_gate 값을 직접 참조한다.)
    """
    snapshots = history['snapshots']
    n = len(snapshots)
    if n == 0:
        print("[visualize_epoch_snapshots] 저장된 스냅샷이 없습니다.")
        return

    part_colors = {0: '#FF5722', 1: '#FFAA00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}
    part_names  = {0: '#00(Outer)', 1: '#03(Plate)', 2: '#06(Inner)', 3: '#07(Patch1)', 4: '#08(Patch2)'}
    base_np = base_coords.numpy() if torch.is_tensor(base_coords) else np.asarray(base_coords)
    pl = part_labels.numpy() if (part_labels is not None and torch.is_tensor(part_labels)) else part_labels

    thresh_low = pruning_state.get('thresh_low', 0.08) if pruning_state is not None else 0.08
    z_gate_hist = np.array(history['z_gate']) if pruning_state is not None else None  # [epoch, 5]

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, snap in enumerate(snapshots):
        ax = axes[i]
        coords_np = snap['coords'].numpy() if torch.is_tensor(snap['coords']) else np.asarray(snap['coords'])
        snap_epoch = snap['epoch']
        for part_id in range(5):
            mask = (pl == part_id) if pl is not None else slice(None)
            c = part_colors.get(part_id, '#888888')
            is_deleted = (z_gate_hist is not None and snap_epoch < len(z_gate_hist)
                          and z_gate_hist[snap_epoch, part_id] < thresh_low)
            result_style = ':' if is_deleted else '-'
            result_alpha = 0.15 if is_deleted else 1.0
            result_linewidth = 1.0 if is_deleted else 1.6
            base_label = part_names.get(part_id, f'part{part_id}')
            label = f'{base_label} [DELETED]' if is_deleted else base_label
            ax.plot(base_np[mask, 0], base_np[mask, 1], 'o--', color=c, alpha=0.30, linewidth=1.0, markersize=2)
            ax.plot(coords_np[mask, 0], coords_np[mask, 1], 's' + result_style, color=c, alpha=result_alpha,
                    linewidth=result_linewidth, markersize=3, label=label if i == 0 else None)
        y_pna = snap.get('y_pna', None)
        if y_pna is not None:
            ax.axhline(y_pna, color='gray', linestyle=':', linewidth=1.2, label='y_pna' if i == 0 else None)
        pred_mp = snap.get('pred_mp', None)
        mp_err_txt = ""
        if pred_mp is not None and target_mp_val:
            mp_err = abs(pred_mp - target_mp_val) / target_mp_val * 100
            mp_err_txt = f" | MpErr {mp_err:.1f}%"
        ax.set_title(f"epoch {snap['epoch']}{mp_err_txt}", fontsize=9, fontweight='bold')
        ax.set_xlabel('X (mm)', fontsize=7); ax.set_ylabel('Y (mm)', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('uni_section_v18 — Epoch별 단면 형상 변화', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    try:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'uni_section_v18_epoch_snapshots.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"\nEpoch 스냅샷 결과 저장: {out_path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # [command_v17_paratune.md Phase 2] argparse는 __main__ 내부에서만 구성 — run_training은
    # diagnostic/diagnostic_part_id를 안전한 기본값(False/4)의 키워드 인자로 받으므로, 프로그램적으로
    # (CLI 인자 없이) run_training을 직접 호출하는 기존 호출부는 변경 없이 그대로 동작한다.
    parser = argparse.ArgumentParser(
        description="uni_section_v18: build_bpillar_section() hat-profile 초기 형상 갱신 및 학습 실행")
    parser.add_argument('--diagnostic', action='store_true',
                         help='candidate part의 log_alpha를 강제로 낮춰 ewma_z 하강 궤적을 관찰하는 진단 모드')
    parser.add_argument('--diagnostic-part-id', type=int, default=4, dest='diagnostic_part_id',
                         help='진단 대상 part_id (CANDIDATE_PARTS 중 하나, 기본 4=Patch2)')
    # Jupyter/ipykernel 환경에서 실행 시 sys.argv에 커널이 주입하는 `--f=...` 등 낯선 인자가
    # 섞여 들어와 parse_args()가 SystemExit(2)를 던지는 문제를 방지하기 위해 parse_known_args 사용.
    args, _unknown = parser.parse_known_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print("uni_section_v18: command_v18.md 반영 (build_bpillar_section() 초기 형상 cosmetic 갱신)")
    print("  - Part 0(Outer Hat)/Part 2(Inner Hat) else 분기 y_coord_node를 _hat_profile() 트라페조이드로 교체")
    print("  - Part 1(Inner Plate)/3(Patch1)/4(Patch2) y값, 노드 개수/x-spacing, fix_x/fix_y, 엣지, join_pairs 전부 v17_01과 동일")
    print("  - build_bpillar_section() 이외 학습 로직(CGDN/train_step/run_training/시각화)은 v17_01.py와 완전히 동일")
    if args.diagnostic:
        print(f"  - [DIAGNOSTIC MODE 활성] part {args.diagnostic_part_id}")

    data, node_registry = build_bpillar_section()
    print(f"\n데이터: nodes={data.x.shape} | edges={data.edge_index.shape}")

    # ── 초기 형상 정보(학습 시작 전) — 단면적(질량)/파트별 Mp 등 baseline 출력 ──
    _init_coords = data.x[:, :2]
    _init_t = data.x[:, 6:7]
    _init_fy = data.x[:, 7:8]
    _init_part_ids = data.x[:, 4].long()
    _init_area_total, _init_area_per_part = compute_section_area(
        _init_coords, _init_t, data.edge_index, part_ids=_init_part_ids)
    _init_mp_total, _init_y_pna = compute_edge_mp_pna(
        _init_coords, _init_t, _init_fy, data.edge_index)
    print(f"\n{'=' * 78}")
    print("[초기 형상 정보] (학습 시작 전 baseline)")
    print(f"  단면적(질량 proxy) : {_init_area_total:>10.1f} mm²")
    for _pid in sorted(_init_area_per_part.keys()):
        print(f"    part{_pid}: {_init_area_per_part[_pid]:>8.1f} mm²")
    print(f"  Mp(초기 형상)      : {_init_mp_total.item():>14,.0f} N·mm")
    print(f"  y_pna(초기 형상)   : {_init_y_pna.item():>10.2f} mm")
    print(f"{'=' * 78}")

    TARGET_MP = 27_421_470  # N·mm (v10~v18과 동일)
    target_mps = {0: TARGET_MP}

    weights = {
        'w_phys':      10.0,
        'w_collision':  5.0,   # ALDA로 대체되어 loss에 미사용(호환성 유지)
        'w_order':      1.0,
        'w_mass':       2.0,   # ALDA로 대체되어 loss에 미사용(호환성 유지)
        'w_smooth':     0.5,
        'w_anchor':     0.02,
        'w_sat':        0.01,
        'w_sparse':     W_SPARSE,
    }

    history, base_coords, result_coords, part_labels, best_feasible, best_mp, pruning_state = run_training(
        data,
        target_mps=target_mps,
        target_area=None,          # None → 초기 단면적 자동 스냅샷
        max_epochs=1500,
        lr=1e-3,
        weights=weights,
        curriculum=True,
        curriculum_ratio=(0.2, 0.7),
        snapshot_interval=10,
        feasibility_mp_err=0.02,
        feasibility_collision=0.05,
        diagnostic=args.diagnostic,
        diagnostic_part_id=args.diagnostic_part_id,
    )

    visualize_training(history, base_coords, result_coords, TARGET_MP, part_labels=part_labels,
                       best_feasible=best_feasible, pruning_state=pruning_state)

    visualize_epoch_snapshots(history, base_coords, TARGET_MP, part_labels=part_labels,
                               pruning_state=pruning_state)

    final_pred = float(np.sum(history['pred_mp'][-1]))
    final_err  = abs(final_pred - TARGET_MP) / TARGET_MP * 100
    initial_area, _ = compute_section_area(data.x[:, :2].cpu(), data.x[:, 6:7].cpu(), data.edge_index.cpu())
    final_area = history['area'][-1]
    area_change_pct = (final_area - initial_area) / initial_area * 100

    print(f"\n{'=' * 78}")
    print(f"최종 결과 요약 (마지막 epoch 기준)")
    print(f"  Target Mp     : {TARGET_MP:>14,.0f} N·mm")
    print(f"  Final pred_mp : {final_pred:>14,.0f} N·mm")
    print(f"  Final Error   : {final_err:>6.2f}%")
    print(f"  Final l_collision : {history['l_collision'][-1]:.4f}")
    print(f"  Final l_order : {history['l_order'][-1]:.4f}")
    print(f"  Initial area  : {initial_area:>8.1f} mm²")
    print(f"  Final area    : {final_area:>8.1f} mm²  ({area_change_pct:+.1f}% vs 초기)")
    print(f"  ALDA mu_Mp/mu_col (final) : {history['alda_mu_Mp'][-1]:.2e} / {history['alda_mu_col'][-1]:.2e}")
    print(f"  Final temperature : {history['temperature'][-1]:.3f}  |  Final TAU_GATE : {history['current_tau_gate'][-1]:.4f}")
    print(f"  Pruning 최종 상태: {pruning_state['state']}")
    deleted_parts = [i for i, s in enumerate(pruning_state['state']) if s == 'DELETED']
    if deleted_parts:
        print(f"  ★ 프루닝된 파트: {deleted_parts} (v17_paratune의 적응형 TAU_GATE로 실제 트리거 발동)")
    else:
        print(f"  프루닝 미발동 — candidate_parts {CANDIDATE_PARTS}의 최종 ewma_z: "
              f"{[round(v, 3) for i, v in enumerate(pruning_state['ewma_z'].tolist()) if i in CANDIDATE_PARTS]}")
    if best_feasible['found']:
        print(f"\n  ★ Feasible 모델(물리 제약 + Mp 동시 만족) 발견: epoch {best_feasible['epoch']}, "
              f"Mp err={best_feasible['mp_rel_err']*100:.2f}%, l_collision={best_feasible['l_collision']:.4f}")
        print(f"    (best_feasible['state_dict']를 model.load_state_dict()로, "
              f"best_feasible['pruning_state']를 함께 복원해 사용 권장)")
    else:
        print(f"\n  ⚠ Feasible 모델을 찾지 못함 — best-Mp 체크포인트(epoch {best_mp['epoch']}, "
              f"err {best_mp['mp_rel_err']*100:.2f}%)를 참고해 원인 분석 필요.")
    print(f"{'=' * 78}")
