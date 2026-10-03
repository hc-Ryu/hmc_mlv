#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, LayerNorm
from torch_geometric.data import Data


# In[2]:


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# In[3]:


"""
ImplicitPNASolver Validation Code v4
─────────────────────────────────────
v2 대비 추가:
  - run_training() : snapshot_interval마다 구조 형상(coords)과 y_pna 스냅샷 저장
  - visualize_epoch_snapshots() : 학습 진행에 따른 단면 형상 변화 + PNA 위치 시각화
    (Panel 2의 Base vs Result 스타일로, epoch 0→50→100→... 그리드 형태)

기존 기능 (v2 그대로):
  - Forward  : bisection y_pna 기반 Mp vs 해석적 Mp
  - Backward : IFT gradient vs Analytic vs Finite Difference (3-way)
  - Epoch별  : pred_mp vs target_mp 수렴 정확도
  - 구조      : 1 section, 3 parts (Outer/Reinf/Inner), 30 nodes
  - 모델      : 전체 CGDN v3 (hidden=128, layers=4, heads=4) 코드 그대로
"""

# ══════════════════════════════════════════════════════════════════
# ImplicitPNASolver + CGDN  (20260312_v3.py 그대로 — 수정 없음)
# ══════════════════════════════════════════════════════════════════

def compute_edge_mp_pna(coords, t, fy, edge_index, n_iter=50):
    """
    Thick Edge (2D Plate) PNA 이분탐색 + Mp 계산
    엣지 두께를 Y축으로 투영하여 수평 엣지도 연속적으로 처리 → 평형 잔차 ≈ 0.
    Returns: (mp_total, y_pna)
    """
    mask = edge_index[0] < edge_index[1]
    u, v = edge_index[0][mask], edge_index[1][mask]

    y_u, y_v = coords[u, 1], coords[v, 1]
    x_u, x_v = coords[u, 0], coords[v, 0]
    L = torch.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2 + 1e-12)
    t_e  = t[u].squeeze(-1)
    fy_e = fy[u].squeeze(-1)

    # 1. Y축 투영 두께
    dx   = torch.abs(x_u - x_v)
    t_y  = t_e * (dx / (L + 1e-12))
    y_max = torch.maximum(y_u, y_v)
    y_min = torch.minimum(y_u, y_v)
    y_top = y_max + t_y / 2.0
    y_bot = y_min - t_y / 2.0
    H     = torch.clamp(y_top - y_bot, min=1e-12)

    Area_fy = L * t_e * fy_e

    # ── Forward 1: Bisection for y_pna (no gradient) ──
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

    # ── Forward 2: Mp 계산 (y_pna 고정) ──
    alpha        = torch.clamp((y_top - y_pna) / H, 0.0, 1.0)
    centroid_top = y_top - (alpha * H) / 2.0
    centroid_bot = y_bot + ((1.0 - alpha) * H) / 2.0
    m_top        = alpha * (centroid_top - y_pna)
    m_bot        = (1.0 - alpha) * (y_pna - centroid_bot)
    mp_total     = torch.sum(Area_fy * (m_top + m_bot))

    return mp_total, y_pna


class ImplicitPNASolver(torch.autograd.Function):
    """
    미분 가능한 소성 중립축(PNA) 및 전소성 모멘트(Mp) 계산기
    Forward : Edge-based 이분탐색으로 순 축력 = 0인 y_pna 탐색 (평형 잔차 ~1e-8)
    Backward: IFT + ∂Mp/∂y_pna = 0 (평형 조건) → y_pna 고정 하에서 직접 미분만 계산
    """

    @staticmethod
    def forward(ctx, coords, t, fy, edge_index, n_iter=40):
        mp_pred, y_pna = compute_edge_mp_pna(coords, t, fy, edge_index, n_iter)

        mask = edge_index[0] < edge_index[1]
        ctx.save_for_backward(coords, t, fy, y_pna.reshape(1), edge_index)
        ctx.mask = mask
        return mp_pred

    @staticmethod
    def backward(ctx, grad_output):
        coords, t, fy, y_pna_buf, edge_index = ctx.saved_tensors
        mask = ctx.mask
        y_pna = y_pna_buf[0].detach()

        # ∂Mp/∂y_pna = -net_force = 0 (평형 조건) → 간접항 소거
        # y_pna 고정 상태에서 Mp의 coords에 대한 직접 미분만 계산 (Thick Edge)
        with torch.enable_grad():
            coords_g = coords.detach().requires_grad_(True)
            u, v = edge_index[0][mask], edge_index[1][mask]
            y_u, y_v = coords_g[u, 1], coords_g[v, 1]
            x_u, x_v = coords_g[u, 0], coords_g[v, 0]
            L = torch.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2 + 1e-12)
            t_e  = t[u].squeeze(-1)
            fy_e = fy[u].squeeze(-1)

            dx_b   = torch.abs(x_u - x_v)
            t_y_b  = t_e * (dx_b / (L + 1e-12))
            y_max  = torch.maximum(y_u, y_v)
            y_min  = torch.minimum(y_u, y_v)
            y_top_b = y_max + t_y_b / 2.0
            y_bot_b = y_min - t_y_b / 2.0
            H_b     = torch.clamp(y_top_b - y_bot_b, min=1e-12)
            Area_fy_b = L * t_e * fy_e

            alpha_b      = torch.clamp((y_top_b - y_pna) / H_b, 0.0, 1.0)
            centroid_top_b = y_top_b - (alpha_b * H_b) / 2.0
            centroid_bot_b = y_bot_b + ((1.0 - alpha_b) * H_b) / 2.0
            m_top_b      = alpha_b * (centroid_top_b - y_pna)
            m_bot_b      = (1.0 - alpha_b) * (y_pna - centroid_bot_b)
            mp_direct    = torch.sum(Area_fy_b * (m_top_b + m_bot_b))

        (grad_coords,) = torch.autograd.grad(mp_direct, coords_g)
        return grad_coords * grad_output, None, None, None, None


def calculate_mpl(coords, t, fy, edge_index, min_edge_length=1e-6):
    if edge_index is None or edge_index.numel() == 0:
        return torch.tensor(0.0, device=coords.device)

    src, dst = edge_index
    edge_len = torch.norm(coords[src] - coords[dst], dim=1)
    valid_mask = edge_len > min_edge_length

    if not valid_mask.any():
        return torch.tensor(0.0, device=coords.device)

    return ImplicitPNASolver.apply(coords, t, fy, edge_index[:, valid_mask])


# In[4]:


## Model Architecture (v3)
## ─────────────────────────────────────────────────────────────

class FiLMGenerator(nn.Module):
    """target_mp [B, 1] → (gamma, beta) [B, hidden]

    Fix 1: target_mp 정규화 (1,500,000 → 1.5) — 미정규화 시 Linear 가중치에 의해
            gamma/beta ≈ ±3.6M 발생 → h = gamma*h + beta로 hidden state 완전 파괴
    Fix 2: 마지막 레이어 zero-init → 초기 delta_gamma≈0, beta≈0 → identity mapping 보장
    Fix 3: (1 + delta_gamma) 공식 → 초기 gamma=1, 안정적 학습 시작
    """
    MP_SCALE = 1e6  # 1,500,000 → 1.5, 1,000,000 → 1.0

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, hidden_channels * 2),
        )
        ## 마지막 레이어 zero-init: 초기 delta_gamma=0, beta=0 → h = (1+0)*h + 0 = h
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, target_mp):
        ## 정규화: 1,500,000 → 1.5, 1,000,000 → 1.0
        target_mp_norm = target_mp / self.MP_SCALE
        out = self.net(target_mp_norm)
        delta_gamma, beta = torch.chunk(out, 2, dim=-1)
        ## Identity mapping 유도: gamma = 1 + δγ (초기 δγ≈0 → gamma≈1)
        gamma = 1.0 + delta_gamma
        return gamma, beta


class CGDNBlock(nn.Module):
    """GATv2Conv → LayerNorm → FiLM modulation → GELU → Residual  (AdaIN pattern)"""
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
        h = self.norm(h)              ## Normalize FIRST (AdaIN pattern)
        h = gamma * h + beta          ## FiLM AFTER norm → conditioning preserved
        h = F.gelu(h)
        h = h + h_res
        return h


class CGDN(nn.Module):
    """
    Constraint-aware Graph Deformation Network (v3)

    v3 개선사항:
    - in_channels=8: fix_x, fix_y 분리 (이전 is_fixed 단일 → 축별 독립 고정)
    - fix_x_mask, fix_y_mask: X축/Y축 변위 독립 hard constraint
    - join_pairs: 접합 노드 쌍을 hard projection으로 동시 이동 보장

    입력 노드 특징 (in_channels=8):
        [x, y, fix_x, fix_y, part_id, section_id, t, fy]
        fix_x=1: X좌표 고정, fix_y=1: Y좌표 고정 (독립 설정 가능)
    엣지 특징 (edge_dim=4):
        [길이, 각도, part_id, edge_type]
    조건부 입력:
        target_mp [N, 1] (노드별 목표 전소성 모멘트)
    join_pairs [P, 2]:
        hard joining 쌍 인덱스 — 두 노드의 좌표를 평균으로 hard projection
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        edge_dim: int = 4,
        max_displacement: float = 50.0,
    ):
        super().__init__()
        self.max_displacement = max_displacement
        self.num_layers = num_layers

        ## 1. Node Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            LayerNorm(hidden_channels),
            nn.GELU(),
        )

        ## 2. Per-block FiLM Generators
        self.film_generators = nn.ModuleList([
            FiLMGenerator(hidden_channels) for _ in range(num_layers)
        ])

        ## 3. GATv2 Message-Passing Blocks
        self.blocks = nn.ModuleList([
            CGDNBlock(hidden_channels, heads=heads, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        ## 4. Coordinate Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x, edge_index, edge_attr, target_mp,
                fix_x_mask, fix_y_mask, join_pairs=None):
        """
        x             : [N, in_channels]  노드 특징
        edge_index    : [2, E]
        edge_attr     : [E, edge_dim]
        target_mp     : [N, 1]   노드별 목표 전소성 모멘트
        fix_x_mask    : [N, 1]   X축 고정 Boolean 마스크
        fix_y_mask    : [N, 1]   Y축 고정 Boolean 마스크
        join_pairs    : [P, 2]   hard joining 노드 쌍 인덱스 (없으면 None)
        """
        h = self.node_encoder(x)

        ## Per-block FiLM conditioning
        for i, block in enumerate(self.blocks):
            gamma, beta = self.film_generators[i](target_mp)
            h = block(h, edge_index, edge_attr, gamma, beta)

        delta_coords = self.decoder(h)
        delta_coords = torch.clamp(delta_coords, -self.max_displacement, self.max_displacement)

        ## ── Hard constraint: X축/Y축 독립 고정 ──
        delta_x = delta_coords[:, 0:1] * (~fix_x_mask).float()
        delta_y = delta_coords[:, 1:2] * (~fix_y_mask).float()
        delta_coords = torch.cat([delta_x, delta_y], dim=1)

        new_coords = x[:, :2] + delta_coords

        ## ── Hard joining: 접합 노드 쌍을 좌표 평균으로 projection ──
        ## 두 노드 u, v의 변형 후 좌표를 (coord_u + coord_v) / 2로 강제
        ## gradient는 두 노드에 균등 분배됨 (autograd 정상 동작)
        if join_pairs is not None and join_pairs.shape[0] > 0:
            u_idx = join_pairs[:, 0]
            v_idx = join_pairs[:, 1]
            mid = (new_coords[u_idx] + new_coords[v_idx]) * 0.5  # [P, 2]
            new_coords = new_coords.clone()
            new_coords[u_idx] = mid
            new_coords[v_idx] = mid

        return new_coords, delta_coords



# In[5]:


## Loss Functions
## ─────────────────────────────────────────────────────────────

def compute_smoothness_loss(new_coords, base_coords, edge_index, edge_attr):
    """v3: 구조 엣지(edge_type=0) 기준 상대 길이 변화량 최소화"""
    src, dst = edge_index
    edge_type = edge_attr[:, 3]

    ## 구조 엣지만 사용 + 양방향 중복 제거
    mask = (src < dst) & torch.isclose(edge_type, torch.zeros_like(edge_type))
    if not mask.any():
        return torch.tensor(0.0, device=new_coords.device)

    src = src[mask]
    dst = dst[mask]
    new_diff = new_coords[src] - new_coords[dst]
    base_diff = base_coords[src] - base_coords[dst]
    new_len = torch.norm(new_diff, dim=1)
    base_len = torch.norm(base_diff, dim=1)
    rel_change = (new_len - base_len) / torch.clamp(base_len, min=1.0)
    return torch.mean(rel_change ** 2)


def compute_smoothness_loss_angle(new_coords, edge_index, edge_attr):
    """현재 각도만 고려해 노드별 좌우 엣지 각도 최소화 & 90도 미만 제한"""
    src, dst = edge_index
    edge_type = edge_attr[:, 3]

    ## 구조 엣지만 사용, 양방향 중복 제거 (undirected unique)
    mask = (src < dst) & torch.isclose(edge_type, torch.zeros_like(edge_type))
    if not mask.any():
        return torch.tensor(0.0, device=new_coords.device)

    src = src[mask]
    dst = dst[mask]

    # adjacency by structural edge
    num_nodes = new_coords.shape[0]
    # src->dst & dst->src (undirected)
    all_u = torch.cat([src, dst])
    all_v = torch.cat([dst, src])

    # node별 모든 이웃 저장
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(all_u.tolist(), all_v.tolist()):
        adj[u].append(v)

    left_angles = []
    right_angles = []

    for node, neighbors in enumerate(adj):
        if len(neighbors) < 2:
            continue

        # 좌/우 분류: x 좌표 기준
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

    # “각도 최소화” = 원점(0) 방향으로 유지 + 좌우 일치
    # 1) 절대 각도 크기 최소화
    # results += torch.mean(left_angles.pow(2) + right_angles.pow(2))

    # 2) 좌우 각도 차이 최소화
    angle_diff = (left_angles - right_angles + math.pi) % (2.0 * math.pi) - math.pi
    results += torch.mean(angle_diff.pow(2))

    # 90도 미만 제한
    max_rad = math.pi / 2.0
    left_violation = torch.relu(left_angles.abs() - max_rad)
    right_violation = torch.relu(right_angles.abs() - max_rad)
    results += torch.mean(left_violation.pow(2) + right_violation.pow(2))

    return results


def compute_mass_loss(new_coords, t, edge_index, edge_attr, target_area=None):
    """총 질량(또는 목표 질량 오차)을 계산

    - 기본 동작: 엣지 길이 * 두께 합산 → 면적(질량)
    - ``target_area``가 주어지면 상대 오차 제곱을 반환하여
      손실로 사용할 수 있음.
    """

    src, dst = edge_index
    edge_type = edge_attr[:, 3]

    ## 구조 엣지(edge_type=0)만 질량 근사에 사용 + 양방향 중복 제거
    mask = (src < dst) & torch.isclose(edge_type, torch.zeros_like(edge_type))
    src = src[mask]
    dst = dst[mask]

    seg_len = torch.norm(new_coords[src] - new_coords[dst], dim=1)
    t_src = t[src].squeeze(-1)
    area = torch.sum(seg_len * t_src)

    if target_area is not None and target_area > 0:
        return area, torch.abs(area - target_area) / (target_area + 1e-12)
    else:
        # target_area==0 또는 None인 경우: 기본 스케일 이관
        return area, area * 1e-6


def compute_collision_loss(new_coords, part_ids, section_ids, margin=0.5):
    """층별 다중 파트 간섭 방지 (Y좌표 기반)"""
    total_collision_loss = torch.tensor(0.0, device=new_coords.device)
    unique_sections = torch.unique(section_ids)

    for lvl in unique_sections:
        lvl_mask = (section_ids == lvl)
        y_lvl = new_coords[lvl_mask, 1]
        p_lvl = part_ids[lvl_mask]

        mask_1 = (p_lvl == 0)  # Outer
        mask_2 = (p_lvl == 1)  # Reinf
        mask_3 = (p_lvl == 2)  # Inner

        if mask_2.any() and mask_3.any():
            gap_3_2 = torch.clamp(y_lvl[mask_3].max() - y_lvl[mask_2].min() + margin, min=0.0)
            total_collision_loss += gap_3_2 ** 2
            gap_2_1 = torch.clamp(y_lvl[mask_2].max() - y_lvl[mask_1].min() + margin, min=0.0)
            total_collision_loss += gap_2_1 ** 2
        elif mask_3.any() and mask_1.any():
            gap_3_1 = torch.clamp(y_lvl[mask_3].max() - y_lvl[mask_1].min() + margin, min=0.0)
            total_collision_loss += gap_3_1 ** 2

    return total_collision_loss


def compute_collision_loss_v2(base_coords, new_coords, part_ids, section_ids, margin=0.5, 
                              width_axis=0, thickness_axis=1, search_radius=5.0):
    """
    초기 형상(Base Coords) 기준으로 파트 간 대응 쌍을 찾고, 
    변형된 형상(New Coords) 기준으로 간섭을 검사하는 완벽한 로직.
    """
    total_collision_loss = torch.tensor(0.0, device=new_coords.device)
    unique_sections = torch.unique(section_ids)

    for lvl in unique_sections:
        lvl_mask = (section_ids == lvl)
        
        # Base와 New 좌표 모두 마스킹 추출
        base_lvl = base_coords[lvl_mask]
        new_lvl = new_coords[lvl_mask]
        p_lvl = part_ids[lvl_mask]

        mask_outer = (p_lvl == 0)  # Outer (그릇 위쪽)
        mask_reinf = (p_lvl == 1)  # Reinf (그릇 중간)
        mask_inner = (p_lvl == 2)  # Inner (그릇 아래쪽)
        
        def calc_lagrangian_interference(b_top, b_bot, n_top, n_bot):
            if len(b_top) == 0 or len(b_bot) == 0:
                return torch.tensor(0.0, device=new_coords.device)
            
            # 1. [핵심 수정] 대응 쌍(Neighbor) 탐색은 '초기 좌표(base_coords)' 기준
            # 원래 설계에서 폭 방향으로 가까웠던 노드들만 짝을 지어줌
            w_top_base = b_top[:, width_axis].unsqueeze(1)
            w_bot_base = b_bot[:, width_axis].unsqueeze(0)
            
            w_dist_base = torch.abs(w_top_base - w_bot_base)
            neighbor_mask = (w_dist_base < search_radius)
            
            # 2. 간섭량(Violation) 계산은 '변형 후 좌표(new_coords)' 기준
            # 짝지어진 노드들이 현재 어떻게 변형되었는지 확인
            t_top_new = n_top[:, thickness_axis].unsqueeze(1)
            t_bot_new = n_bot[:, thickness_axis].unsqueeze(0)
            
            # 위반 조건: Top의 Y >= Bottom의 Y + margin
            violation = torch.clamp(margin - (t_top_new - t_bot_new), min=0.0)
            
            # 3. 초기 상태에서 가까웠던(neighbor_mask) 노드들의 현재 간섭량만 추출
            loss = torch.sum((violation ** 2) * neighbor_mask)
            return loss

        if mask_outer.any() and mask_reinf.any():
            total_collision_loss += calc_lagrangian_interference(
                base_lvl[mask_outer], base_lvl[mask_reinf],
                new_lvl[mask_outer], new_lvl[mask_reinf]
            )
            
        if mask_reinf.any() and mask_inner.any():
            total_collision_loss += calc_lagrangian_interference(
                base_lvl[mask_reinf], base_lvl[mask_inner],
                new_lvl[mask_reinf], new_lvl[mask_inner]
            )

    return total_collision_loss


def compute_collision_loss_v3(new_coords, part_ids, section_ids, margin=2, parts_order_in_sections=None):
    """
    모든 섹션(section)과 파트(part)에 대해 일반화된 계층적 침투 방지 손실을 계산합니다.

    part 계층 순서 결정
    - parts_order_in_sections가 주어지면 필요 시 섹션별로 전역 순서를 override합니다.
      예: {0:[0,2,1], 1:[0,1,2]} 같은 형태로 사용할 수 있습니다.
    - 지정되지 않았으면 섹션 내 고유 part_ids를 오름차순 정렬합니다.
    """
    total_loss = torch.tensor(0.0, device=new_coords.device, requires_grad=True)

    unique_sections = torch.unique(section_ids)
    valid_pairs_count = 0

    def compute_segment_penetration_loss(coords_outer, coords_inner, normal_direction_CW, margin=0.01):
        if coords_outer.shape[0] < 2 or coords_inner.shape[0] == 0:
            return torch.tensor(0.0, device=coords_outer.device, requires_grad=True)

        A = coords_outer[:-1]
        B = coords_outer[1:]
        AB = B - A

        P = coords_inner.unsqueeze(1)  # (M, 1, 2)
        A_exp = A.unsqueeze(0)        # (1, S, 2)
        AB_exp = AB.unsqueeze(0)      # (1, S, 2)

        AB_squared = torch.sum(AB_exp ** 2, dim=-1) + 1e-8  # (1, S)
        AP = P - A_exp  # (M, S, 2)
        t = torch.sum(AP * AB_exp, dim=-1) / AB_squared  # (M, S)

        # 2번 조건: C가 A-B 사이에 있는 경우만 고려
        valid_mask = (t >= 0.0) & (t <= 1.0)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=coords_outer.device, requires_grad=True)

        C = A_exp + t.unsqueeze(-1) * AB_exp  # (M, S, 2)

        # 모든 AB 세그먼트를 고려
        # tangent, normal은 세그먼트 방향으로 계산
        tangent = AB_exp / (torch.norm(AB_exp, dim=-1, keepdim=True) + 1e-8)  # (1, S, 2)
        if normal_direction_CW == True:
            normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
        elif normal_direction_CW == False:
            normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        else:
            raise ValueError("normal_direction_CW must be True (CW) or False (CCW)")

        # broadcast normal to (M, S, 2)
        normal = normal.expand(P.shape[0], -1, -1)

        CP = P - C
        projection = torch.sum(CP * normal, dim=-1)  # (M, S)
        violation = torch.relu(margin - projection)

        # 유효한 세그먼트만 포함
        violation = violation * valid_mask.float()

        # 평균을 유효 쌍의 개수로
        total_valid = valid_mask.sum().float().clamp(min=1.0)
        return torch.sum(violation)

    for sec in unique_sections:
        sec_mask = (section_ids == sec)
        sec_parts = part_ids[sec_mask]

        full_order = torch.tensor(parts_order_in_sections[int(sec.item())], dtype=part_ids.dtype, device=part_ids.device)
        unique_parts = torch.unique(sec_parts)
        ordered_parts = full_order[torch.isin(full_order, unique_parts)]

        if len(ordered_parts) < 2:
            continue

        for i in range(len(ordered_parts) - 1):
            outer_part_id = ordered_parts[i]

            inner_range = range(i + 1, len(ordered_parts))
            for j in inner_range:
                inner_part_id = ordered_parts[j]

                mask_outer = sec_mask & (part_ids == outer_part_id)
                mask_inner = sec_mask & (part_ids == inner_part_id)

                coords_outer = new_coords[mask_outer]
                coords_inner = new_coords[mask_inner]

                # 아래 파트가 위 파트를 침투하는 경우 평가 (기존 로직)
                loss_outer_inner = compute_segment_penetration_loss(coords_outer, coords_inner, normal_direction_CW=True, margin=margin)
                total_loss = total_loss + loss_outer_inner
                valid_pairs_count += 1

                # 위 파트가 아래 파트를 침투하는 경우도 추가 평가 (요청 사항)
                loss_inner_outer = compute_segment_penetration_loss(coords_inner, coords_outer, normal_direction_CW=False, margin=margin)
                total_loss = total_loss + loss_inner_outer
                valid_pairs_count += 1

    if valid_pairs_count > 0:
        total_loss = total_loss / valid_pairs_count

    return total_loss


def compute_section_continuity_loss(new_coords, base_coords, section_ids, part_ids):
    """인접 섹션 간 같은 파트의 평균 변위 차이 최소화"""
    loss = torch.tensor(0.0, device=new_coords.device)
    delta = new_coords - base_coords

    unique_sections = torch.unique(section_ids)
    unique_parts = torch.unique(part_ids)

    for i in range(len(unique_sections) - 1):
        sec_a = unique_sections[i]
        sec_b = unique_sections[i + 1]

        for part in unique_parts:
            mask_a = (section_ids == sec_a) & (part_ids == part)
            mask_b = (section_ids == sec_b) & (part_ids == part)

            if mask_a.any() and mask_b.any():
                mean_delta_a = delta[mask_a].mean(dim=0)
                mean_delta_b = delta[mask_b].mean(dim=0)
                loss += torch.sum((mean_delta_b - mean_delta_a) ** 2)
    return loss


def compute_shape_continuity_loss(new_coords, section_ids, part_ids):
    """
    인접 섹션 간 같은 파트의 '순수 형태(Shape)' 유사성을 강제합니다.
    (위치 이동은 무시하고 모양 자체를 비교)
    """
    loss = torch.tensor(0.0, device=new_coords.device)
    unique_sections = torch.unique(section_ids)
    unique_parts = torch.unique(part_ids)

    valid_pairs = 0

    for i in range(len(unique_sections) - 1):
        sec_a = unique_sections[i]
        sec_b = unique_sections[i + 1]

        for part in unique_parts:
            mask_a = (section_ids == sec_a) & (part_ids == part)
            mask_b = (section_ids == sec_b) & (part_ids == part)

            if mask_a.any() and mask_b.any():
                coords_a = new_coords[mask_a]
                coords_b = new_coords[mask_b]

                # 1. 무게중심(Centroid) 정렬: 위치(Translation) 차이 제거
                centered_a = coords_a - coords_a.mean(dim=0)
                centered_b = coords_b - coords_b.mean(dim=0)

                # 2. Chamfer Distance 연산을 위한 거리 행렬 계산
                # cdist는 (N, 2)와 (M, 2) 텐서 간의 모든 점-대-점 유클리디안 거리를 (N, M) 행렬로 반환
                dist_matrix = torch.cdist(centered_a, centered_b)

                # A의 각 점에서 B까지의 최단 거리 평균
                loss_a_to_b = dist_matrix.min(dim=1)[0].mean()
                # B의 각 점에서 A까지의 최단 거리 평균
                loss_b_to_a = dist_matrix.min(dim=0)[0].mean()

                # 양방향 거리를 더하여 대칭적인 형태 차이 페널티 부여
                loss += (loss_a_to_b + loss_b_to_a)
                valid_pairs += 1

    if valid_pairs > 0:
        loss = loss / valid_pairs

    return loss


def compute_repulsive_keepout_loss(new_coords, section_ids, keepout, height_multiplier=1.0):
    """
    가상의 자석(척력)을 이용해 노드를 부드럽게 밀어내는 Loss.
    - height_multiplier: 언덕의 높이 (에폭이 지날수록 증가시킴)
    """
    total_repulsive_loss = torch.tensor(0.0, device=new_coords.device)
    
    if keepout is None:
        return total_repulsive_loss

    unique_sections = torch.unique(section_ids)
    
    for lvl in unique_sections:
        lvl_idx = int(lvl.item())
        if lvl_idx in keepout:
            mask = (section_ids == lvl)
            x = new_coords[mask, 0]
            y = new_coords[mask, 1]

            for kz in keepout[lvl_idx]:
                k_x1, k_x2, k_y1, k_y2 = kz
                k_xmin, k_xmax, k_ymin, k_ymax = min(k_x1, k_x2), max(k_x1, k_x2), min(k_y1, k_y2), max(k_y1, k_y2)

                # 언덕의 중심점과 영향력 반경 설정 (0으로 나누기 방지)
                center_x = (k_xmin + k_xmax) / 2.0
                center_y = (k_ymin + k_ymax) / 2.0
                radius_x = max((k_xmax - k_xmin) / 2.0, 1e-6)
                radius_y = max((k_ymax - k_ymin) / 2.0, 1e-6)

                # 박스 내부에 들어온(또는 근접한) 노드들만 필터링
                in_box = (x > k_xmin) & (x < k_xmax) & (y > k_ymin) & (y < k_ymax)

                if in_box.any():
                    x_in = x[in_box]
                    y_in = y[in_box]

                    # # X, Y축 각각의 정규화 거리 (중앙=0.0, 경계=1.0 미만)
                    # dist_x = ((x_in - center_x) / radius_x) ** 2
                    # dist_y = ((y_in - center_y) / radius_y) ** 2

                    # # 거리가 0(중앙)일 때 최대값 1.0을 가지고, 멀어질수록 0에 수렴
                    # repulsion = torch.exp(-dist_x -dist_y)

                    # X, Y축 각각의 정규화 거리 (중앙=0.0, 경계=1.0 미만)
                    dist_x = abs((x_in - center_x) / radius_x)
                    dist_y = abs((y_in - center_y) / radius_y)

                    # 거리가 0(중앙)일 때 최대값 1.0을 가지고, 멀어질수록 0에 수렴
                    repulsion = torch.exp(-dist_x -dist_y +0.5)

                    # 커리큘럼 높이를 곱해 최종 합산
                    total_repulsive_loss += torch.sum(repulsion) * height_multiplier

    return total_repulsive_loss


# In[6]:


## Training Step
## ─────────────────────────────────────────────────────────────

def get_curriculum_weights(epoch, total_epochs, curriculum_ratio):
    """
    에폭 진행도에 따라 손실 함수의 가중치 multiplier(0→1)를 반환합니다.
    Phase 1 (0 ~ curriculum_ratio * total_epochs):
        형상/경계조건 안정화 우선
    Phase 2 (curriculum_ratio ~ 1.0):
        0→1로 증가
    Phase 3 (1.0 ~ total_epochs):
        모든 손실 가중치 1.0 고정
    """
    phase1_epochs, phase3_epochs = 0, total_epochs

    if curriculum_ratio[0] > 0.0:
        phase1_epochs = int(total_epochs * curriculum_ratio[0])
        denom = max(total_epochs - phase1_epochs, 1)  # div-by-zero 방지
        if epoch < phase1_epochs:
            progress = 0.0

    if curriculum_ratio[1] > 0.0:
        phase3_epochs = int(total_epochs * curriculum_ratio[1])
        if epoch >= phase3_epochs:
            progress = 1.0

    if epoch >= phase1_epochs and epoch < phase3_epochs:
        x = (epoch - phase1_epochs) / (phase3_epochs - phase1_epochs)
        progress = 0.5 * (1 + math.sin(math.pi * (x - 0.5))) # 0.0 → 1.0, 초반 완만 → 후반 가파른 증가

    s_phys       = 1.0
    s_smooth     = progress
    s_mass       = progress
    s_collision  = 1.0
    s_continuity = 1.0
    s_shape      = progress
    s_keepout    = progress
    return s_phys, s_smooth, s_mass, s_collision, s_continuity, s_shape, s_keepout


def train_step(model, data, optimizer, target_mps, target_area, keepout,
               epoch, max_epochs, weights, curriculum, 
               curriculum_ratio, parts_order_in_sections):
    """
    v3 변경사항:
    - 노드 피처 8열: [x, y, fix_x, fix_y, part_id, section_id, t, fy]
    - fix_x_mask, fix_y_mask 분리 추출
    - join_pairs 전달 (hard joining 적용)
    - 열 인덱스 전체 업데이트
    - target_area: 전체(또는 섹션별) 목표 질량을 지정하면
      compute_mass_loss에서 오차 기반 손실을 계산함
    """
    model.train()
    optimizer.zero_grad()

    x          = data.x                          ## [N, 8]
    # print(f"x: {x}")
    edge_index = data.edge_index                 ## [2, E]
    edge_attr  = data.edge_attr                  ## [E, 4]
    join_pairs = data.join_pairs if hasattr(data, 'join_pairs') else None
    base_coords = data.x[:, :2].detach()         ## [N, 2] 초기 좌표

    ## ── 노드 피처 추출 (v3: 8열 기준) ──
    fix_x_mask  = x[:, 2].bool().unsqueeze(1)    ## X축 고정 마스크
    fix_y_mask  = x[:, 3].bool().unsqueeze(1)    ## Y축 고정 마스크
    part_ids    = x[:, 4]
    section_ids = x[:, 5]
    t           = x[:, 6].unsqueeze(1)
    fy          = x[:, 7].unsqueeze(1)

    unique_sections = torch.unique(section_ids)

    ## ── 노드별 목표 Mp 텐서 생성 ──
    target_mp_node = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
    for section in unique_sections:
        section_mask = (section_ids == section)
        section_int = int(section.item())
        target_mp_node[section_mask] = target_mps[section_int]

    ## ── GNN Forward Pass ──
    new_coords, delta_coords = model(
        x, edge_index, edge_attr, target_mp_node,
        fix_x_mask, fix_y_mask, join_pairs
    )

    ## ── 층별 물리 손실 (L_phys) ──
    l_phys_total = torch.tensor(0.0, device=x.device)
    pred_mp_sections = []

    for section in unique_sections:
    # for section in range(1):
        section_mask = (section_ids == section)
        coords_section = new_coords[section_mask]
        t_section = t[section_mask]
        fy_section = fy[section_mask]

        src, dst = edge_index
        edge_mask = section_mask[src] & section_mask[dst]         # same-section edge 필터

        # structure edges only (edge_type == 0.0) to avoid zero-length part-binding edges
        edge_type = edge_attr[:, 3]
        physical_mask = edge_mask & torch.isclose(edge_type, torch.zeros_like(edge_type))

        edge_index_section = edge_index[:, physical_mask]

        # local index로 재매핑 (섹션 노드만 0..n-1)
        local_index = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        local_index[section_mask] = torch.arange(section_mask.sum(), device=x.device)
        edge_index_section = local_index[edge_index_section]

        pred_mp_section = calculate_mpl(coords_section, t_section, fy_section, edge_index_section)

        section_int = int(section.item())
        target_mp_section = torch.tensor(target_mps[section_int], dtype=torch.float32, device=x.device)

        # l_phys_section = ((pred_mp_section - target_mp_section) / target_mp_section) ** 2
        l_phys_section = abs((pred_mp_section - target_mp_section) / target_mp_section)
        # l_phys_section = (pred_mp_section - target_mp_section) ** 2
        # l_phys_section = abs(pred_mp_section - target_mp_section)
        l_phys_total += l_phys_section.squeeze()
        pred_mp_sections.append(pred_mp_section.item())

    num_sections = len(unique_sections)
    l_phys_total = torch.sqrt(l_phys_total) / num_sections
    pred_mp_sections = np.array(pred_mp_sections)

    ## ── 커리큘럼 가중치 (0→1 multiplier) ──
    s_phys, s_smooth, s_mass, s_collision, s_continuity, s_shape, s_keepout = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    if curriculum:
        (s_phys, s_smooth, s_mass, s_collision, s_continuity, s_shape, s_keepout
         ) = get_curriculum_weights(epoch, max_epochs, curriculum_ratio)

    ## ── 다목적 손실 계산 ──
    # l_smooth     = compute_smoothness_loss(new_coords, base_coords, edge_index, edge_attr)
    l_smooth     = compute_smoothness_loss_angle(new_coords, edge_index, edge_attr)
    area, l_mass = compute_mass_loss(new_coords, t, edge_index, edge_attr, target_area)
    # l_collision  = compute_collision_loss(new_coords, part_ids, section_ids)
    l_collision  = compute_collision_loss_v3(new_coords, part_ids, section_ids, parts_order_in_sections=parts_order_in_sections)
    l_continuity = compute_section_continuity_loss(new_coords, base_coords, section_ids, part_ids)
    l_shape      = compute_shape_continuity_loss(new_coords, section_ids, part_ids)
    l_keepout    = compute_repulsive_keepout_loss(new_coords, section_ids, keepout, height_multiplier=s_keepout)

    ## ── Total Loss ──
    loss = (weights['w_phys']       * l_phys_total * s_phys
          + weights['w_smooth']     * l_smooth     * s_smooth
          + weights['w_mass']       * l_mass       * s_mass
          + weights['w_collision']  * l_collision  * s_collision
          + weights['w_continuity'] * l_continuity * s_continuity
          + weights['w_shape']      * l_shape      * s_shape
          + weights['w_keepout']    * l_keepout    * s_keepout
          )

    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    # weights_norm = {
    #     "w_phys_norm":       (weights['w_phys']       * l_phys_total * s_phys) / (loss.item() + 1e-12),
    #     "w_smooth_norm":     (weights['w_smooth']     * l_smooth     * s_smooth) / (loss.item() + 1e-12),
    #     "w_mass_norm":       (weights['w_mass']       * l_mass       * s_mass) / (loss.item() + 1e-12),
    #     "w_collision_norm":  (weights['w_collision']  * l_collision  * s_collision) / (loss.item() + 1e-12),
    #     "w_continuity_norm": (weights['w_continuity'] * l_continuity * s_continuity) / (loss.item() + 1e-12),
    #     "w_shape_norm":      (weights['w_shape']      * l_shape      * s_shape) / (loss.item() + 1e-12),
    #     "w_keepout_norm":    (weights['w_keepout']    * l_keepout    * s_keepout) / (loss.item() + 1e-12)
    # }


    return {
        "loss":          loss.item(),
        "pred_mp":       pred_mp_sections,
        "l_phys":        l_phys_total.item(),
        "l_smooth":      l_smooth.item(),
        "initial_area":  area.item() if epoch == 0 else None,  # 첫 에폭에서 초기 면적 기록
        "area":          area.item(),
        "l_mass":        l_mass.item(),
        "l_collision":   l_collision.item(),
        "l_continuity":  l_continuity.item(),
        "l_shape":       l_shape.item(),
        "l_keepout":     l_keepout.item(),
        "new_coords":    new_coords.detach(),
        # "Weights_Norm":  weights_norm,
    }

def training(model, data, optimizer, target_mps, target_area, keepout, max_epochs, 
             weights, curriculum, curriculum_ratio, parts_order_in_sections):

    loss_hist         = []
    l_phys_hist       = []
    l_smooth_hist     = []
    initial_area      = None
    area_hist         = []
    l_mass_hist       = []
    l_collision_hist  = []
    l_continuity_hist = []
    l_shape_hist      = []
    l_keepout_hist    = []

    for epoch in range(max_epochs):

        info = train_step(model, data, optimizer, target_mps, target_area, keepout,
                          epoch, max_epochs, weights, curriculum, 
                          curriculum_ratio, parts_order_in_sections)

        loss_hist.append(info['loss'])
        l_phys_hist.append(info['l_phys'])
        l_smooth_hist.append(info['l_smooth'])
        area_hist.append(info['area'])
        l_mass_hist.append(info['l_mass'])
        l_collision_hist.append(info['l_collision'])
        l_continuity_hist.append(info['l_continuity'])
        l_shape_hist.append(info['l_shape'])
        l_keepout_hist.append(info['l_keepout'])

        if epoch == 0:
            print(f"Epoch ||  Loss  ||  Phys  |  Smth  | Area(cu/tg) | "
                  f" Mass  |  Coll  |  Cont  | Shape  | Keepout")

        if (epoch+1) % 100 == 0:
            print(f"{epoch:05d} || {info['loss']:.4f} || {info['l_phys']:.4f} | "
                f"{info['l_smooth']:.4f} | {info['area']:4.0f} / {target_area:4.0f} | "
                f"{info['l_mass']:.4f} | {info['l_collision']:6.4f} | "
                f"{info['l_continuity']:.4f} | {info['l_shape']:.4f} | "
                f"{info['l_keepout']:.4f}")


        if epoch == 0:
            initial_area = info['area']

    return (loss_hist, info, l_phys_hist, l_smooth_hist, initial_area, area_hist, l_mass_hist,
        l_collision_hist, l_continuity_hist, l_shape_hist, l_keepout_hist)


# In[7]:


## Data Construction (v3: 8-feature nodes + join_pairs)
## ─────────────────────────────────────────────────────────────
"""
B-Pillar 3x3 구조:
    section 0  (1층):  part 0, 1,    2
    section 1  (2층):  part 0, 1,    2
    section 2  (3층):  part 0, 1,    2
    section 3  (4층):  part 0, 1,    2
    section 4  (5층):  part 0, 1,    2
    section 5  (6층):  part 0, 1,    2
    section 6  (7층):  part 0, 1,    2
    section 7  (8층):  part 0, 1,    2
    section 8  (9층):  part 0, 1, 3, 2
    section 9  (10층): part 0, 1, 3, 2
    section 10 (11층): part 0, 1, 3, 2
    section 11 (12층): part 0, 1, 3, 2
    section 12 (13층): part 0,    3, 2
    section 13 (14층): part 0,    3, 2
    section 14 (15층): part 0,    3, 2
    section 15 (16층): part 0, 4, 3, 2
    section 16 (17층): part 0, 4, 3, 2

노드 피처 (8열):
  [x, y, fix_x, fix_y, part_id, section_id, t, fy]
  fix_x=1: X축 고정, fix_y=1: Y축 고정 (독립 설정)

경계조건:
  - is_fixed(xy 동시): 양 끝 플랜지 노드 i=0,1,8,9 → fix_x=fix_y=1
  - section 0,2의 Outer 중간 노드 i=3,6 → fix_x=fix_y=1

join_pairs:
  - 플랜지 위치(i=0,9)에서 파트 간 hard joining
  - 실제 용접/볼트 접합 표현 (동시이동 hard 보장)
"""
parts_in_sections = {
    0: [0, 4, 3, 2],  # 기존 Section 16의 파트 구성
}

parts_order_in_sections = {
    0: [0, 4, 3, 2],  # 기존 Section 16의 파트 계층 순서
}


lower_section = {
    0: [[  0.  ,   0.  ],
       [ 31.56,   0.  ],
       [ 63.12,   0.  ],
       [ 94.68,   6.  ],
       [126.24,  12.  ],
       [157.92,  42.  ],
       [189.48,  60.  ],
       [221.04,  60.  ],
       [252.6 ,  54.  ],
       [284.16,  54.  ],
       [315.84,  54.  ],
       [347.4 ,  54.  ],
       [378.96,  60.  ],
       [410.52,  60.  ],
       [442.08,  42.  ],
       [473.64,  12.  ],
       [505.32,   6.  ],
       [536.88,   0.  ],
       [568.44,   0.  ],
       [600.  ,   0.  ]],
    1: [[120.  , -18.  ],
       [138.96, -18.  ],
       [157.92, -18.  ],
       [176.76, -18.  ],
       [195.72, -18.  ],
       [214.8 ,   0.  ],
       [233.64,  24.  ],
       [252.6 ,  36.  ],
       [271.56,  36.  ],
       [290.52,  36.  ],
       [309.48,  36.  ],
       [328.44,  36.  ],
       [347.4 ,  36.  ],
       [366.36,  24.  ],
       [385.2 ,   0.  ],
       [404.16, -18.  ],
       [423.24, -18.  ],
       [442.08, -18.  ],
       [461.04, -18.  ],
       [480.  , -18.  ]],
    2: [[  0.  ,   0.  ],
       [ 31.56,   0.  ],
       [ 63.12,   0.  ],
       [ 94.68, -24.  ],
       [126.36, -30.  ],
       [157.92, -30.  ],
       [189.48, -30.  ],
       [221.04, -30.  ],
       [252.6 , -18.  ],
       [284.16, -18.  ],
       [315.84, -18.  ],
       [347.4 , -18.  ],
       [378.96, -30.  ],
       [410.52, -30.  ],
       [442.08, -30.  ],
       [473.64, -30.  ],
       [505.32, -24.  ],
       [536.88,   0.  ],
       [568.44,   0.  ],
       [600.  ,   0.  ]]
}

upper_section = {
    0: [[300.  ,   0.  ],
       [315.78,   0.  ],
       [331.56,   0.  ],
       [347.34,   6.  ],
       [363.12,  12.  ],
       [378.96,  42.  ],
       [394.74,  60.  ],
       [410.52,  60.  ],
       [426.3 ,  54.  ],
       [442.08,  54.  ],
       [457.92,  54.  ],
       [473.7 ,  54.  ],
       [489.48,  60.  ],
       [505.26,  60.  ],
       [521.04,  42.  ],
       [536.82,  12.  ],
       [552.66,   6.  ],
       [568.44,   0.  ],
       [584.22,   0.  ],
       [600.  ,   0.  ]],
    4: [[360.  , -18.  ],
       [369.48, -18.  ],
       [378.96, -18.  ],
       [388.38, -18.  ],
       [397.86, -18.  ],
       [407.4 ,   0.  ],
       [416.82,  24.  ],
       [426.3 ,  36.  ],
       [435.78,  36.  ],
       [445.26,  36.  ],
       [454.74,  36.  ],
       [464.22,  36.  ],
       [473.7 ,  36.  ],
       [483.18,  24.  ],
       [492.6 ,   0.  ],
       [502.08, -18.  ],
       [511.62, -18.  ],
       [521.04, -18.  ],
       [530.52, -18.  ],
       [540.  , -18.  ]],
    3: [[360.  , -24.  ],
       [369.48, -24.  ],
       [378.96, -24.  ],
       [388.38, -24.  ],
       [397.86, -24.  ],
       [407.4 , -24.  ],
       [416.82, -12.  ],
       [426.3 , -12.  ],
       [435.78, -12.  ],
       [445.26, -15.  ],
       [454.74, -15.  ],
       [464.22, -12.  ],
       [473.7 , -12.  ],
       [483.18, -12.  ],
       [492.6 , -24.  ],
       [502.08, -24.  ],
       [511.62, -24.  ],
       [521.04, -24.  ],
       [530.52, -24.  ],
       [540.  , -24.  ]],
    2: [[300.  ,   0.  ],
       [315.78,   0.  ],
       [331.56,   0.  ],
       [347.34, -24.  ],
       [363.18, -30.  ],
       [378.96, -30.  ],
       [394.74, -30.  ],
       [410.52, -30.  ],
       [426.3 , -18.  ],
       [442.08, -18.  ],
       [457.92, -18.  ],
       [473.7 , -18.  ],
       [489.48, -30.  ],
       [505.26, -30.  ],
       [521.04, -30.  ],
       [536.82, -30.  ],
       [552.66, -24.  ],
       [568.44,   0.  ],
       [584.22,   0.  ],
       [600.  ,   0.  ]]
}

num_sections = 1  # 17에서 1로 변경
num_nodes = 20
num_nodes_total = sum(len(parts)*num_nodes for parts in parts_in_sections.values())

x = torch.zeros((num_nodes_total, 8), dtype=torch.float32)
node_registry = {}
current_idx = 0
for section in range(num_sections):
    parts_in_section = parts_in_sections[section]

    for part in parts_in_section:
        for i in range(num_nodes):
            # 오직 upper_section(기존 최상단)의 좌표만 사용
            if part == 1:
                x_coord = upper_section[4][i][0]
                y_coord = upper_section[4][i][1]
            else:
                x_coord = upper_section[part][i][0]
                y_coord = upper_section[part][i][1]

            # 기존 Section 16은 전체가 고정되어 있었으나, 최적화를 위해 플랜지만 고정
            fix_x = 1.0 if (i in [0, 1, 2, 17, 18, 19]) else 0.0
            fix_y = 1.0 if (i in [0, 1, 2, 17, 18, 19]) else 0.0

            t_val  = 1.5 if part != 1 else 2.0
            fy_val = 1500.0 if part != 2 else 1200.0

            ## [x, y, fix_x, fix_y, part_id, section_id, t, fy]
            x[current_idx] = torch.tensor([x_coord, y_coord, fix_x, fix_y,
                                            float(part), float(section), t_val, fy_val])
            node_registry[(section, part, i)] = current_idx
            current_idx += 1

## ── Edge Construction ──
src_list, dst_list = [], []
edge_attr_list = []

def add_edge(u, v, part_id, edge_type):
    src_list.extend([u, v])
    dst_list.extend([v, u])
    dx = x[v, 0] - x[u, 0]
    dy = x[v, 1] - x[u, 1]
    length = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(dy, dx)
    edge_attr_list.extend([[length, angle, part_id, edge_type],
                            [length, -angle, part_id, edge_type]])

# Intra-section (횡방향)
for section in range(num_sections):
    parts = parts_in_sections[section]
    for part in parts:
        for i in range(num_nodes - 1):
            u = node_registry[(section, part, i)]
            v = node_registry[(section, part, i+1)]
            add_edge(u, v, part_id=part, edge_type=0.0)

edge_index_intra_section = torch.tensor([src_list, dst_list], dtype=torch.long)

# Inter-section (종방향: 3D 연속성) 엣지 생성 로직 삭제 (단일 층이므로 불필요)
# for section in range(num_sections - 1):
#     next_section = section + 1
#     # 현재 층과 다음 층에 공통으로 존재하는 파트만 추출 (교집합)
#     common_parts = set(parts_in_sections[section]).intersection(parts_in_sections[next_section])
#     for part in common_parts:
#         for i in range(num_nodes):
#             u = node_registry[(section, part, i)]
#             v = node_registry[(next_section, part, i)]
#             add_edge(u, v, part_id=part, edge_type=1.0)

# Flange binding (파트 끝점 결합)
for section in range(num_sections):
    parts = parts_in_sections[section]
    for i in [0, 19]:  # 플랜지 위치
        # 해당 층에 존재하는 파트들끼리 서로 모두 묶어줌 (완전연결/클릭)
        for p1_idx in range(len(parts)):
            for p2_idx in range(p1_idx + 1, len(parts)):
                p1 = parts[p1_idx]
                p2 = parts[p2_idx]
                u = node_registry[(section, p1, i)]
                v = node_registry[(section, p2, i)]
                add_edge(u, v, part_id=0.0, edge_type=2.0)

# Part binding (메시지 패싱 엣지 전체 연결, edge_type=2; hard joining은 끝단만)
join_pairs_list = []
# for section in [0, 1, 2]:
#     for i in range(num_nodes):
#         u0 = node_registry[(section, 0, i)]
#         u2 = node_registry[(section, 2, i)]
#         if section in [0, 1]:
#             u1 = node_registry[(section, 1, i)]

#         # hard joining: 실제 접합 위치(끝단)만 강제
#         if i in [3, 4, 5, 6]:
#             # join_pairs_list.append([u0, u2])
#             if section in [0, 1]:
#                 join_pairs_list.append([u0, u1])
#                 # join_pairs_list.append([u1, u2])

edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float32)
join_pairs = torch.tensor(join_pairs_list, dtype=torch.long)

## ── Keep-out Zone 설정 (단일 층으로 변경, 기존 Section 16의 제약조건) ──
keepout = {
    0: [(0.0, 600.0, -60.0, -120.0), (0.0, 600.0, 120.0, 180.0)]
}

print(f"Nodes: {x.shape} | Edges: {edge_index.shape} | "
      f"Edge Features: {edge_attr.shape} | Join Pairs: {join_pairs.shape}")


## ─────────────────────────────────────────────────────────────
## 2D Visualization (단일 section이므로 3D graph 대신 2D graph만 출력)
## ─────────────────────────────────────────────────────────────

part_name = {0: 'Part0', 1: 'Part1', 2: 'Part2', 3: 'Part3', 4: 'Part4', 5: 'Part5'}
color_map = {0: '#FF5722', 1: '#FFEE00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}

x_np  = x.cpu().numpy()
ei_np = edge_index.cpu().numpy()
ea_np = edge_attr.cpu().numpy()

plt.figure(figsize=(8, 6))
## upper section만 고려하므로 lower section은 출력하지 않음
plt.plot(np.array(upper_section[0])[:,0], np.array(upper_section[0])[:,1], label='Part 0', color=color_map[0])
plt.plot(np.array(upper_section[2])[:,0], np.array(upper_section[2])[:,1], label='Part 2', color=color_map[2])
plt.plot(np.array(upper_section[3])[:,0], np.array(upper_section[3])[:,1], label='Part 3', color=color_map[3])
plt.plot(np.array(upper_section[4])[:,0], np.array(upper_section[4])[:,1], label='Part 4', color=color_map[4])
plt.title('Upper Section (Initial)')
plt.axis('equal')
plt.xlim(-50, 650)
plt.ylim(-90, 150)
plt.legend()
for sec_id, rects in keepout.items():
    if sec_id == 0:
        for kx_min, kx_max, ky_min, ky_max in rects:
            plt.gca().add_patch(plt.Rectangle((kx_min, ky_min), kx_max - kx_min, ky_max - ky_min,
                                             edgecolor='red', facecolor='red', linestyle='--', linewidth=1.1, alpha=0.6))
plt.tight_layout()
plt.show()


# In[8]:


## Training
## ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    ## ── Model Configuration ──
    in_channels      = 8     ## [x, y, fix_x, fix_y, part_id, section_id, t, fy]
    hidden_channels  = 128
    num_layers       = 4
    heads            = 4
    edge_feature_dim = 4
    learning_rate    = 1e-3
    weight_decay     = 1e-4
    max_epochs       = 500
    target_mps       = {
        0: 55645928  # 기존 Section 16의 목표 전소성 모멘트
    }
    target_area      = 0
    curriculum       = True
    curriculum_ratio = [0.1, 0.5] # phase 1, phase 3
    weights          = {'w_phys': 10.0, 
                        'w_smooth': 0.1, 
                        'w_mass': 1.0,
                        'w_collision': 10.0, 
                        'w_continuity': 0.01,
                        'w_shape': 0.1,
                        'w_keepout': 10.0}
    print(f"{chr(10).join(f'{k}: {v}' for k, v in weights.items())}")
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = CGDN(in_channels=in_channels,
                 hidden_channels=hidden_channels,
                 num_layers=num_layers,
                 heads=heads,
                 edge_dim=edge_feature_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), 
                            lr=learning_rate, 
                            # weight_decay=weight_decay,
                            )
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                join_pairs=join_pairs).to(device)

    ## ── Training ──
    (loss_hist, info, l_phys_hist, l_smooth_hist, initial_area, area_hist, l_mass_hist, 
     l_collision_hist, l_continuity_hist, l_shape_hist, l_keepout_hist
     ) = training(model, data, optimizer, target_mps, target_area, keepout, max_epochs,
                  weights, curriculum, curriculum_ratio, parts_order_in_sections)

    ## ── Loss 시각화 ──
    epochs = list(range(max_epochs))
    labels = ['Total loss', 'L_phys', 'L_smooth', 'L_mass', 'L_collision', 'L_continuity', 'L_shape', 'L_keepout']
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, 10), sharex=True)
    hists  = [loss_hist, l_phys_hist, l_smooth_hist, l_mass_hist, l_collision_hist,
               l_continuity_hist, l_shape_hist, l_keepout_hist]

    for ax, lbl, hist, c in zip(axes, labels, hists, [f'C{i}' for i in range(len(labels))]):
        ax.plot(epochs, hist, label=lbl, color=c)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Epoch')
    plt.suptitle('CGDN v3 Training Loss', fontweight='bold')
    plt.tight_layout()
    plt.show()


# In[ ]:


## 2D Section-by-Section Visualization (v3: 열 인덱스 업데이트)
## ─────────────────────────────────────────────────────────────

## section_offsets, num_nodes_in_section 동적 계산 (parts_in_sections 기반)
num_nodes_in_section = [len(parts_in_sections[s]) * num_nodes for s in range(num_sections)]
section_offsets      = [int(sum(num_nodes_in_section[:s])) for s in range(num_sections)]

print(f"{'=' * 70}")
print(f"Initial and Final Mp (N·mm)")
print(f"{'Section':>9}     | {'initial':>10} | {'predicted':>10} | {'target':>10} | {'% error':>8}")
print(f"{'─' * 14}|{'─' * 12}|{'─' * 12}|{'─' * 12}|{'─' * 10}")

for i in range(num_sections - 1, -1, -1):
    s = section_offsets[i]
    e = s + num_nodes_in_section[i]

    section_mask = (data.x[:, 5] == i)
    src, dst = edge_index
    edge_mask = section_mask[src] & section_mask[dst]         # same-section edge 필터

    # structure edges only (edge_type == 0.0) to avoid zero-length part-binding edges
    edge_type = edge_attr[:, 3]
    edge_type = edge_type.to(edge_index.device)
    edge_mask = edge_mask.to(edge_index.device)
    physical_mask = edge_mask & torch.isclose(edge_type, torch.zeros_like(edge_type))

    edge_index_section = edge_index[:, physical_mask]

    # local index로 재매핑 (섹션 노드만 0..n-1)
    local_index = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
    local_index[section_mask] = torch.arange(section_mask.sum(), device=x.device)
    edge_index_section = local_index[edge_index_section].to(device)

    init_mp = calculate_mpl(data.x[s:e, :2], data.x[s:e, 6:7], data.x[s:e, 7:8], edge_index_section)
    print(f"  Section {i:2d}  | {init_mp:>10.0f} | {info['pred_mp'][i]:>10.0f} | {target_mps[i]:>10.0f} | {100 * abs(info['pred_mp'][i] - target_mps[i]) / target_mps[i]:>7.2f}%")


print("=" * 70)
print(f"Initial Area : {initial_area:.0f} mm²")
print(f"Final Area   : {info['area']:.0f} mm²")
print("=" * 70)
print(f"Keepout")
new_np = info['new_coords'].cpu().numpy()
xcoords = new_np[:, 0]
ycoords = new_np[:, 1]
violations = []
part_ids   = data.x[:, 4].detach().cpu().numpy().astype(int)
section_ids = data.x[:, 5].detach().cpu().numpy().astype(int)

for sec_id, kz_list in keepout.items():
    for kz in kz_list:
        kx_min, kx_max, ky_min, ky_max = kz
        sec_mask = (section_ids == int(sec_id))
        mask = sec_mask & ((xcoords > kx_min) & (xcoords < kx_max) &
                           (ycoords > ky_min) & (ycoords < ky_max))
        idxs = np.where(mask)[0]
        for i in idxs:
            violations.append((int(section_ids[i]), int(part_ids[i]), int(i), float(xcoords[i]), float(ycoords[i])))
if len(violations) == 0:
    print("  No keepout violations detected.")
else:
    print(f"  Found {len(violations)} keepout-violating nodes:")
    print(f"  {'section':>7}  {'part':>4}  {'node_id':>7}    {'x':>8}    {'y':>8}")
    for s, p, nid, xv, yv in violations:
        print(f"  {s:7d}  {p:4d}  {nid:7d}    {xv:8.3f}    {yv:8.3f}")
print("=" * 70)

## ── 2D subplot 시각화 (모든 섹션 하나의 figure에, section 16 위 / section 0 아래) ──

def draw_section_on_axes(ax_base, ax_def, coords, edge_index, edge_attr, x_features, title,
                          deformed_coords, section_start, keepout, part_name, color_map):
    """Draw base and deformed cross-section onto provided axes (no new figure)."""
    section_end = section_start + coords.shape[0]

    ei_mask = (edge_index[0] >= section_start) & (edge_index[0] < section_end) & \
              (edge_index[1] >= section_start) & (edge_index[1] < section_end)
    local_edge_index = edge_index[:, ei_mask] - section_start
    edge_attr_tmp = edge_attr[ei_mask, :]

    # Only show/draw edges with edge_attr[:,3] == 1 (target edge type)
    type1_edge_mask = edge_attr_tmp[:, 3] == 0
    local_edge_index = local_edge_index[:, type1_edge_mask]
    edge_attr_tmp = edge_attr_tmp[type1_edge_mask, :]

    sec_id    = int(x_features[0, 5].item())
    fix_x     = x_features[:, 2].cpu().numpy() > 0
    fix_y     = x_features[:, 3].cpu().numpy() > 0
    is_fixed  = fix_x | fix_y
    p_ids     = x_features[:, 4].cpu().numpy().astype(int)
    coords_np = coords.cpu().detach().numpy()
    ei        = local_edge_index.cpu().numpy()

    marker_map = {True: 's', False: 'o'}
    label_map  = {True: 'Fixed', False: 'Free'}

    def _draw(ax, pts, subtitle):
        for j in range(ei.shape[1]):
            sd, dd = ei[0, j], ei[1, j]
            ax.plot([pts[sd, 0], pts[dd, 0]], [pts[sd, 1], pts[dd, 1]],
                    color=color_map.get(int(edge_attr_tmp[j, 2])), linewidth=1.5, zorder=1)

        if keepout is not None and sec_id in keepout:
            from matplotlib.patches import Rectangle
            for kz in keepout[sec_id]:
                kx_min, kx_max, ky_min, ky_max = kz
                rect = Rectangle((kx_min, ky_min), kx_max - kx_min, ky_max - ky_min,
                                  linewidth=2.0, edgecolor='red', linestyle='--',
                                  facecolor='red', alpha=0.2, zorder=0)
                ax.add_patch(rect)

        for lid, color in color_map.items():
            for fix in [True, False]:
                mask_node = (p_ids == lid) & (is_fixed == fix)
                if not mask_node.any():
                    continue
                lbl = f'{part_name.get(lid, f"Part{lid}")} ({label_map[fix]})'
                ax.scatter(pts[mask_node, 0], pts[mask_node, 1],
                           c=color, marker=marker_map[fix],
                           s=20 if fix else 10, edgecolors='k',
                           linewidths=1.0 if fix else 0.5,
                           zorder=3, label=lbl)

        # for j in range(len(pts)):
        #     ax.annotate(str(j), (pts[j, 0], pts[j, 1]),
        #                 fontsize=6, ha='center', va='bottom',
        #                 xytext=(0, 4), textcoords='offset points', color='#333333')

        ax.set_title(subtitle, fontsize=8, fontweight='bold')
        ax.set_xlabel('X (mm)', fontsize=7)
        ax.set_ylabel('Y (mm)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        handles_l, labels_l = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_l, handles_l))
        ax.legend(by_label.values(), by_label.keys(), fontsize=5, loc='upper right')

    _draw(ax_base, coords_np, f'{title} — Base')

    if deformed_coords is not None and ax_def is not None:
        def_np = deformed_coords.cpu().detach().numpy()
        _draw(ax_def, def_np, f'{title} — Deformed')
        for j in range(len(coords_np)):
            dx = def_np[j, 0] - coords_np[j, 0]
            dy = def_np[j, 1] - coords_np[j, 1]
            if np.sqrt(dx**2 + dy**2) > 1e-3:
                ax_def.annotate('',
                    xy=(def_np[j, 0], def_np[j, 1]),
                    xytext=(coords_np[j, 0], coords_np[j, 1]),
                    arrowprops=dict(arrowstyle='-', color='red', lw=1.2, alpha=0.7))
    
# 기존 하드코딩된 section_order 변경 (단일 층: [0]이 됨)
section_order = list(range(num_sections - 1, -1, -1))
n_rows = len(section_order)

fig, axes = plt.subplots(n_rows, 2, figsize=(48, 16 * n_rows), squeeze=False)
# fig.suptitle('B-Pillar Cross Sections — All Sections (Top=Sec16, Bottom=Sec0)',
#              fontsize=14, fontweight='bold')

for row_idx, i in enumerate(section_order):
    s = section_offsets[i]
    e = s + num_nodes_in_section[i]

    base_coords_sec  = data.x[s:e, :2]
    deformed_sec     = info['new_coords'][s:e]
    section_features = data.x[s:e, :]

    target_mp_val  = target_mps[i]

    ## ========================================
    section_mask = torch.tensor(section_ids == i)

    src, dst = edge_index
    edge_mask = section_mask[src] & section_mask[dst]         # same-section edge 필터

    # structure edges only (edge_type == 0.0) to avoid zero-length part-binding edges
    edge_type = edge_attr[:, 3]
    physical_mask = edge_mask & torch.isclose(edge_type, torch.zeros_like(edge_type))
    # physical_mask = torch.logical_and(edge_mask, torch.isclose(edge_type, torch.zeros_like(edge_type)))

    edge_index_section = edge_index[:, physical_mask]

    # local index로 재매핑 (섹션 노드만 0..n-1)
    local_index = torch.full((data.x.shape[0],), -1, dtype=torch.long)
    local_index[section_mask] = torch.arange(section_mask.sum(), device=x.device)
    edge_index_section = local_index[edge_index_section]
    ## ========================================

    initial_mp_val = calculate_mpl(base_coords_sec,
                                   section_features[:, 6:7],
                                   section_features[:, 7:8],
                                   edge_index_section).item()
    final_mp_val   = info['pred_mp'][i]

    title_str = (f'Sec {i}  Target:{target_mp_val:.0f} | '
                 f'Init:{initial_mp_val:.0f} → Final:{final_mp_val:.0f}')

    draw_section_on_axes(
        ax_base=axes[row_idx, 0],
        ax_def=axes[row_idx, 1],
        coords=base_coords_sec,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        x_features=section_features,
        title=title_str,
        deformed_coords=deformed_sec,
        section_start=s,
        keepout=keepout,
        part_name=part_name,
        color_map=color_map
    )

plt.tight_layout()
plt.show()

# In[12]:


# 2. 메타데이터 생성: (section 번호, part 번호, 노드 인덱스)
current_metadata = []
for sec in range(num_sections):
    for part in sorted(parts_in_sections[sec]):
        for node_idx in range(num_nodes):
            current_metadata.append((sec, part, node_idx))

# 3. 흩어진 텐서 리스트와 메타데이터 짝짓기
paired_data = list(zip(current_metadata, info['new_coords']))

# 4. 원하는 기준으로 정렬
# x[0][1]: part (1순위)
# x[0][0]: section (2순위)
# x[0][2]: node index (3순위 - 10개 노드의 원래 순서 유지)
paired_data.sort(key=lambda x: (x[0][1], x[0][0], x[0][2]))

# 5. 정렬된 데이터에서 텐서만 다시 추출하여 하나의 2D 텐서 행렬로 병합
sorted_tensors = [data for meta, data in paired_data]
info['new_coords_sorted'] = torch.stack(sorted_tensors).cpu().numpy()

# 6. 각 part가 끝날 때마다 (0,0) 행 삽입
coords = info['new_coords_sorted']  # (570, 2)
sorted_metas = [meta for meta, _ in paired_data]

coords_list = []
prev_part = None
thickness_map = {0: 1.5, 1: 2.0, 2: 1.5, 3: 1.5, 4: 1.5}
for i, meta in enumerate(sorted_metas):
    section, part, node_idx = meta
    current_part = part
    if prev_part is not None and current_part != prev_part:
        marker_thickness = thickness_map.get(int(prev_part), 1.0)
        coords_list.append([int(0), int(0), int(0), int(0), int(0), marker_thickness])  # 파트 경계에서 실제 두께로 00000t
    x_val, y_val = coords[i]
    coords_list.append([int(section), int(node_idx) + 1, x_val, y_val, int(0), int(0)])  # node_idx를 1~20으로 수정
    prev_part = current_part

if prev_part is not None:
    last_thickness = thickness_map.get(int(prev_part), 1.0)
    coords_list.append([int(0), int(0), int(0), int(0), int(0), last_thickness])  # 마지막도 실제 두께

coords_with_zeros = np.array(coords_list)

## ----- CSV로 좌표 저장 (part 순서로 정렬 후 저장) -----
import pandas as pd
df = pd.DataFrame(coords_with_zeros, columns=['floor_idx', 'point_idx', 'x_mm', 'y_mm', 'r_mm', 't_mm'])
df.to_csv('floor_profiles_template.csv', index=False)


# In[13]:


## curriculum learning 시각화

import matplotlib.pyplot as plt
import numpy as np

max_epochs = 500
curriculum_ratio = [0.1, 0.5]

epochs = np.arange(max_epochs)
weights_history = {
    # 's_phys': [],
    's_smooth': [],
    # 's_mass': [],
    # 's_collision': [],
    # 's_continuity': [],
    # 's_shape': [],
    # 's_keepout': []
}

for epoch in epochs:
    s_phys, s_smooth, s_mass, s_collision, s_continuity, s_shape, s_keepout = \
        get_curriculum_weights(epoch, max_epochs, curriculum_ratio)
    # weights_history['s_phys'].append(s_phys)
    weights_history['s_smooth'].append(s_smooth)
    # weights_history['s_mass'].append(s_mass)
    # weights_history['s_collision'].append(s_collision)
    # weights_history['s_continuity'].append(s_continuity)
    # weights_history['s_shape'].append(s_shape)
    # weights_history['s_keepout'].append(s_keepout)

plt.figure(figsize=(8, 3))
for key, values in weights_history.items():
    plt.plot(epochs, values, label=key)

plt.title('Curriculum Loss Weight Schedule')
plt.xlabel('Epoch')
plt.ylabel('Weight multiplier (0→1)')
# plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[38]:


sec_ids = np.array(sorted(target_mps.keys()))
pred_vals = np.array([info['pred_mp'][int(i)] for i in sec_ids], dtype=float)
target_vals = np.array([target_mps[int(i)] for i in sec_ids], dtype=float)

abs_err = np.abs(pred_vals - target_vals)
pct_err = abs_err / (target_vals + 1e-12) * 100.0

fig, ax1 = plt.subplots(figsize=(5, 3))
width = 0.35
x = np.arange(len(sec_ids))
plt.plot(x, pct_err, 'o-', color='tab:green', linewidth=2, markersize=6, label='Relative error (%)')
# mean_err = np.mean(pct_err)
# ax1.axhline(mean_err, color='tab:blue', linestyle='--', linewidth=1.5, label=f'Mean error {mean_err:.2f}%')
plt.xlabel('Section ID')
plt.ylabel('Relative error (%)')
plt.xticks(x, sec_ids+1)
plt.tick_params(axis='y')
plt.ylim(0, 10)
plt.grid(False)
plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
plt.title('Relative Error of Mp (%)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 상위 오차 섹션 확인
top_n = 5
top_idx = np.argsort(-pct_err)[:top_n]
print('Top error sections (상대오차):')
for idx in top_idx:
    print(f'Section {sec_ids[idx]}: pred={pred_vals[idx]:.0f}, target={target_vals[idx]:.0f}, err={pct_err[idx]:.2f}%')

