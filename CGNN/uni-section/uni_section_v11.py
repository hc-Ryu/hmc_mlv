#!/usr/bin/env python
# coding: utf-8

"""
uni_section_v11.py
─────────────────────────────────────
uni_section_v10.py + 부재 제거(위상 경량화, Topology Lightweighting) 기능 추가.
설계 근거: docs/idea/idea_v0.md (Synod idea→review→design 숙의)
  - idea 라운드: L0 Hard-Concrete 존재 게이트 + 동시 최적화 + λ0 어닐링
  - review 라운드(v10 코드 대조): mass loss 역방향 패널티 / T_MIN 하한 / clearance 음수 버그
  - design 라운드: 구체 구현(HardConcreteGate, z 주입 위치, one-sided mass, collision z-gating)

목표 재정의:
  기존(v10)은 목표 Mp를 만족하되 모든 부재를 유지(경량화 항 없음, 오히려 초기면적 보존).
  v11은 "두께가 일정 이하로 내려가는 부재를 제거하여 질량을 줄인다"를 학습 목표로 추가.
  → 부재 소거는 회피 대상이 아니라 원하는 결과.

v10 → v11 핵심 변경점:
  [A] 파트별 존재 게이트 z_i ∈ [0,1] (HardConcreteGate, L0 relaxation)
      - t_final = z_i · t_raw  (게이트를 T_MIN 바깥에 적용 → z→0이면 물리 두께 진짜 0)
      - z는 Mp/면적/질량/충돌에 모두 전파(두께를 통해)
  [B] 동시 최적화 + λ0 어닐링: Stage2(epoch 128) 이후 희소화 압력 점증
      - 기존 2단계 thickness_gate 스케줄은 유지(option A), z 희소화를 그 위에 얹음
  [C] mass loss 개조(review §5 BLOCKER):
      v10  l_mass = ((area - target)/target)^2   ← 면적 감소도 처벌 → 경량화 방해
      v11  l_mass = relu(area - target)^2 (상한만)  +  λ0·mean(E[z]) (희소화 보상)
  [D] 충돌 게이팅(review §5 BUG 수정):
      clearance(음수 가능)에 z를 곱하면 부호반전 → 폐기.
      대신 최종 violation 항에 z_i·z_j 를 곱해 사라지는 부재의 충돌 무력화.
  [E] 필수 부재 보호: S_protect = {0, 1} = Outer Hat(#00), Inner Plate(#03) → z=1 고정.
      (주의: #03은 파트 '이름', part_id는 1. Inner Hat #06=2, Patch1 #07=3, Patch2 #08=4 가 프루닝 후보.)
  [F] 추론 시 하드컷: z<τ(0.5) 부재를 제거로 확정(학습 중엔 소프트 유지).

v10과 동일하게 유지:
  - CGDN(GATv2+FiLM) 백본, build_bpillar_section, native autograd Mp,
    collision v5 기하(부호 앵커/clearance), mesh order loss, 비대칭 Huber phys,
    커리큘럼, dual-threshold feasibility + best-Mp 체크포인트.
  - verify_thickness_gradient: raw 두께 텐서에 직접 작용(모델 forward 무관) → 게이트 영향 없음, 그대로 유지.
"""

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
# SECTION 0: Mp 계산 — native autograd (v8 §3.1 / v10 유지)
# ══════════════════════════════════════════════════════════════════

def compute_edge_mp_pna(coords, t, fy, edge_index, n_iter=50):
    """
    Thick Edge (2D Plate) PNA 이분탐색 + Mp 계산 (v10 그대로).
    y_pna는 no_grad(Envelope Theorem 적용점), Mp는 미분 가능 연산.
    t가 z-게이트된 t_final이면 Mp에 z가 자동 전파됨(Mp ∝ t).
    Returns: (mp_total, y_pna)
    """
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

    alpha        = torch.clamp((y_top - y_pna) / H, 0.0, 1.0)
    centroid_top = y_top - (alpha * H) / 2.0
    centroid_bot = y_bot + ((1.0 - alpha) * H) / 2.0
    m_top        = alpha * (centroid_top - y_pna)
    m_bot        = (1.0 - alpha) * (y_pna - centroid_bot)
    mp_total     = torch.sum(Area_fy * (m_top + m_bot))

    return mp_total, y_pna


def calculate_mpl(coords, t, fy, edge_index):
    mp_total, _ = compute_edge_mp_pna(coords, t, fy, edge_index)
    return mp_total


def verify_thickness_gradient(coords, t, fy, edge_index, eps=1e-4):
    """
    [v8 §3.1 / v10 유지] 학습 전 유한차분으로 ∂Mp/∂t 검증 — 실패 시 학습 중단.
    NOTE: raw 두께 텐서 t에 직접 작용(모델 forward·게이트와 무관)하므로 v11에서도 무수정 유지.
    """
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


# ══════════════════════════════════════════════════════════════════
# SECTION 0b: [v11 신규] Hard-Concrete 파트 존재 게이트 (L0 relaxation)
# ══════════════════════════════════════════════════════════════════

class HardConcreteGate(nn.Module):
    """
    [idea_v0.md 원칙 A] 파트별 존재 게이트 z_i ∈ [0,1] — L0 완화(Hard-Concrete, Louizos et al. 2018).

    - 학습 중: stretched concrete 확률 샘플(연속·미분 가능). 하드 프리징 없음(원칙 C).
    - 추론 중: deterministic sigmoid 게이트. 별도로 z<τ 하드컷은 호출부에서 확정.
    - 보호 파트(S_protect): z=1 고정, log_alpha 학습 제외(전체 붕괴 방지, 원칙 E / 가드).
    - z_expected = P(z>0): 경량화(희소화) 패널티에 사용.

    stretch 파라미터 gamma=-0.1, zeta=1.1, temperature beta=2/3 (표준값).
    init_log_alpha≈2.5 → 초기 z≈1(거의 열림)으로 시작해 안정적 워밍업 확보.
    """
    gamma = -0.1
    zeta  = 1.1
    beta  = 2.0 / 3.0

    def __init__(self, n_parts=5, protect_indices=(0, 1), init_log_alpha=2.5):
        super().__init__()
        self.n_parts = n_parts
        protected = torch.zeros(n_parts, dtype=torch.bool)
        if protect_indices:
            protected[list(protect_indices)] = True
        self.register_buffer('protected_mask', protected)          # [P] bool
        self.register_buffer('trainable_mask', ~protected)         # [P] bool

        # 학습 가능한 log_alpha는 비보호 파트에 대해서만 등록
        init = torch.full((int((~protected).sum().item()),), float(init_log_alpha))
        self.log_alpha = nn.Parameter(init)

    def _full_log_alpha(self):
        """비보호 파트는 학습값, 보호 파트는 큰 상수(→ z≈1)로 채운 [P] 텐서."""
        full = torch.full((self.n_parts,), 100.0, device=self.log_alpha.device)
        full = full.clone()
        full[self.trainable_mask] = self.log_alpha
        return full

    def forward(self):
        """
        Returns:
            z          [P] ∈ [0,1] — 실제 두께에 곱해질 게이트 (train=stochastic, eval=deterministic)
            z_expected [P] ∈ [0,1] — P(z>0), 희소화 패널티용 (보호 파트=1)
        """
        log_alpha = self._full_log_alpha()

        if self.training:
            u = torch.rand_like(log_alpha).clamp(1e-7, 1.0 - 1e-7)   # log NaN 가드
            s = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + log_alpha) / self.beta)
        else:
            s = torch.sigmoid(log_alpha)

        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_stretched, 0.0, 1.0)

        # 보호 파트 z=1 고정 (상수 1을 그래프에 주입: z*mask + (1-mask))
        keep = self.trainable_mask.float()
        z = z * keep + (1.0 - keep)

        # P(z>0) = sigmoid(log_alpha - beta·log(-gamma/zeta))
        const = self.beta * math.log(-self.gamma / self.zeta)
        z_expected = torch.sigmoid(log_alpha - const)
        z_expected = z_expected * keep + (1.0 - keep)   # 보호 파트는 1

        return z, z_expected

    @torch.no_grad()
    def hard_gate(self, tau=0.5):
        """[원칙 F] 추론용 하드컷: deterministic z가 τ 미만이면 0, 이상이면 원래 값."""
        was_training = self.training
        self.eval()
        z, _ = self.forward()
        self.train(was_training)
        z_hard = torch.where(z < tau, torch.zeros_like(z), z)
        return z_hard


# ══════════════════════════════════════════════════════════════════
# SECTION 0c: CGDN 백본 (v10 + 파트 존재 게이트)
# ══════════════════════════════════════════════════════════════════

class FiLMGenerator(nn.Module):
    """target_mp [B, 1] → (gamma, beta) [B, hidden] (v10 그대로)"""
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
    """GATv2Conv → LayerNorm → FiLM → GELU → Residual (v10 그대로)"""
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


class CGDN(nn.Module):
    """
    Constraint-aware Graph Deformation Network v11
    [v10 유지] T_MAX 2.5 / DELTA_SCALE 1.35 / bias 0.03 / Ghost Gate 삭제 / 2단계 두께 게이트
    [v11 신규] 파트 존재 게이트 z (HardConcreteGate) → t_final = z · t_raw (T_MIN 바깥 게이팅)

    forward() 반환값: (new_coords, delta_coords, t_final, delta_t_part, z_part, z_expected)
    """

    DELTA_SCALE = 1.35
    T_MIN       = 0.1    # mm — t_raw의 하한 (게이트 바깥에 위치)
    T_MAX       = 2.5    # mm

    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        edge_dim: int = 4,
        max_displacement: float = 50.0,
        n_parts: int = 5,
        protect_indices=(0, 1),      # [원칙 E] Outer Hat(#00)=0, Inner Plate(#03)=1
    ):
        super().__init__()
        self.max_displacement = max_displacement
        self.num_layers = num_layers

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

        # [v11] 파트 존재 게이트
        self.part_gate = HardConcreteGate(n_parts=n_parts, protect_indices=protect_indices)

    @staticmethod
    def leaky_tanh(x):
        """0.95·tanh(x) + 0.05·x/3 — gradient 하한 확보 (v10 유지)"""
        return 0.95 * torch.tanh(x) + 0.05 * x / 3.0

    def forward(self, x, edge_index, edge_attr, target_mp,
                fix_x_mask, fix_y_mask, join_pairs=None, thickness_gate=1.0):
        h = self.node_encoder(x)

        for i, block in enumerate(self.blocks):
            gamma, beta = self.film_generators[i](target_mp)
            h = block(h, edge_index, edge_attr, gamma, beta)

        # ── 좌표 예측 (v10 그대로) ──
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

        # ── 두께 예측 (v10 그대로) ──
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

        # [§4] 2단계 학습 게이트 (v10 유지, option A)
        delta_t_part = delta_t_part * thickness_gate

        # Soft clamp (logit-sigmoid, T_MAX=2.5) → t_raw (T_MIN 하한 포함)
        t_min, t_max = self.T_MIN, self.T_MAX
        t_initial_frac = (t_initial - t_min) / (t_max - t_min)
        t_initial_frac = torch.clamp(t_initial_frac, 1e-4, 1.0 - 1e-4)
        t_initial_logit = torch.logit(t_initial_frac)
        t_raw = t_min + (t_max - t_min) * torch.sigmoid(t_initial_logit + delta_t_part)

        # ── [v11] 파트 존재 게이트 주입: t_final = z · t_raw (T_MIN 바깥 → z→0이면 진짜 0) ──
        z_part, z_expected = self.part_gate()             # [P], [P]
        z_node = z_part[part_ids_local].unsqueeze(-1)     # [N,1]
        t_final = t_raw * z_node

        return new_coords, delta_coords, t_final, delta_t_part, z_part, z_expected


# ══════════════════════════════════════════════════════════════════
# SECTION 1: Loss Functions
# ══════════════════════════════════════════════════════════════════

def asymmetric_huber_phys(pred_mp, target_mp, delta=0.05, under_w=2.0):
    """[command_v10.md §5 / v10 유지] 비대칭 Huber phys loss."""
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
    """노드별 좌우 엣지 각도 최소화 & 90도 미만 제한 (v10 그대로)"""
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


def compute_mass_loss_v11(new_coords, t_final, edge_index, edge_attr, target_area):
    """
    [v11 개조 — review §5 BLOCKER] 경량화형 mass loss.
    v10: l_mass = ((area - target)/target)^2  → 면적 감소도 처벌(경량화 방해).
    v11: l_area_over = relu(area - target)^2   → 초기면적 '상한'만 처벌, 감소는 자유.
         (희소화 보상 λ0·mean(E[z])는 train_step에서 별도 항으로 결합 — 원칙 B/λ0 어닐링)
    t_final은 이미 z-게이트된 값이므로 area 자체가 살아있는 부재만 반영.
    """
    src, dst = edge_index
    edge_type = edge_attr[:, 3]

    mask = (src < dst) & torch.isclose(edge_type, torch.zeros_like(edge_type))
    src = src[mask]
    dst = dst[mask]

    seg_len = torch.norm(new_coords[src] - new_coords[dst], dim=1)
    t_src = t_final[src].squeeze(-1)
    area = torch.sum(seg_len * t_src)

    over = torch.relu(area - target_area) / (target_area + 1e-12)
    l_area_over = over ** 2
    return area, l_area_over


def compute_anchor_loss(new_coords, base_coords, fix_x_mask, fix_y_mask):
    disp = new_coords - base_coords
    return torch.mean(disp[:, 0] ** 2 + disp[:, 1] ** 2)


def compute_saturation_loss(delta_t_part, delta_scale=1.35, knee=0.9):
    """L_sat = relu(|delta_t| - knee×scale)^2 (v10 유지)"""
    threshold = knee * delta_scale
    return torch.mean(torch.relu(delta_t_part.abs() - threshold) ** 2)


def compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr, eps=0.5):
    """[command_v10.md §3 / v10 유지] 2D Mesh Order Loss."""
    src, dst = edge_index
    mask = (src < dst) & (edge_attr[:, 3] == 0.0)
    if not mask.any():
        return torch.tensor(0.0, device=new_coords.device)

    e0 = base_coords[dst[mask]] - base_coords[src[mask]]
    e0_hat = e0 / (e0.norm(dim=1, keepdim=True) + 1e-8)
    e_new = new_coords[dst[mask]] - new_coords[src[mask]]
    proj = (e_new * e0_hat).sum(dim=1)
    violation = torch.relu(eps - proj)
    return torch.mean(violation ** 2)


# ── [command_v10.md §2] Collision v5 (+ v11 z-게이팅) ──────────────

def _signed_projections(coords_seg, coords_pts):
    """점→세그먼트 부호 있는 법선 투영 (v10 그대로)."""
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
                          degenerate_clearance=-5.0):
    """[§2 / D2 / v10 그대로] 초기 형상 부호 앵커 + 쌍별 clearance 산정·고정."""
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

                    directions = []
                    for (c_seg, c_pt, roles) in [(ca, cb, (a, b)), (cb, ca, (b, a))]:
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
                        directions.append({
                            'seg_part': roles[0], 'pt_part': roles[1],
                            'sigma': sigma, 'clearance': clearance,
                        })
                    if directions:
                        spec[(sec_int, a, b)] = directions
    return spec


def compute_collision_loss_v11(new_coords, t_final, part_ids, section_ids,
                               collision_spec, z_part):
    """
    [command_v10.md §2 + v11 원칙 D] Collision v5 + 존재 게이트 마스킹.
      gap = sigma × projection - (t_seg + t_pt)/2 - clearance
      violation = relu(-gap).clamp(max=1.0)
      violation' = z_i · z_j · violation        ← [v11] 사라지는 부재의 충돌 무력화
      loss = mean(violation'^2)
    ⚠️ clearance(음수 가능)에 z를 곱하지 않음 — 부호반전 버그 회피(review §5).
       두께 항 -(t_i+t_j)/2 는 이미 z 게이트됨 → 자동 접촉 이완, 기하항은 z_i·z_j로 차단.
    """
    total_loss = torch.tensor(0.0, device=new_coords.device, requires_grad=True)
    n_dirs = 0

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
        # [v11] 존재 게이트 (미분 가능): 두 파트의 z 곱
        z_pair = z_part[a] * z_part[b]

        for d in directions:
            c_seg = part_coords[d['seg_part']]
            c_pt  = part_coords[d['pt_part']]
            if c_seg.shape[0] < 2 or c_pt.shape[0] == 0:
                continue
            proj, valid = _signed_projections(c_seg, c_pt)
            if valid.sum() == 0:
                continue

            t_sum_half = (part_t[d['seg_part']] + part_t[d['pt_part']]) / 2.0
            gap = d['sigma'] * proj - t_sum_half - d['clearance']
            violation = torch.relu(-gap).clamp(max=1.0) * valid.float()
            violation = violation * z_pair    # [v11] 게이트 적용
            loss_dir = (violation ** 2).sum() / valid.float().sum().clamp_min(1.0)
            total_loss = total_loss + loss_dir
            n_dirs += 1

    if n_dirs > 0:
        total_loss = total_loss / n_dirs
    return total_loss


# ══════════════════════════════════════════════════════════════════
# SECTION 2: 커리큘럼 & 2단계 두께 게이트 & [v11] λ0 어닐링
# ══════════════════════════════════════════════════════════════════

STAGE2_EPOCH = 128        # 두께 게이트 중심점 (v10 유지)
GATE_STEEPNESS = 0.6

# [v11] 희소화(경량화) 어닐링 파라미터 — 실험 튜닝 대상(idea_v0.md §5 남은 불확실성)
LAMBDA0_MAX  = 1.0        # 희소화 압력 상한
LAMBDA0_WARM = 40         # STAGE2 이후 선형 워밍업 epoch 수
PRUNE_TAU    = 0.5        # 추론 하드컷 임계


def thickness_gate_value(epoch):
    """gate = sigmoid(0.6×(epoch-128)) (v10 유지)"""
    return float(torch.sigmoid(torch.tensor(GATE_STEEPNESS * (epoch - STAGE2_EPOCH))).item())


def lambda0_value(epoch, stage2=STAGE2_EPOCH, lam_max=LAMBDA0_MAX, warm=LAMBDA0_WARM):
    """
    [v11 / idea_v0.md 원칙 B] 희소화 가중치 어닐링.
    Stage 1(형상·두께 확보) 동안 0 → Stage 2 진입 후 선형 워밍업 → lam_max.
    Mp가 붕괴했다 복구되는 비연속 경로를 피하고, 부재가 자연 도태되게 함.
    """
    if epoch < stage2:
        return 0.0
    return lam_max * min(1.0, (epoch - stage2) / max(warm, 1))


def get_curriculum_weights_v10(epoch, total_epochs, curriculum_ratio):
    """v8/v10 3-stage curriculum 유지 (s_mass는 두께 게이트가 대체)."""
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


# ══════════════════════════════════════════════════════════════════
# SECTION 3: Data Setup (build_bpillar_section — v10 그대로, 수정 없음)
# ══════════════════════════════════════════════════════════════════

def build_bpillar_section():
    """B-Pillar 5-Part 단면 (v8/v10과 100% 동일)"""
    part_configs = [
        (0, 30.0, 2.30, 1470.0, True),   # #00 Outer Hat   [보호]
        (1, 28.05, 1.60,  980.0, False), # #03 Inner Plate [보호]
        (2, 29.0, 1.60, 1470.0, True),   # #06 Inner Hat   [프루닝 후보]
        (3, 24.0, 1.40,  980.0, False),  # #07 Patch 1     [프루닝 후보]
        (4, 22.0, 1.60,  440.0, False),  # #08 Patch 2     [프루닝 후보]
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
                    y_coord_node = 60.0

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
                    y_coord_node = 45.0

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


def compute_y_pna_ref(coords, t, fy, edge_index, n_iter=50):
    with torch.no_grad():
        _, y_pna = compute_edge_mp_pna(coords, t, fy, edge_index, n_iter)
        return y_pna.item() if torch.is_tensor(y_pna) else y_pna


def compute_section_area(coords, t, edge_index, part_ids=None):
    """단면 면적 계산: A = Σ(L × t_e) [mm²] (v10 그대로)"""
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
               curriculum_ratio, collision_spec):
    """
    v10 대비 변경점 (v11):
    - forward가 z_part, z_expected 추가 반환 → t_final은 이미 z 게이트됨
    - mass loss: one-sided over-area + λ0·mean(E[z]) 희소화 항 (경량화 동력)
    - collision: violation에 z_i·z_j 곱
    - 로깅: z_part, pruned parts(z<τ), l_sparsity
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

    gate = thickness_gate_value(epoch)

    new_coords, delta_coords, t_final, delta_t_part, z_part, z_expected = model(
        x, edge_index, edge_attr, target_mp_node,
        fix_x_mask, fix_y_mask, join_pairs, thickness_gate=gate
    )

    ## ── 물리 손실: 비대칭 Huber (§5) — coords·t(=z·t_raw) 모두에 grad ──
    l_phys_terms = []
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
        pred_mp_sections.append(pred_mp_section.item())

    l_phys_total = torch.stack(l_phys_terms).mean()
    pred_mp_sections = np.array(pred_mp_sections)
    mp_rel_err = float(np.abs(np.sum(pred_mp_sections) - sum(target_mps.values())) / sum(target_mps.values()))

    ## ── 커리큘럼 가중치 ──
    s_phys, s_smooth = 1.0, 1.0
    if curriculum:
        s_phys, s_smooth = get_curriculum_weights_v10(epoch, max_epochs, curriculum_ratio)

    ## ── 다목적 손실 계산 ──
    l_smooth     = compute_smoothness_loss_angle(new_coords, edge_index, edge_attr)
    area, l_area_over = compute_mass_loss_v11(new_coords, t_final, edge_index, edge_attr, target_area)
    l_collision  = compute_collision_loss_v11(new_coords, t_final, part_ids, section_ids,
                                              collision_spec, z_part)
    l_order  = compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr)
    l_anchor = compute_anchor_loss(new_coords, base_coords, fix_x_mask, fix_y_mask)
    l_sat    = compute_saturation_loss(delta_t_part, delta_scale=model.DELTA_SCALE)

    ## ── [v11] 희소화(경량화) 손실: λ0(epoch) · mean(E[z]) ──
    lam0 = lambda0_value(epoch)
    # Mp 실행가능(오차<5%)일 때만 희소화 압력 인가 → Mp 붕괴 방지(연속 재분배 여지)
    feas_gate = float(torch.sigmoid(torch.tensor(10.0 * (0.05 - mp_rel_err))).item())
    l_sparsity = z_expected.mean()

    ## ── mass over-area 게이트: 두께 게이트 × Mp 오차 5% 연속 게이트 (v10 유지) ──
    mass_gate = gate * feas_gate

    ## ── 가중치 적용 후 항별 기여도 ──
    contrib_phys      = weights['w_phys']      * l_phys_total * s_phys
    contrib_smooth    = weights['w_smooth']    * l_smooth     * s_smooth
    contrib_mass      = weights['w_mass']      * l_area_over  * mass_gate
    contrib_collision = weights['w_collision'] * l_collision
    contrib_order     = weights['w_order']     * l_order
    contrib_anchor    = weights['w_anchor']    * l_anchor
    contrib_sat       = weights['w_sat']       * l_sat
    contrib_sparsity  = lam0 * feas_gate * l_sparsity          # [v11]

    loss = (contrib_phys + contrib_smooth + contrib_mass
            + contrib_collision + contrib_order + contrib_anchor + contrib_sat
            + contrib_sparsity)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    # [v11 가드] 보호 파트는 항상 살아있어야 함(전체 붕괴/Mp=0 방지)
    with torch.no_grad():
        prot = model.part_gate.protected_mask
        if prot.any():
            assert z_part.detach()[prot].min().item() > 0.99, \
                "[guard] 보호 파트 z가 1에서 이탈 — 게이트 고정 로직 확인 필요"

    # 포화/게이트 모니터링
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

        z_np = z_part.detach().cpu().numpy()
        pruned_parts = int((z_np < PRUNE_TAU).sum())

    return {
        "loss":          loss.item(),
        "pred_mp":       pred_mp_sections,
        "mp_rel_err":    mp_rel_err,
        "l_phys":        l_phys_total.item(),
        "l_smooth":      l_smooth.item(),
        "area":          area.item(),
        "l_mass":        l_area_over.item(),
        "mass_gate":     mass_gate,
        "l_collision":   l_collision.item(),
        "l_order":       l_order.item(),
        "l_anchor":      l_anchor.item(),
        "l_sat":         l_sat.item(),
        "l_sparsity":    l_sparsity.item(),
        "lambda0":       lam0,
        "new_coords":    new_coords.detach(),
        "thickness_gate": gate,
        "delta_t_mean":  delta_t_part.mean().item(),
        "saturated_parts": saturated_parts,
        "t_per_part":    t_per_part,
        "z_part":        z_np,
        "pruned_parts":  pruned_parts,
    }


def run_training(data, target_mps, target_area,
                  max_epochs=300, lr=1e-3, weights=None, curriculum=True,
                  curriculum_ratio=(0.2, 0.7), snapshot_interval=10,
                  feasibility_mp_err=0.02, feasibility_collision=0.05):
    """
    v10 run_training 대비 변경점 (v11):
    - forward 반환 언패킹에 z_part/z_expected 반영(모두 train_step 내부 처리)
    - z_part 별도 optimizer 그룹 불필요(part_gate.log_alpha는 main_params에 포함)
    - 로깅에 z_part, pruned_parts, l_sparsity, lambda0 추가
    - 최종 하드컷(z<τ) 요약 출력
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data   = data.to(device)

    model = CGDN(
        in_channels=8,
        hidden_channels=128,
        num_layers=4,
        heads=4,
        edge_dim=4,
        max_displacement=50.0,
        n_parts=5,
        protect_indices=(0, 1),   # Outer Hat(#00), Inner Plate(#03)
    ).to(device)

    # [§4/D4] 파라미터 그룹 분리: thickness_decoder만 별도 그룹 (part_gate는 main 그룹)
    thick_param_ids = {id(p) for p in model.thickness_decoder.parameters()}
    main_params  = [p for p in model.parameters() if id(p) not in thick_param_ids]
    thick_params = list(model.thickness_decoder.parameters())
    assert len(main_params) + len(thick_params) == len(list(model.parameters()))
    optimizer = optim.AdamW(
        [{'params': main_params,  'name': 'main'},
         {'params': thick_params, 'name': 'thickness_decoder'}],
        lr=lr, weight_decay=1e-4)

    if weights is None:
        weights = {
            'w_phys':      10.0,
            'w_collision':  5.0,
            'w_order':      1.0,
            'w_mass':       2.0,
            'w_smooth':     0.5,
            'w_anchor':     0.02,
            'w_sat':        0.01,
        }

    x = data.x
    part_labels_t = x[:, 4].cpu().long()
    edge_index_cpu = data.edge_index.cpu()
    base_coords = x[:, :2].detach().cpu()
    base_t_cpu  = x[:, 6:7].cpu()
    fy_full     = x[:, 7:8].cpu()
    part_ids    = x[:, 4]
    section_ids = x[:, 5]

    # ── [v8 §3.1 유지] 학습 전 ∂Mp/∂t 유한차분 검증 게이트 (raw 두께, 게이트 무관) ──
    verify_thickness_gradient(x[:, :2], x[:, 6:7], x[:, 7:8], data.edge_index)

    # ── target_area: 초기 단면적 스냅샷 ──
    if target_area is None:
        target_area, _ = compute_section_area(x[:, :2].cpu(), base_t_cpu, edge_index_cpu)
    print(f"[mass] target_area = {target_area:.1f} mm² (초기 단면적 스냅샷, one-sided 상한)")

    # ── [§2] Collision v5 spec ──
    collision_spec = build_collision_spec(x[:, :2], x[:, 6:7], part_ids, section_ids)
    print(f"[collision v5] 전쌍({len(collision_spec)}쌍) 부호 앵커 & clearance (초기 형상 기준):")
    for (sec, a, b), dirs in collision_spec.items():
        info = " | ".join(f"{d['seg_part']}seg/{d['pt_part']}pt σ={d['sigma']:+.0f} clr={d['clearance']:.2f}"
                          for d in dirs)
        print(f"    sec{sec} Part{a}-Part{b}: {info}")

    prot = model.part_gate.protected_mask.cpu().numpy()
    prot_ids = [i for i in range(5) if prot[i]]
    prune_ids = [i for i in range(5) if not prot[i]]
    print(f"[prune] 보호 파트(z=1 고정): {prot_ids} | 프루닝 후보: {prune_ids}")

    history = {
        'loss': [], 'pred_mp': [], 'mp_rel_err': [], 'l_phys': [], 'l_smooth': [],
        'area': [], 'l_mass': [], 'mass_gate': [], 'l_collision': [], 'l_order': [],
        'l_anchor': [], 'l_sat': [], 'l_sparsity': [], 'lambda0': [],
        'thickness_gate': [], 'delta_t_mean': [], 'saturated_parts': [],
        'pruned_parts': [], 'z_part': [], 'snapshots': [], 't_per_part': [],
    }

    best_feasible = {
        'found': False, 'epoch': None, 'mp_rel_err': None,
        'l_collision': None, 'state_dict': None, 'z_part': None, 'area': None,
    }
    best_mp = {
        'epoch': None, 'mp_rel_err': float('inf'),
        'l_collision': None, 'state_dict': None,
    }
    first_feasible_epoch = None
    stage2_done = False

    print(f"\n{'=' * 90}")
    print(f"[ uni_section_v11 ] Training  |  Target Mp = {target_mps[0]:,.0f} N·mm  |  Epochs: {max_epochs}")
    print(f"  CGDN: hidden=128, layers=4, heads=4  |  Curriculum: {curriculum} {curriculum_ratio}")
    print(f"  (v10 collision v5 / mesh order / 2-stage gate@{STAGE2_EPOCH} / asym Huber)")
    print(f"  [v11] 파트 프루닝: HardConcrete gate + one-sided mass + λ0 어닐링(max={LAMBDA0_MAX}, warm={LAMBDA0_WARM})")
    print(f"  T_MAX={CGDN.T_MAX}mm | DELTA_SCALE={CGDN.DELTA_SCALE} | w_order={weights['w_order']} | grad_clip=5.0")
    print(f"  Feasibility 기준: Mp err < {feasibility_mp_err*100:.1f}%  AND  l_collision < {feasibility_collision}")
    print(f"{'=' * 90}")
    print(f"Epoch ||  Loss  ||  MpErr% |  Area  | Coll  | Order || tGate | λ0 | E[z]  | z(parts)             | pruned")

    new_coords = None
    for epoch in range(max_epochs):
        # [§4/D4] Stage 2 진입: thickness 그룹만 optimizer state 국소 리셋 + lr 0.3×
        if (not stage2_done) and epoch == STAGE2_EPOCH:
            for group in optimizer.param_groups:
                if group.get('name') == 'thickness_decoder':
                    for p in group['params']:
                        optimizer.state.pop(p, None)
                    group['lr'] = 0.3 * lr
            stage2_done = True
            print(f"[Stage 2] epoch {epoch}: thickness_decoder 그룹 AdamW state 리셋, lr → {0.3*lr:.1e} "
                  f"| λ0 어닐링 시작(부재 프루닝 압력 인가)")

        info = train_step(model, data, optimizer, target_mps, target_area,
                           epoch, max_epochs, weights, curriculum,
                           curriculum_ratio, collision_spec)

        for key in ('loss', 'pred_mp', 'mp_rel_err', 'l_phys', 'l_smooth', 'area', 'l_mass',
                    'mass_gate', 'l_collision', 'l_order', 'l_anchor', 'l_sat', 'l_sparsity',
                    'lambda0', 'thickness_gate', 'delta_t_mean', 'saturated_parts', 'pruned_parts'):
            history[key].append(info[key])
        history['t_per_part'].append(info['t_per_part'])
        history['z_part'].append(info['z_part'])
        new_coords = info['new_coords']

        is_feasible = (info['mp_rel_err'] < feasibility_mp_err) and (info['l_collision'] < feasibility_collision)
        if is_feasible:
            if first_feasible_epoch is None:
                first_feasible_epoch = epoch
            # [v11] feasible 중에서는 '가장 가벼운(면적 최소)' 해를 선호
            better = (not best_feasible['found']) or (info['area'] < (best_feasible['area'] or float('inf')))
            if better:
                best_feasible['found']       = True
                best_feasible['epoch']       = epoch
                best_feasible['mp_rel_err']  = info['mp_rel_err']
                best_feasible['l_collision'] = info['l_collision']
                best_feasible['area']        = info['area']
                best_feasible['z_part']      = info['z_part']
                best_feasible['state_dict']  = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if info['mp_rel_err'] < best_mp['mp_rel_err']:
            best_mp['epoch']       = epoch
            best_mp['mp_rel_err']  = info['mp_rel_err']
            best_mp['l_collision'] = info['l_collision']
            best_mp['state_dict']  = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch <= 10 or (epoch - 10) % 20 == 0:
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
            z_str = "[" + " ".join(f"{v:.2f}" for v in info['z_part']) + "]"
            print(f"{epoch:05d} || {info['loss']:.4f} || {info['mp_rel_err']*100:6.2f}% | "
                  f"{info['area']:6.1f} | {info['l_collision']:.4f} | {info['l_order']:.4f} || "
                  f"{info['thickness_gate']:.2f} | {info['lambda0']:.2f} | {info['l_sparsity']:.3f} | "
                  f"{z_str} | {info['pruned_parts']}{flag}")

    final_new_coords = new_coords.detach().cpu() if new_coords is not None else base_coords

    print(f"\n{'─' * 90}")
    if best_feasible['found']:
        z_str = "[" + " ".join(f"{v:.2f}" for v in best_feasible['z_part']) + "]"
        hard = [i for i, v in enumerate(best_feasible['z_part']) if v < PRUNE_TAU]
        print(f"[Feasibility] 첫 만족 epoch: {first_feasible_epoch}  |  "
              f"최경량 feasible epoch: {best_feasible['epoch']}  |  "
              f"Mp err: {best_feasible['mp_rel_err']*100:.2f}%  |  "
              f"area: {best_feasible['area']:.1f} mm²  |  l_collision: {best_feasible['l_collision']:.4f}")
        print(f"[Prune] best-feasible z={z_str}  →  하드컷(z<{PRUNE_TAU}) 제거 부재: {hard if hard else '없음'}")
    else:
        print(f"[Feasibility] 학습 전체(max_epochs={max_epochs}) 동안 "
              f"'Mp err < {feasibility_mp_err*100:.1f}% AND l_collision < {feasibility_collision}' "
              f"조건을 동시에 만족한 epoch이 없음.")
    print(f"[Best-Mp] epoch {best_mp['epoch']}: Mp err = {best_mp['mp_rel_err']*100:.2f}%, "
          f"l_collision = {best_mp['l_collision']:.4f}  (collision 무관 추적)")
    print(f"{'─' * 90}")

    return history, base_coords, final_new_coords, part_labels_t, best_feasible, best_mp


# ══════════════════════════════════════════════════════════════════
# SECTION 5: 시각화
# ══════════════════════════════════════════════════════════════════

def visualize_training(history, base_coords, result_coords, target_mp_val, part_labels=None,
                        best_feasible=None):
    fig, axes = plt.subplots(2, 4, figsize=(26, 9))
    axes = axes.flatten()
    epochs = list(range(len(history['loss'])))

    ax = axes[0]
    ax.plot(epochs, history['loss'], color='#2196F3', linewidth=1.2, label='Total Loss')
    ax.axvline(STAGE2_EPOCH, color='gray', linestyle='--', linewidth=1.0, label=f'Stage 2 ({STAGE2_EPOCH})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Total Loss 수렴', fontweight='bold'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_yscale('log')

    ax = axes[1]
    base_np = base_coords.numpy(); result_np = result_coords.numpy()
    part_colors = {0: '#FF5722', 1: '#FFAA00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}
    part_names  = {0: '#00(Outer)', 1: '#03(Plate)', 2: '#06(Inner)', 3: '#07(Patch1)', 4: '#08(Patch2)'}
    pl = part_labels.numpy() if part_labels is not None else None
    z_final = history['z_part'][-1] if history['z_part'] else np.ones(5)
    for part_id in range(5):
        mask = (pl == part_id) if pl is not None else slice(None)
        c = part_colors[part_id]; name = part_names[part_id]
        pruned = z_final[part_id] < PRUNE_TAU
        alpha_r = 0.15 if pruned else 1.0
        tag = " [PRUNED]" if pruned else ""
        ax.plot(base_np[mask, 0], base_np[mask, 1], 'o--', color=c, alpha=0.30, linewidth=1.2)
        ax.plot(result_np[mask, 0], result_np[mask, 1], 's-', color=c, alpha=alpha_r, linewidth=1.8,
                label=f'{name} (z={z_final[part_id]:.2f}){tag}')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title('단면 형상: Base vs Result (프루닝 반영)', fontweight='bold')
    ax.legend(loc='best', fontsize=6.5, ncol=1); ax.grid(True, alpha=0.3)

    ax = axes[2]
    pred_mp_total = [float(np.sum(v)) for v in history['pred_mp']]
    ax.plot(epochs, [v / 1e6 for v in pred_mp_total], color='#2196F3', linewidth=1.2, label='Pred Mp')
    ax.axhline(target_mp_val / 1e6, color='#FF5722', linestyle=':', linewidth=2.0, label='Target Mp')
    if best_feasible is not None and best_feasible.get('found'):
        ax.axvline(best_feasible['epoch'], color='#4CAF50', linestyle='--', linewidth=1.5,
                   label=f"Best feasible (epoch {best_feasible['epoch']})")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mp (MN·mm)')
    ax.set_title('Mp 수렴', fontweight='bold'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(epochs, history['l_smooth'], label='Smooth', linewidth=1.0)
    ax.plot(epochs, history['l_mass'], label='Mass(over)', linewidth=1.0)
    ax.plot(epochs, history['l_collision'], label='Collision', linewidth=1.0)
    ax.plot(epochs, history['l_order'], label='Order', linewidth=1.2, color='#E91E63')
    ax.plot(epochs, history['l_sat'], label='Sat', linewidth=1.0, linestyle='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss term')
    ax.set_title('보조 손실 항 추이', fontweight='bold'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[4]
    z_hist = np.array(history['z_part'])  # [E, 5]
    for part_id in range(5):
        ax.plot(epochs, z_hist[:, part_id], color=part_colors[part_id], linewidth=1.4,
                label=part_names[part_id])
    ax.axhline(PRUNE_TAU, color='red', linestyle=':', linewidth=1.0, label=f'τ={PRUNE_TAU}')
    ax.axvline(STAGE2_EPOCH, color='gray', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Epoch'); ax.set_ylabel('존재 게이트 z')
    ax.set_title('[v11] 파트 존재 게이트 z 추이 (프루닝)', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.1)

    ax = axes[5]
    ax.plot(epochs, [e * 100 for e in history['mp_rel_err']], color='#FF5722', linewidth=1.2, label='Mp rel err (%)')
    ax.axhline(2.0, color='gray', linestyle=':', linewidth=1.0, label='Feasibility 임계 (2%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mp 상대오차 (%)')
    ax.set_title('Mp 오차 추이', fontweight='bold'); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3); ax.set_yscale('log')

    ax = axes[6]
    ax.plot(epochs, history['area'], color='#009688', linewidth=1.4, label='Section Area (mm²)')
    ax.axvline(STAGE2_EPOCH, color='gray', linestyle='--', linewidth=1.0, label=f'Stage 2')
    ax2 = ax.twinx()
    ax2.plot(epochs, history['lambda0'], color='#795548', linewidth=1.0, linestyle='--', label='λ0')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Area (mm²)'); ax2.set_ylabel('λ0')
    ax.set_title('[v11] 단면적(경량화) & λ0 어닐링', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)

    ax = axes[7]
    ax.plot(epochs, history['pruned_parts'], color='#9C27B0', linewidth=1.6, drawstyle='steps-post',
            label='pruned parts (z<τ)')
    ax.axvline(STAGE2_EPOCH, color='gray', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Epoch'); ax.set_ylabel('제거된 부재 수')
    ax.set_title('[v11] 프루닝된 부재 수', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_ylim(-0.2, 3.2)

    plt.suptitle('uni_section_v11 학습 결과  |  v10 + 부재 프루닝(HardConcrete gate + one-sided mass + λ0)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    try:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'uni_section_v11_result.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"\n결과 저장: {out_path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("uni_section_v11: v10 + 부재 프루닝(위상 경량화) — docs/idea/idea_v0.md 반영")
    print("  - [A] HardConcrete 파트 존재 게이트 z (L0 relaxation), t_final = z·t_raw (T_MIN 바깥)")
    print("  - [B] 동시 최적화 + λ0 어닐링(Stage2 이후 희소화 압력 점증)")
    print("  - [C] mass loss 개조: one-sided over-area + λ0·mean(E[z]) 희소화 보상")
    print("  - [D] collision violation에 z_i·z_j 곱 (clearance 수축 버그 회피)")
    print("  - [E] 보호 파트 S_protect={0,1}=Outer Hat/Inner Plate z=1 고정")
    print("  - [F] 추론 하드컷: z<0.5 부재 제거 확정")

    data, node_registry = build_bpillar_section()
    print(f"\n데이터: nodes={data.x.shape} | edges={data.edge_index.shape}")

    TARGET_MP = 47_421_470  # N·mm (v10 그대로)
    target_mps = {0: TARGET_MP}

    weights = {
        'w_phys':      10.0,
        'w_collision':  5.0,
        'w_order':      1.0,
        'w_mass':       2.0,
        'w_smooth':     0.5,
        'w_anchor':     0.02,
        'w_sat':        0.01,
    }

    history, base_coords, result_coords, part_labels, best_feasible, best_mp = run_training(
        data,
        target_mps=target_mps,
        target_area=None,
        max_epochs=300,
        lr=1e-3,
        weights=weights,
        curriculum=True,
        curriculum_ratio=(0.2, 0.7),
        snapshot_interval=10,
        feasibility_mp_err=0.02,
        feasibility_collision=0.05,
    )

    visualize_training(history, base_coords, result_coords, TARGET_MP, part_labels=part_labels,
                       best_feasible=best_feasible)

    final_pred = float(np.sum(history['pred_mp'][-1]))
    final_err  = abs(final_pred - TARGET_MP) / TARGET_MP * 100
    z_final = history['z_part'][-1]
    pruned_final = [i for i, v in enumerate(z_final) if v < PRUNE_TAU]
    print(f"\n{'=' * 90}")
    print(f"최종 결과 요약 (마지막 epoch 기준)")
    print(f"  Target Mp     : {TARGET_MP:>14,.0f} N·mm")
    print(f"  Final pred_mp : {final_pred:>14,.0f} N·mm")
    print(f"  Final Error   : {final_err:>6.2f}%")
    print(f"  Final Area    : {history['area'][-1]:>10.1f} mm²  (초기 {history['area'][0]:.1f} 대비 "
          f"{(history['area'][-1]/history['area'][0]-1)*100:+.1f}%)")
    print(f"  Final l_collision : {history['l_collision'][-1]:.4f}")
    print(f"  Final z(parts): [" + " ".join(f"{v:.2f}" for v in z_final) + "]")
    print(f"  제거된 부재(z<{PRUNE_TAU}): {pruned_final if pruned_final else '없음'}")
    if best_feasible['found']:
        z_bf = best_feasible['z_part']
        hard = [i for i, v in enumerate(z_bf) if v < PRUNE_TAU]
        print(f"\n  ★ 최경량 Feasible 모델: epoch {best_feasible['epoch']}, "
              f"Mp err={best_feasible['mp_rel_err']*100:.2f}%, area={best_feasible['area']:.1f} mm², "
              f"l_collision={best_feasible['l_collision']:.4f}")
        print(f"    제거 부재: {hard if hard else '없음'}  |  "
              f"(best_feasible['state_dict']를 load_state_dict로 복원해 사용 권장)")
    else:
        print(f"\n  ⚠ Feasible 모델을 찾지 못함 — best-Mp 체크포인트(epoch {best_mp['epoch']}, "
              f"err {best_mp['mp_rel_err']*100:.2f}%) 참고. λ0_max 하향 또는 warmup 연장 검토.")
    print(f"{'=' * 90}")
