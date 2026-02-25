#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

class ImplicitPNASolver(torch.autograd.Function):
    """
    미분 가능한 소성 중립축(PNA) 및 전소성 모멘트(Mp) 계산기
    ─────────────────────────────────────────────────────
    Forward : Bisection Method로 인장력 = 압축력 평형점(y_pna) 탐색
    Backward: Implicit Function Theorem(IFT)으로 ∂y_pna/∂coords 계산 후
              chain-rule을 통해 ∂Mp/∂coords 를 GNN까지 전파
    """

    @staticmethod
    def forward(ctx, coords, t, fy, edge_index, n_iter=30):
        """
        coords    : [N, 2]  (x, y)
        t         : [N, 1]  두께
        fy        : [N, 1]  항복강도
        edge_index: [2, E]  (사용 안 하지만 인터페이스 유지)
        n_iter    : bisection 반복 횟수
        """
        # ── 1. Bisection으로 PNA 탐색 (no grad) ──
        with torch.no_grad():
            y = coords[:, 1]                       # [N]
            t_flat = t.squeeze(-1)                 # [N]
            fy_flat = fy.squeeze(-1)               # [N]

            y_lo = y.min().clone()
            y_hi = y.max().clone()

            for _ in range(n_iter):
                y_mid = 0.5 * (y_lo + y_hi)
                # 인장(위쪽) / 압축(아래쪽) 힘
                F_tens = torch.sum(t_flat * fy_flat * torch.clamp(y - y_mid, min=0.0))
                F_comp = torch.sum(t_flat * fy_flat * torch.clamp(y_mid - y, min=0.0))

                if F_tens > F_comp:
                    y_lo = y_mid
                else:
                    y_hi = y_mid

            y_pna = 0.5 * (y_lo + y_hi)           # scalar tensor

        # ── 2. 전소성 모멘트 Mp = Σ A_i · fy_i · |y_i − y_pna| ──
        d = torch.abs(coords[:, 1] - y_pna)       # [N]
        area = t_flat                              # 단위 길이당 면적
        mp_pred = torch.sum(area * fy_flat * d)    # scalar

        # backward 에 필요한 텐서 저장
        ctx.save_for_backward(coords, t, fy, y_pna.unsqueeze(0), edge_index)
        return mp_pred

    @staticmethod
    def backward(ctx, grad_output):
        """
        Implicit Function Theorem 적용
        ──────────────────────────────
        평형 조건  g(y_pna, coords) = F_tens − F_comp = 0

        IFT에 의해:
            ∂y_pna/∂y_i = − (∂g/∂y_i) / (∂g/∂y_pna)

        ∂Mp/∂y_i = A_i·fy_i·sign(y_i−y_pna)
                  + Σ_j A_j·fy_j·(−sign(y_j−y_pna)) · (∂y_pna/∂y_i)

        두 번째 항이 IFT 보정 항으로, PNA 이동에 의한 간접 효과를 반영합니다.
        """
        coords, t, fy, y_pna_buf, edge_index = ctx.saved_tensors
        y_pna = y_pna_buf.squeeze(0)   # scalar

        y = coords[:, 1]
        t_flat = t.squeeze(-1)
        fy_flat = fy.squeeze(-1)

        s = torch.sign(y - y_pna)      # +1(인장), −1(압축)

        # ── ∂g/∂y_pna  (평형식의 y_pna에 대한 미분) ──
        #  g = Σ t·fy·clamp(y−y_pna,0) − Σ t·fy·clamp(y_pna−y,0)
        #  ∂g/∂y_pna = −Σ t·fy · 1[y>y_pna] − Σ t·fy · 1[y<y_pna]
        #            = −Σ t·fy  (y_pna 정확히 위의 노드 무시)
        dg_dy_pna = -torch.sum(t_flat * fy_flat)   # scalar (항상 음수)

        # ── ∂g/∂y_i ──
        #  i번째 노드가 인장(y_i>y_pna) → ∂g/∂y_i = +t_i·fy_i
        #  i번째 노드가 압축(y_i<y_pna) → ∂g/∂y_i = −t_i·fy_i  ⟹  = s_i · t_i · fy_i
        dg_dy = s * t_flat * fy_flat               # [N]

        # ── IFT: ∂y_pna/∂y_i = −dg_dy / dg_dy_pna ──
        dy_pna_dy = -dg_dy / (dg_dy_pna + 1e-12)  # [N]

        # ── ∂Mp/∂y_i (직접 항 + IFT 보정 항) ──
        direct = t_flat * fy_flat * s                               # [N]
        indirect = -torch.sum(t_flat * fy_flat * s) * dy_pna_dy     # [N]
        dMp_dy = direct + indirect                                  # [N]

        grad_coords = torch.zeros_like(coords)
        grad_coords[:, 1] = grad_output * dMp_dy

        return grad_coords, None, None, None, None   # n_iter 추가로 5개


def calculate_mpl(coords, t, fy, edge_index):
    """Wrapper: ImplicitPNASolver.apply 호출"""
    return ImplicitPNASolver.apply(coords, t, fy, edge_index)


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm

# ═══════════════════════════════════════════════════════════
#  C-GDN (Constraint-aware Graph Deformation Network)
# ═══════════════════════════════════════════════════════════
#  - 표준 템플릿(Base Section)을 목표 Mp에 맞춰 **변형**하는 구조
#  - FiLM 레이어로 매 GATv2 블록에 Mp 조건 주입
#  - Hard constraint: 고정점 마스킹
# ═══════════════════════════════════════════════════════════

class FiLMGenerator(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)
    target_mp [B, 1] → (gamma, beta) [B, hidden]
    """
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, hidden_channels * 2),
        )

    def forward(self, target_mp):
        """target_mp: [1, 1] or [B, 1]"""
        out = self.net(target_mp)                       # [B, 2H]
        gamma, beta = torch.chunk(out, 2, dim=-1)       # [B, H] each
        return gamma, beta


class CGDNBlock(nn.Module):
    """
    단일 Message-Passing 블록
    GATv2Conv → FiLM modulation → LayerNorm → GELU → Residual
    """
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

        # FiLM conditioning: γ ⊙ h + β
        h = gamma * h + beta

        h = self.norm(h)
        h = F.gelu(h)
        h = h + h_res   # residual
        return h


class CGDN(nn.Module):
    """
    Constraint-aware Graph Deformation Network

    입력 노드 특징 (R^6):
        [x, y, is_fixed, layer_id, t, fy]

    엣지 특징 (R^4):
        [선분 길이, 각도, 레이어 ID, 플랜지 여부]

    조건부 입력:
        target_mp  (목표 전소성 모멘트)
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        edge_dim: int = 4,
        max_displacement: float = 50.0,     # 최대 변위 클리핑 (mm)
    ):
        super().__init__()
        self.max_displacement = max_displacement

        # 1. Node Encoder ─────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            LayerNorm(hidden_channels),
            nn.GELU(),
        )

        # 2. FiLM Generator (Mp 조건 주입) ────────
        self.film_gen = FiLMGenerator(hidden_channels)

        # 3. GATv2 Message-Passing Blocks ─────────
        self.blocks = nn.ModuleList([
            CGDNBlock(hidden_channels, heads=heads, edge_dim=edge_dim)
            for _ in range(num_layers)
        ])

        # 4. Coordinate Decoder ───────────────────
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.GELU(),
            nn.Linear(64, 2),               # 출력: (Δx, Δy)
        )

    def forward(self, x, edge_index, edge_attr, target_mp, is_fixed_mask):
        """
        Parameters
        ----------
        x             : [N, 6]   노드 특징
        edge_index    : [2, E]
        edge_attr     : [E, 4]   엣지 특징
        target_mp     : [1, 1]   목표 전소성 모멘트
        is_fixed_mask : [N, 1]   고정점 Boolean 마스크

        Returns
        -------
        new_coords    : [N, 2]   변형 후 좌표
        delta_coords  : [N, 2]   변위 벡터
        """
        # ── Encode ──
        h = self.node_encoder(x)

        # ── FiLM parameters (한 번 생성, 모든 블록에 공유) ──
        gamma, beta = self.film_gen(target_mp)          # [1, H]

        # ── Message Passing ──
        for block in self.blocks:
            h = block(h, edge_index, edge_attr, gamma, beta)

        # ── Decode displacement ──
        delta_coords = self.decoder(h)                  # [N, 2]

        # 변위 클리핑 (안정성)
        delta_coords = torch.clamp(delta_coords,
                                   -self.max_displacement,
                                    self.max_displacement)

        # ── Hard Constraint: 고정점은 변위 = 0 ──
        delta_coords = delta_coords * (~is_fixed_mask).float()

        # ── 최종 좌표 = Base 좌표 + Δ ──
        new_coords = x[:, :2] + delta_coords

        return new_coords, delta_coords


# In[3]:


import torch
import torch.optim as optim

# ═══════════════════════════════════════════════════════════
#  Physics-Informed Loss Functions
# ═══════════════════════════════════════════════════════════

def compute_smoothness_loss(new_coords, edge_index):
    """
    L_smooth – Laplacian Smoothing
    인접 노드 간의 변위 차이를 규제하여 매끄러운 곡선을 유도합니다.
    (CAD 서피스 생성이 용이한 형상)
    """
    src, dst = edge_index
    diff = new_coords[src] - new_coords[dst]
    return torch.mean(torch.norm(diff, dim=1) ** 2)


def compute_collision_loss(new_coords, layer_ids, margin=0.5):
    """
    L_collision – 레이어 간 간섭(자기교차) 방지 소프트 페널티
    Inner(layer_id=0)와 Outer(layer_id=1) 레이어가 서로 겹치거나
    최소 간격(margin) 이하로 가까워지는 것을 방지합니다.

    Parameters
    ----------
    new_coords : [N, 2]
    layer_ids  : [N]    (0=Inner, 1=Outer)
    margin     : float  최소 허용 간격 (mm)
    """
    inner_mask = (layer_ids == 0)
    outer_mask = (layer_ids == 1)

    if inner_mask.sum() == 0 or outer_mask.sum() == 0:
        return torch.tensor(0.0, device=new_coords.device)

    inner_y = new_coords[inner_mask, 1]   # [N_in]
    outer_y = new_coords[outer_mask, 1]   # [N_out]

    # Inner의 최대 y가 Outer의 최소 y보다 작아야 함 (+ margin)
    # 위반량 = max(0, inner_max_y − outer_min_y + margin)
    gap_violation = torch.clamp(inner_y.max() - outer_y.min() + margin, min=0.0)
    return gap_violation ** 2


def compute_mass_loss(new_coords, t, edge_index):
    """
    L_mass – 경량화 보조 손실
    단면적 = Σ (segment_length_i × t_i) 를 최소화하여
    Mp를 만족하는 여러 해 중 가장 가벼운 해를 선택하도록 유도합니다.
    """
    src, dst = edge_index
    seg_len = torch.norm(new_coords[src] - new_coords[dst], dim=1)   # [E]
    t_src = t[src].squeeze(-1)
    area = torch.sum(seg_len * t_src)
    return area


# ═══════════════════════════════════════════════════════════
#  Training Step
# ═══════════════════════════════════════════════════════════

def train_step(model, data, optimizer, target_mp_value,
               w_phys=1.0, w_smooth=0.1, w_mass=0.01,
               w_collision=1.0, w_fix=10.0):
    """
    L_total = w1·L_phys + w2·L_smooth + w3·L_mass + w4·L_collision + w5·L_fix
    """
    model.train()
    optimizer.zero_grad()

    # ── 데이터 추출 ──
    x          = data.x                          # [N, 6]
    edge_index = data.edge_index                 # [2, E]
    edge_attr  = data.edge_attr                  # [E, 4]

    is_fixed_mask = x[:, 2].bool().unsqueeze(1)  # [N, 1]
    layer_ids     = x[:, 3]                      # [N]  (0=Inner, 1=Outer)
    t             = x[:, 4].unsqueeze(1)         # [N, 1]
    fy            = x[:, 5].unsqueeze(1)         # [N, 1]

    target_mp = torch.tensor(
        [[target_mp_value]], dtype=torch.float32, device=x.device
    )

    # ── 1. Forward Pass (형상 변형 예측) ──
    new_coords, delta_coords = model(
        x, edge_index, edge_attr, target_mp, is_fixed_mask
    )

    # ── 2. Differentiable Mp 계산 (Implicit PNA Solver) ──
    pred_mp = calculate_mpl(new_coords, t, fy, edge_index)

    # ── 3. 다목적 Physics-Informed 손실 함수 ──

    # L_phys: 목표 Mp 대비 상대 오차 제곱
    l_phys = ((pred_mp - target_mp) / target_mp) ** 2

    # L_smooth: 형상 연속성 (Laplacian smoothing)
    l_smooth = compute_smoothness_loss(new_coords, edge_index)

    # L_mass: 경량화 (단면적 최소화)
    l_mass = compute_mass_loss(new_coords, t, edge_index)

    # L_collision: 레이어 간 간섭 방지
    l_collision = compute_collision_loss(new_coords, layer_ids)

    # L_fix: 고정점 위반 페널티
    fixed_nodes = is_fixed_mask.squeeze()       # [N]
    if fixed_nodes.any():
        l_fix = torch.sum(torch.norm(delta_coords[fixed_nodes], dim=1))
    else:
        l_fix = torch.tensor(0.0, device=x.device)

    # ── Total Loss ──
    loss = (w_phys     * l_phys
          + w_smooth   * l_smooth
          + w_mass     * l_mass
          + w_collision * l_collision
          + w_fix      * l_fix)

    # ── 4. Backward & Optimize ──
    loss.backward()
    optimizer.step()

    return {
        "loss":        loss.item(),
        "pred_mp":     pred_mp.item(),
        "l_phys":      l_phys.item(),
        "l_smooth":    l_smooth.item(),
        "l_mass":      l_mass.item(),
        "l_collision": l_collision.item(),
        "l_fix":       l_fix.item() if isinstance(l_fix, torch.Tensor) else l_fix,
        "new_coords":  new_coords.detach(),
    }


# ═══════════════════════════════════════════════════════════
#  실행 예시 (Dummy Data)
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 모델 초기화 ──
    model = CGDN(
        in_channels=6, hidden_channels=128,
        num_layers=4, heads=4, edge_dim=4,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ── Dummy PyG Data 생성 ──
    from torch_geometric.data import Data
    num_nodes = 50

    # 노드 특징: [x, y, is_fixed, layer_id, t, fy]
    dummy_x = torch.rand((num_nodes, 6)).to(device)
    dummy_x[:, 2] = torch.randint(0, 2, (num_nodes,)).float()   # 고정점 여부
    dummy_x[:, 3] = torch.randint(0, 2, (num_nodes,)).float()   # Inner(0) / Outer(1)
    dummy_x[:, 4] = 1.5       # 두께 1.5 mm
    dummy_x[:, 5] = 1500.0    # 항복강도 1500 MPa

    # 엣지: 인접 노드 연결 (체인 그래프 + 약간의 랜덤)
    src = torch.arange(0, num_nodes - 1)
    dst = torch.arange(1, num_nodes)
    extra_src = torch.randint(0, num_nodes, (50,))
    extra_dst = torch.randint(0, num_nodes, (50,))
    dummy_edge_index = torch.cat([
        torch.stack([src, dst]),
        torch.stack([dst, src]),
        torch.stack([extra_src, extra_dst]),
    ], dim=1).to(device)

    num_edges = dummy_edge_index.shape[1]
    # 엣지 특징: [길이, 각도, 레이어_ID, 플랜지_여부]
    dummy_edge_attr = torch.rand((num_edges, 4)).to(device)

    data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr)
    target_mp_req = 4500.0   # 목표 전소성 모멘트 (kN·mm)

    # ── Training Loop ──
    print("=" * 60)
    print("C-GDN Training (Dummy Data)")
    print("=" * 60)
    for epoch in range(100):
        info = train_step(model, data, optimizer, target_mp_req)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {info['loss']:.4f} | "
                f"Mp: {info['pred_mp']:.1f} / {target_mp_req:.1f} | "
                f"L_phys: {info['l_phys']:.4f}  "
                f"L_smooth: {info['l_smooth']:.4f}  "
                f"L_collision: {info['l_collision']:.4f}"
            )


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════
#  B-Pillar 단면 형상 시각화 (노드 & 엣지)
# ═══════════════════════════════════════════════════════════

def visualize_section(coords, edge_index, x_features, title="B-Pillar Cross Section",
                      deformed_coords=None, figsize=(14, 7)):
    """
    Parameters
    ----------
    coords         : [N, 2]  노드 좌표 (x, y)
    edge_index     : [2, E]  엣지 인덱스
    x_features     : [N, 6]  노드 특징 [x, y, is_fixed, layer_id, t, fy]
    deformed_coords: [N, 2]  변형 후 좌표 (선택)
    """
    is_fixed  = x_features[:, 2].cpu().numpy().astype(bool)
    layer_ids = x_features[:, 3].cpu().numpy().astype(int)
    coords_np = coords.cpu().detach().numpy()
    ei = edge_index.cpu().numpy()

    n_plots = 2 if deformed_coords is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    def _draw(ax, pts, subtitle):
        # ── 엣지 그리기 ──
        for i in range(ei.shape[1]):
            s, d = ei[0, i], ei[1, i]
            ax.plot([pts[s, 0], pts[d, 0]],
                    [pts[s, 1], pts[d, 1]],
                    color='#b0b0b0', linewidth=0.6, zorder=1)

        # ── 노드 그리기 (레이어별 색상, 고정/자유 마커) ──
        colors_map = {0: '#2196F3', 1: '#FF5722'}   # Inner=파랑, Outer=주황
        marker_map = {True: 's', False: 'o'}         # 고정=사각, 자유=원
        label_map  = {True: 'Fixed', False: 'Free'}
        layer_name = {0: 'Inner', 1: 'Outer'}

        for lid in [0, 1]:
            for fix in [True, False]:
                mask = (layer_ids == lid) & (is_fixed == fix)
                if not mask.any():
                    continue
                lbl = f'{layer_name[lid]} ({label_map[fix]})'
                ax.scatter(pts[mask, 0], pts[mask, 1],
                           c=colors_map[lid],
                           marker=marker_map[fix],
                           s=50, edgecolors='k', linewidths=0.5,
                           zorder=3, label=lbl)

        # ── 노드 인덱스 표시 ──
        for i in range(len(pts)):
            ax.annotate(str(i), (pts[i, 0], pts[i, 1]),
                        fontsize=6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points',
                        color='#555555')

        ax.set_title(subtitle, fontsize=13, fontweight='bold')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    # ── 원본(Base) 형상 ──
    _draw(axes[0], coords_np, f'{title} — Base Shape')

    # ── 변형(Deformed) 형상 ──
    if deformed_coords is not None:
        def_np = deformed_coords.cpu().detach().numpy()
        _draw(axes[1], def_np, f'{title} — Deformed Shape')

        # 변위 화살표 오버레이
        for i in range(len(coords_np)):
            dx = def_np[i, 0] - coords_np[i, 0]
            dy = def_np[i, 1] - coords_np[i, 1]
            if np.sqrt(dx**2 + dy**2) > 1e-4:
                axes[1].annotate('',
                    xy=(def_np[i, 0], def_np[i, 1]),
                    xytext=(coords_np[i, 0], coords_np[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.0, alpha=0.6))

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════
#  시각화 실행
# ═══════════════════════════════════════════════════════════
base_coords = data.x[:, :2]              # 초기 좌표
deformed    = info['new_coords']         # 학습 후 변형 좌표

visualize_section(
    coords=base_coords,
    edge_index=data.edge_index,
    x_features=data.x,
    title='B-Pillar Cross Section',
    deformed_coords=deformed,
)

