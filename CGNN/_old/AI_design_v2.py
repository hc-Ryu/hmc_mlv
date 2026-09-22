#!/usr/bin/env python
# coding: utf-8

"""
AI_design_v2.py
─────────────────────────────────────
docs/command/command_v1.md 실행 (2026-07-28 순차/윈도우 정정 반영). `uni-section/code/uni_section_v18.py`
(단일 섹션)를 **k=17(최상단/원본)부터 k=1(최하단)까지 순차적으로** 하나씩 학습한다. `AI_design_v1.py`
(17섹션을 한 그래프로 합쳐 동시 학습하는 joint 방식)는 idea_v0.md 요구사항 1·2("위→아래 순차 생성")를
위반해 폐기됐다 — 이 파일이 그 대체다.

핵심 구조 (command_v1.md §1~§5):
  - 바깥 루프: k=17..1. 매 스텝 **새로 학습되는 건 섹션 k 하나뿐**이다. k+1, k+2(이미 확정된 위쪽
    섹션)는 detach된 상수 좌표로 continuity loss의 타깃 역할만 한다 — 다중 모델을 동시에 학습하는
    "윈도우"가 아니라, "직전 최대 2개 섹션의 결과를 계속 참조하는 순차 학습"이다.
  - k=17 학습이 끝나면 연속 파트(S_PROTECT=[0,1,2]) 두께를 그 시점 평균값으로 영구 고정하고, k=16..1은
    그 값을 그대로 쓰며 학습 대상에서 제외한다(요구사항 4). `CGDNSeq.forward()`가 원본
    `uni_section_v18.CGDN.forward()`를 그대로 호출한 뒤, 연속 파트에 해당하는 `t_final`만 고정값으로
    덮어써 그래디언트를 끊는다 — log_alpha(게이트 파라미터)와는 무관하다(두께는 thickness_decoder의
    출력이지 log_alpha가 아니다 — /synod design 세션에서 Gemini/OpenAI 둘 다 이 둘을 혼동했음을
    실제 소스 대조로 확인 후 수정).
  - candidate 파트(CANDIDATE_PARTS=[3,4])는 각 섹션이 원본 그대로의 독립 `log_alpha`(shape (5,))로
    학습한다 — 섹션마다 완전히 새로 인스턴스화되므로 사전 (17,2) 확장이 필요 없다.
  - 인접 섹션 연속성: 학습 중인 k의 좌표를, 이미 확정된 k+1(더 큰 가중치)·k+2(더 작은 가중치) 좌표와
    비교하는 hinge loss. `build_bpillar_section()`은 결정적(같은 토폴로지)이라 노드 인덱스가 k에
    관계없이 항상 같은 파트/위치를 가리키므로, 최근접점 탐색(cdist) 없이 **인덱스 대응 비교**로
    충분하다(더 저렴하고 정확).
  - 원본 `train_step()`은 내부에서 직접 `loss.backward()`를 호출하므로(loss.backward()를 wrapper
    바깥에서 나중에 할 수 없음), continuity loss를 backward 이전에 더하려면 물리 loss 계산 부분을
    복제하는 것이 불가피하다(이전 세션들에서 반복 확인된 사실) — `sequential_train_step()`이 그 역할.

/synod design 세션: Gemini(pro→flash, conf 95), OpenAI(o3, conf 77) — 둘 다 "S_PROTECT 두께를
log_alpha 슬라이스로 고정"하는 안을 제시했으나 실제로는 log_alpha가 게이트(존재 여부)만 제어하고
두께와 무관함을 실제 forward() 소스(line 354-390)로 재확인, t_final 직접 override 방식으로 수정.
"""

import os
import math

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Gulim'

from torch_geometric.data import Data

import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, os.path.join(_THIS_DIR, 'uni-section', 'code'))
import uni_section_v18 as usv18


# ══════════════════════════════════════════════════════════════════
# SECTION 0: 상수 / 인덱싱 규약
# ══════════════════════════════════════════════════════════════════

TOP_K, BOT_K = 17, 1                       # idea_v0.md 1-indexed: k=17 최상단(원본), k=1 최하단
CONTINUOUS_PARTS = usv18.S_PROTECT         # [0, 1, 2] Outer Hat, Inner Plate, Inner Hat
CANDIDATE_PARTS = usv18.CANDIDATE_PARTS    # [3, 4] Patch 1, Patch 2

N_EPOCH_PER_SECTION = 1500                 # 섹션당 학습 epoch. 원본 uni_section_v18.py __main__(line
                                            # 1774)이 단일 섹션에 max_epochs=1500을 쓰고, 이번 순차
                                            # 설계의 각 스텝도 물리/커리큘럼 상수(STAGE2_EPOCH=10,
                                            # GATE_ACTIVE_EPOCH=11, pruning confirm_epochs=20)가 동일한
                                            # "진짜 단일 섹션 학습"이므로 그대로 맞춘다(command_v1.md
                                            # §6.4: 실측 후 조정 — 20-epoch 스모크 테스트는 검증용으로만
                                            # 사용했고, 실제 결과를 보려면 이 정도 길이가 필요).
W_CONT_K1 = 1.0                            # k+1(직접 인접)과의 continuity 가중치
W_CONT_K2 = 0.3                            # k+2(한 단계 더 위)와의 continuity 가중치, 더 완만하게

PART_COLORS = {0: '#FF5722', 1: '#FFAA00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}
PART_NAMES  = {0: '#00(Outer)', 1: '#03(Plate)', 2: '#06(Inner)', 3: '#07(Patch1)', 4: '#08(Patch2)'}


def verify_thickness_gradient_relaxed(coords, t, fy, edge_index, eps=1e-4,
                                       warn_thresh=1e-3, fail_thresh=5e-3):
    """uni_section_v18.verify_thickness_gradient()(line 105-130)의 완화 버전 — 원본은 수정하지
    않는다(command_v1.md 제약). 원본은 rel_err > 1e-3이면 무조건 학습 중단하는데, 순차 학습이
    만드는 새 두께 조합(예: k=17에서 고정된 연속 파트 두께)에서 compute_edge_mp_pna의 envelope
    theorem 근사 오차가 1e-3~5e-3 사이로 커지는 경우가 실측으로 확인됨(bisection n_iter를
    50->5000으로 올려도 값이 동일 — 정밀도 문제가 아니라 PNA 위치가 두께에 따라 움직이는 2차 효과를
    autograd가 무시하는 근사 자체의 한계). rel_err > fail_thresh(5e-3)에서만 실제 중단하고,
    1e-3~5e-3 구간은 경고만 남기고 계속 진행한다(사용자 확인 후 채택된 정책)."""
    coords = coords.detach()
    fy = fy.detach()

    t_leaf = t.detach().clone().requires_grad_(True)
    mp, _ = usv18.compute_edge_mp_pna(coords, t_leaf, fy, edge_index)
    mp.backward()
    grad_sum = t_leaf.grad.sum().item()

    with torch.no_grad():
        mp_p, _ = usv18.compute_edge_mp_pna(coords, t + eps, fy, edge_index)
        mp_m, _ = usv18.compute_edge_mp_pna(coords, t - eps, fy, edge_index)
    fd = (mp_p.item() - mp_m.item()) / (2.0 * eps)

    rel_err = abs(fd - grad_sum) / (abs(fd) + 1e-8)
    if fd <= 0:
        raise RuntimeError(f"[gradcheck] dMp/dt = {fd:.4e} <= 0 -- 물리적으로 비정상!")
    if rel_err > fail_thresh:
        raise RuntimeError(f"[gradcheck] autograd({grad_sum:.6e}) vs FD({fd:.6e}) "
                           f"상대오차 {rel_err:.2e} > {fail_thresh:.0e} (완화된 임계값도 초과)")
    if rel_err > warn_thresh:
        print(f"[gradcheck][WARNING] autograd={grad_sum:.4e} vs FD={fd:.4e} "
              f"rel_err={rel_err:.2e} > {warn_thresh:.0e} (완화 임계값 {fail_thresh:.0e} 이내라 계속 진행)")
    else:
        print(f"[gradcheck] dMp/dt OK -- autograd={grad_sum:.4e}, FD={fd:.4e}, rel_err={rel_err:.2e}")
    return grad_sum, fd, rel_err


def scale_factor(k: int) -> float:
    """initial_section.md §1: s(k) = 1.6 - 0.6*(k-1)/16, k=1..17 (k=17 -> s=1.0 원본/최상단,
    k=1 -> s=1.6 최하단)."""
    return 1.6 - 0.6 * (k - 1) / 16.0


# ══════════════════════════════════════════════════════════════════
# SECTION 1: 섹션 데이터 생성 (command_v1.md §2.1)
# ══════════════════════════════════════════════════════════════════

def build_section_data(k: int, frozen_thickness_continuous=None) -> Data:
    """build_bpillar_section()(uni_section_v18.py line 805)을 그대로 호출해 s(k) 스케일링만 적용.
    frozen_thickness_continuous가 주어지면(k < TOP_K) 연속 파트의 t_val(컬럼 6) 초기값도 그 값으로
    맞춰준다 — 실제 불변 보장은 CGDNSeq.forward()의 t_final override가 담당(§2)."""
    base_data, _ = usv18.build_bpillar_section()
    x = base_data.x.clone()
    s = scale_factor(k)
    x[:, 0] *= s
    x[:, 1] *= s
    x[:, 5] = 0.0   # section_id: 단일 섹션이므로 항상 0 (원본 build_bpillar_section 기본값 유지)

    if frozen_thickness_continuous is not None:
        part_ids = x[:, 4].long()
        for pid, val in frozen_thickness_continuous.items():
            x[part_ids == pid, 6] = float(val)

    return Data(x=x, edge_index=base_data.edge_index, edge_attr=base_data.edge_attr,
                join_pairs=torch.zeros((0, 2), dtype=torch.long))


# ══════════════════════════════════════════════════════════════════
# SECTION 2: CGDNSeq — 연속 파트 두께 override (command_v1.md §3.2)
# ══════════════════════════════════════════════════════════════════

class CGDNSeq(usv18.CGDN):
    """원본 CGDN.forward()를 그대로 호출한 뒤, frozen_continuous_thickness가 설정돼 있으면 연속
    파트(S_PROTECT)에 해당하는 t_final만 고정값으로 덮어써 그래디언트를 끊는다. log_alpha(게이트)는
    건드리지 않는다 — 두께(thickness_decoder 출력)와 게이트(log_alpha)는 서로 무관한 파라미터다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_continuous_thickness = None   # {part_id: float} 또는 None

    def forward(self, x, edge_index, edge_attr, target_mp,
                fix_x_mask, fix_y_mask, join_pairs=None, thickness_gate=1.0,
                gate_active=True, temperature=0.5):
        new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open = super().forward(
            x, edge_index, edge_attr, target_mp, fix_x_mask, fix_y_mask, join_pairs,
            thickness_gate=thickness_gate, gate_active=gate_active, temperature=temperature)

        if self.frozen_continuous_thickness is not None:
            part_ids = x[:, 4].long()
            t_final = t_final.clone()
            for pid, val in self.frozen_continuous_thickness.items():
                t_final[part_ids == pid] = val   # 상수 대입 -> 이 노드들은 autograd에서 끊김

        return new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open


# ══════════════════════════════════════════════════════════════════
# SECTION 3: 인접 섹션 연속성 loss (command_v1.md §4 — 인덱스 대응 비교)
# ══════════════════════════════════════════════════════════════════

def compute_continuity_loss(coords_k, frozen_coords_ref, scale_k, scale_ref, threshold=2.0):
    """build_bpillar_section()은 결정적 토폴로지이므로 노드 인덱스가 k와 무관하게 항상 같은
    파트/위치를 가리킨다 — 최근접점 탐색 없이 인덱스 대응 거리로 충분(더 저렴하고 정확).
    threshold(mm) 이내 변화는 무패널티(요구사항 6 '과도한 제약 금지')."""
    if frozen_coords_ref is None:
        return torch.tensor(0.0, device=coords_k.device)
    ck = coords_k / scale_k
    cf = frozen_coords_ref / scale_ref
    dist = torch.norm(ck - cf, dim=1)
    violation = torch.clamp(dist - threshold, min=0.0)
    return torch.mean(violation ** 2)


# ══════════════════════════════════════════════════════════════════
# SECTION 4: 섹션 단위 학습 (command_v1.md §2.2 — 원본 train_step 물리 loss 재사용)
# ══════════════════════════════════════════════════════════════════

def sequential_train_step(model, data, optimizer, target_mp_k, target_area_k,
                           epoch, max_epochs, weights, curriculum, curriculum_ratio,
                           collision_spec, alda_state, pruning_state,
                           frozen_coords_k1, frozen_coords_k2, scale_k, scale_k1, scale_k2):
    """uni_section_v18.train_step()(line 957-1159)의 단일 섹션 물리 loss 계산을 그대로 재사용하되,
    continuity loss(k+1, k+2)를 backward() 이전에 추가한다. section_id가 항상 0인 단일 섹션이므로
    원본의 unique_sections 순회 로직은 실질적으로 한 번만 돈다 — 그 부분은 그대로 둔다."""
    model.train()
    optimizer.zero_grad()

    x          = data.x
    edge_index = data.edge_index
    edge_attr  = data.edge_attr
    join_pairs = data.join_pairs
    base_coords = x[:, :2].detach()

    fix_x_mask = x[:, 2].bool().unsqueeze(1)
    fix_y_mask = x[:, 3].bool().unsqueeze(1)
    part_ids   = x[:, 4]
    section_ids = x[:, 5]
    fy         = x[:, 7].unsqueeze(1)

    target_mp_node = torch.full((x.shape[0], 1), float(target_mp_k), device=x.device)

    gate = usv18.thickness_gate_value(epoch)
    gate_active = epoch >= usv18.GATE_ACTIVE_EPOCH
    current_temp = usv18.compute_gate_temperature(epoch)

    new_coords, delta_coords, t_final, delta_t_part, z_gate, z_open = model(
        x, edge_index, edge_attr, target_mp_node,
        fix_x_mask, fix_y_mask, join_pairs, thickness_gate=gate,
        gate_active=gate_active, temperature=current_temp,
    )

    pred_mp = usv18.calculate_mpl(new_coords, t_final, fy, edge_index[:, edge_attr[:, 3] == 0.0])
    l_phys = usv18.asymmetric_huber_phys(pred_mp, torch.tensor(float(target_mp_k), device=x.device))
    mp_rel_err = float(torch.abs(pred_mp.detach() - target_mp_k) / abs(target_mp_k))

    if pruning_state is not None:
        newly_deleted = usv18.update_pruning_state(pruning_state, z_open.detach().cpu().numpy())
    current_tau_gate = usv18.TAU_GATE
    gate_multiplier = float(torch.sigmoid(torch.tensor(usv18.SPARSE_K * (current_tau_gate - mp_rel_err))).item())
    w_sparse_effective = weights['w_sparse'] * gate_multiplier
    l_sparse = z_open[usv18.CANDIDATE_PARTS].mean()
    contrib_sparse = w_sparse_effective * l_sparse

    contrib_entropy = torch.tensor(0.0, device=x.device)
    if gate_active:
        z_cand = z_open[usv18.CANDIDATE_PARTS]
        entropy = -torch.mean(z_cand * torch.log(z_cand + usv18.ENTROPY_EPS)
                               + (1.0 - z_cand) * torch.log(1.0 - z_cand + usv18.ENTROPY_EPS))
        contrib_entropy = (usv18.ALPHA_ENT * gate_multiplier) * entropy

    s_phys, s_smooth = 1.0, 1.0
    if curriculum:
        s_phys, s_smooth = usv18.get_curriculum_weights_v10(epoch, max_epochs, curriculum_ratio)

    l_smooth = usv18.compute_smoothness_loss_angle(new_coords, edge_index, edge_attr)
    area, l_mass = usv18.compute_mass_loss(new_coords, t_final, edge_index, edge_attr, target_area_k)
    l_collision = usv18.compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids,
                                                   collision_spec, z_gate=z_gate)
    l_order = usv18.compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr)
    l_anchor = usv18.compute_anchor_loss(new_coords, base_coords, fix_x_mask, fix_y_mask)
    l_sat = usv18.compute_saturation_loss(delta_t_part, delta_scale=model.DELTA_SCALE)

    l_cont1 = compute_continuity_loss(new_coords, frozen_coords_k1, scale_k, scale_k1)
    l_cont2 = compute_continuity_loss(new_coords, frozen_coords_k2, scale_k, scale_k2)
    contrib_continuity = W_CONT_K1 * l_cont1 + W_CONT_K2 * l_cont2

    L_alda, g_Mp_val, g_col_val = usv18.compute_alda_loss(
        area, target_area_k, pred_mp, torch.tensor(float(target_mp_k), device=x.device),
        l_collision, alda_state)
    L_alda_effective = L_alda * max(gate, 0.05)

    contrib_phys   = weights['w_phys']   * l_phys   * s_phys
    contrib_smooth = weights['w_smooth'] * l_smooth * s_smooth
    contrib_order  = weights['w_order']  * l_order
    contrib_anchor = weights['w_anchor'] * l_anchor
    contrib_sat    = weights['w_sat']    * l_sat

    loss = (contrib_phys + contrib_smooth + L_alda_effective
            + contrib_order + contrib_anchor + contrib_sat
            + contrib_sparse + contrib_entropy + contrib_continuity)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    with torch.no_grad():
        t_per_part = {pid: t_final[part_ids.long() == pid].mean().item() for pid in range(5)}

    return {
        "loss": loss.item(), "mp_rel_err": mp_rel_err, "area": area.item(),
        "l_continuity": (W_CONT_K1 * l_cont1 + W_CONT_K2 * l_cont2).item(),
        "l_smooth": l_smooth.item(), "l_mass": l_mass.item(), "l_collision": l_collision.item(),
        "l_order": l_order.item(), "l_anchor": l_anchor.item(), "l_sat": l_sat.item(),
        "z_gate": z_gate.detach().cpu().numpy(), "t_per_part": t_per_part,
        "new_coords": new_coords.detach(),
    }


def run_section(k, target_mp_k, frozen_thickness_continuous, frozen_coords_k1, frozen_coords_k2,
                scale_k1, scale_k2, device, max_epochs=N_EPOCH_PER_SECTION, lr=1e-3):
    """섹션 k 하나를 처음부터 끝까지 학습(원본 run_training()의 단일 섹션 버전 + continuity loss)."""
    data = build_section_data(k, frozen_thickness_continuous).to(device)
    scale_k = scale_factor(k)

    model = CGDNSeq(in_channels=8, hidden_channels=128, num_layers=4, heads=4, edge_dim=4,
                     max_displacement=50.0, num_parts=5, init_log_alpha=2.0).to(device)
    if frozen_thickness_continuous is not None:
        model.frozen_continuous_thickness = frozen_thickness_continuous

    thick_param_ids = {id(p) for p in model.thickness_decoder.parameters()}
    gate_param_ids = {id(model.log_alpha)}
    main_params = [p for p in model.parameters() if id(p) not in thick_param_ids and id(p) not in gate_param_ids]
    thick_params = list(model.thickness_decoder.parameters())
    gate_params = [model.log_alpha]
    optimizer = optim.AdamW(
        [{'params': main_params, 'name': 'main'},
         {'params': thick_params, 'name': 'thickness_decoder'},
         {'params': gate_params, 'name': 'gate_params', 'lr': 0.0}],
        lr=lr, weight_decay=1e-4)

    weights = {'w_phys': 10.0, 'w_order': 1.0, 'w_smooth': 0.5, 'w_anchor': 0.02, 'w_sat': 0.01,
               'w_sparse': usv18.W_SPARSE}
    alda_state = usv18.make_alda_state()
    pruning_state = usv18.make_pruning_state(num_parts=5, candidate_parts=usv18.CANDIDATE_PARTS)

    x = data.x
    verify_thickness_gradient_relaxed(x[:, :2], x[:, 6:7], x[:, 7:8], data.edge_index)
    target_area_k, _ = usv18.compute_section_area(x[:, :2].cpu(), x[:, 6:7].cpu(), data.edge_index.cpu())

    collision_spec = usv18.build_collision_spec(x[:, :2], x[:, 6:7], x[:, 4], x[:, 5])

    history = {'loss': [], 'mp_rel_err': [], 'l_continuity': [], 'area': [],
               'l_smooth': [], 'l_mass': [], 'l_collision': [], 'l_order': [], 'l_anchor': [], 'l_sat': [],
               'z_gate': [], 't_per_part': []}

    gate_stage_done = False
    for epoch in range(max_epochs):
        if (not gate_stage_done) and epoch == usv18.GATE_ACTIVE_EPOCH:
            for group in optimizer.param_groups:
                if group.get('name') == 'gate_params':
                    group['lr'] = 1e-2
            gate_stage_done = True

        info = sequential_train_step(
            model, data, optimizer, target_mp_k, target_area_k, epoch, max_epochs, weights,
            True, (0.2, 0.7), collision_spec, alda_state, pruning_state,
            frozen_coords_k1, frozen_coords_k2, scale_k, scale_k1, scale_k2)

        for key in ('loss', 'mp_rel_err', 'l_continuity', 'area', 'l_smooth', 'l_mass',
                    'l_collision', 'l_order', 'l_anchor', 'l_sat'):
            history[key].append(info[key])
        history['z_gate'].append(info['z_gate'])
        history['t_per_part'].append(info['t_per_part'])

        if epoch % 30 == 0 or epoch == max_epochs - 1:
            print(f"  [k={k:2d}] epoch {epoch:4d} || loss={info['loss']:.4f} "
                  f"| MpErr={info['mp_rel_err']*100:.2f}% | l_cont={info['l_continuity']:.4f} "
                  f"| area={info['area']:.1f}")

    final_coords = info['new_coords']
    final_z_gate = info['z_gate']
    continuous_thickness = None
    if k == TOP_K:
        continuous_thickness = {pid: info['t_per_part'][pid] for pid in CONTINUOUS_PARTS}
        print(f"  [k={k}] 연속 파트 두께 고정: {continuous_thickness}")

    return model, history, final_coords, final_z_gate, continuous_thickness, data.x[:, 4].cpu()


# ══════════════════════════════════════════════════════════════════
# SECTION 5: 목표 Mp 생성 (command_v1.md §5 — 변경 없음)
# ══════════════════════════════════════════════════════════════════

INITIAL_MP_TABLE = [  # index 0 -> k=17, index 16 -> k=1
    21_637_199, 23_286_535, 24_996_460, 26_766_975, 28_598_079,
    30_489_775, 32_442_063, 34_454_943, 36_528_416, 38_662_482,
    40_857_143, 43_112_398, 45_428_249, 47_804_695, 50_241_738,
    52_739_377, 55_297_613,
]


def generate_gp_target_mp(seed=None, sigma=0.25, ell=3.0):
    """initial_section.md §2 표 기준 GP 스무딩 +/-50% 목표 Mp. 반환: {k(1..17): target_mp}."""
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    initial_mp = np.array(INITIAL_MP_TABLE, dtype=np.float64)
    idx = np.arange(17, dtype=np.float64)
    K = sigma ** 2 * np.exp(-(idx[:, None] - idx[None, :]) ** 2 / (2 * ell ** 2))
    K += 1e-6 * np.eye(17)
    alpha = rng.multivariate_normal(np.zeros(17), K)
    alpha = np.clip(alpha, -0.5, 0.5)
    target_mp = initial_mp * (1.0 + alpha)
    return {TOP_K - i: float(target_mp[i]) for i in range(17)}   # k=17..1


# ══════════════════════════════════════════════════════════════════
# SECTION 6: Stage 0 — Static Dry-Run (k=17 하나만, command_v1.md §6.1)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_static_dry_run(device):
    print("[Stage 0] Static dry-run (k=17 섹션 하나) 시작...")
    data = build_section_data(TOP_K).to(device)
    model = CGDNSeq(in_channels=8, hidden_channels=128, num_layers=4, heads=4, edge_dim=4,
                     max_displacement=50.0, num_parts=5, init_log_alpha=2.0).to(device)
    x = data.x
    assert x.shape[1] == 8, f"노드 feature 8열이어야 함 (실제: {x.shape[1]})"
    fix_x_mask = x[:, 2].bool().unsqueeze(1)
    fix_y_mask = x[:, 3].bool().unsqueeze(1)
    target_mp_node = torch.full((x.shape[0], 1), 2.0e7, device=device)
    new_coords, _, t_final, _, z_gate, _ = model(
        x, data.edge_index, data.edge_attr, target_mp_node, fix_x_mask, fix_y_mask, data.join_pairs,
        thickness_gate=0.0, gate_active=False, temperature=1.0)
    assert not torch.isnan(new_coords).any() and not torch.isnan(t_final).any()
    assert z_gate.shape == (5,), f"z_gate shape 오류(단일 섹션이라 (5,)여야 함): {z_gate.shape}"
    if torch.cuda.is_available():
        print(f"[Stage 0] GPU 메모리: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
    print(f"[Stage 0] 통과 — 노드 {x.shape[0]}개, 엣지 {data.edge_index.shape[1]}개.")


# ══════════════════════════════════════════════════════════════════
# SECTION 7: 시각화 — 순차 히스토리 병합 (command_v1.md §7)
# ══════════════════════════════════════════════════════════════════

def visualize_sequential_training(all_histories, section_boundaries,
                                   save_path="reports/figures/AI_design_v2_result.png"):
    """17번의 개별 학습 히스토리를 하나의 글로벌 스텝 축으로 이어붙여 8-panel dashboard로 렌더링.
    section_boundaries: k별 시작 global step 목록(수직선으로 표시, k 라벨 포함)."""
    keys_panel4 = ['l_smooth', 'l_mass', 'l_collision', 'l_order', 'l_sat']
    global_hist = {k: [] for k in ['loss', 'mp_rel_err', 'l_continuity', 'area'] + keys_panel4}
    for h in all_histories:
        for key in global_hist:
            global_hist[key].extend(h[key])
    steps = list(range(len(global_hist['loss'])))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    def add_section_lines(ax):
        for k, s in section_boundaries:
            ax.axvline(s, color='gray', linestyle=':', linewidth=0.6, alpha=0.6)

    ax = axes[0]
    ax.plot(steps, global_hist['loss'], color='#2196F3', linewidth=1.0)
    add_section_lines(ax)
    ax.set_yscale('log'); ax.set_xlabel('Global step (섹션별 이어붙임)'); ax.set_ylabel('Loss')
    ax.set_title('Total Loss (k=17->1 순차)', fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, [e * 100 for e in global_hist['mp_rel_err']], color='#FF5722', linewidth=1.0)
    ax.axhline(2.0, color='gray', linestyle=':', label='Feasibility 2%')
    add_section_lines(ax)
    ax.set_yscale('log'); ax.set_xlabel('Global step'); ax.set_ylabel('Mp 상대오차 (%)')
    ax.set_title('섹션별 Mp 수렴', fontweight='bold'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, global_hist['l_continuity'], color='#00BCD4', linewidth=1.0)
    add_section_lines(ax)
    ax.set_xlabel('Global step'); ax.set_ylabel('l_continuity')
    ax.set_title('인접 섹션(k+1,k+2) 연속성 loss', fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(steps, global_hist['area'], color='#4CAF50', linewidth=1.0)
    add_section_lines(ax)
    ax.set_xlabel('Global step'); ax.set_ylabel('Area (mm²)')
    ax.set_title('섹션별 단면적 추이', fontweight='bold'); ax.grid(True, alpha=0.3)

    for k, s in section_boundaries[::2]:   # 라벨 겹침 방지로 절반만 표기
        axes[0].text(s, axes[0].get_ylim()[1], f'k={k}', fontsize=6, rotation=90, va='top')

    plt.suptitle('AI_design_v2 — Sequential (k=17->1) Training', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] 순차 학습 dashboard 저장: {save_path}")


def plot_sections_3d_plotly_sequential(final_states, save_path="reports/figures/AI_design_v2_3d.html",
                                       z_coef=15.0):
    """initial_section.py의 plot_sections_3d_plotly() 스타일. final_states: {k: (coords, part_ids, z_gate)}.
    z=(k-1)*z_coef로 쌓아 k=1(최하단)이 z=0, k=17(최상단)이 맨 위가 되도록 한다(initial_section.py와
    동일 관례)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    shown = set()
    for k in sorted(final_states.keys()):   # 1 -> 17
        coords, part_ids, z_gate = final_states[k]
        coords_np = coords.cpu().numpy()
        part_ids_np = part_ids.cpu().numpy().astype(int)
        z = (k - 1) * z_coef
        for pid in range(5):
            mask = part_ids_np == pid
            if not mask.any():
                continue
            pts = coords_np[mask]
            g = float(z_gate[pid]) if pid in CANDIDATE_PARTS else 1.0
            opacity = 0.15 + 0.85 * g
            width = 1.0 + 4.0 * g
            show_legend = pid not in shown
            shown.add(pid)
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=np.full(len(pts), z),
                mode='lines+markers',
                line=dict(color=PART_COLORS[pid], width=width),
                marker=dict(size=2, color=PART_COLORS[pid]),
                name=PART_NAMES[pid], legendgroup=PART_NAMES[pid], showlegend=show_legend,
                opacity=opacity,
                hovertemplate=(f'k={k}, {PART_NAMES[pid]}<br>x=%{{x:.1f}}, y=%{{y:.1f}}'
                               + (f'<br>z_gate={g:.2f}' if pid in CANDIDATE_PARTS else '') + '<extra></extra>'),
            ))

    fig.update_layout(
        title='AI_design_v2 — Sequential (k=17 top -> k=1 bottom) B-Pillar',
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)',
                   zaxis_title='Section (k=1 bottom .. k=17 top)', aspectmode='data'),
        width=1100, height=850,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"[viz] 인터랙티브 3D 저장: {save_path}")


# ══════════════════════════════════════════════════════════════════
# SECTION 8: __main__ — k=17..1 순차 바깥 루프 (command_v1.md §2.1)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-per-section", type=int, default=N_EPOCH_PER_SECTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-start", type=int, default=TOP_K, help="테스트용: 이 k부터 시작(기본 17)")
    parser.add_argument("--k-end", type=int, default=BOT_K, help="테스트용: 이 k에서 종료(기본 1)")
    parser.add_argument("--dry-run-only", action="store_true")
    args, _unknown = parser.parse_known_args()

    os.makedirs("weights", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dry_run_only:
        run_static_dry_run(device)
        sys.exit(0)

    run_static_dry_run(device)

    target_mps = generate_gp_target_mp(seed=args.seed)
    print(f"[target_mp] {[f'k={k}:{v:,.0f}' for k, v in target_mps.items()]}")

    frozen_thickness_continuous = None
    frozen_coords = {}     # k -> (coords, scale_k)
    final_states = {}      # k -> (coords, part_ids, z_gate) — 3D 시각화용
    all_histories = []
    section_boundaries = []
    global_step = 0

    print(f"\n{'='*78}\n[ AI_design_v2 ] Sequential Training k={args.k_start}->{args.k_end}"
          f" | epochs/section={args.epochs_per_section}\n{'='*78}")

    for k in range(args.k_start, args.k_end - 1, -1):
        section_boundaries.append((k, global_step))
        k1_coords, k1_scale = frozen_coords.get(k + 1, (None, None))
        k2_coords, k2_scale = frozen_coords.get(k + 2, (None, None))

        model, history, final_coords, final_z_gate, cont_thickness, part_ids = run_section(
            k, target_mps[k], frozen_thickness_continuous,
            k1_coords, k2_coords, k1_scale, k2_scale, device,
            max_epochs=args.epochs_per_section)

        if cont_thickness is not None:
            frozen_thickness_continuous = cont_thickness

        frozen_coords[k] = (final_coords, scale_factor(k))
        final_states[k] = (final_coords, part_ids, final_z_gate)
        all_histories.append(history)
        global_step += len(history['loss'])

        torch.save(model.state_dict(), f"weights/bpillar_seq_k{k:02d}.pt")

    visualize_sequential_training(all_histories, section_boundaries)
    plot_sections_3d_plotly_sequential(final_states)

    print("[done] weights/bpillar_seq_k*.pt, reports/figures/AI_design_v2_result.png, "
          "reports/figures/AI_design_v2_3d.html 저장 완료")
