#!/usr/bin/env python
# coding: utf-8

"""
uni_section_v17_01.py
─────────────────────────────────────
검증 실험: Part3(part_id=3, Patch1)의 z_gate가 프루닝 임계값(thresh_low=0.08)에는
못 미치지만 0.5 근처까지 내려온 시점에서, Part3를 강제로 z=0(완전 제거)으로 만들고
forward만(학습 없이) 돌려서 Mp err/l_collision이 feasibility 기준
(Mp err < 2%, l_collision < 0.05)을 유지하는지 확인한다.

설계 근거(/synod general 세션, Gemini flash/medium + OpenAI gpt4o 병렬 검토):
  - uni_section_v17.py의 함수/클래스를 그대로 import해서 재사용(중복 방지).
  - run_training()은 임의 epoch 시점의 raw state_dict를 반환하지 않으므로(best_feasible/
    best_mp만 조건부로 저장), 동일 시드로 run_training과 완전히 같은 학습 루프를 얇게 재현하며
    (train_step/optimizer 스케줄/update_pruning_state 로직은 v17 그대로 재사용) 실제 프루닝
    임계값과 비교되는 pruning_state['ewma_z'][3]가 처음 0.5 이하로 내려오는 epoch에서
    state_dict를 캡처한다. (초기 시도에서는 eval 모드 결정론적 z_gate로 crossing을 판정했으나,
    이는 HardConcrete stochastic 샘플의 EWMA보다 항상 높게 나와 300epoch 동안 0.76까지만
    내려가고 0.5에 도달하지 못함을 확인 — 실제 pruning 판정은 stochastic 샘플의 EWMA를 쓰므로
    이 실험도 동일 지표로 통일함.)
  - t_raw는 forward()가 직접 반환하지 않으므로, t_final = t_raw * z_gate[part_id]에서
    t_raw = t_final / z_gate[part_id]로 역산해 복구한다(Gemini의 monkey-patch 방식 대신 채택 —
    compute_gates는 모듈 전역 함수이지 model의 바인딩 메서드가 아니므로 model.compute_gates
    패치는 동작하지 않음을 확인함). new_coords는 z_gate에 의존하지 않으므로 natural/forced
    양쪽에서 동일하게 재사용 가능.
"""

import copy
import sys
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import uni_section_v17 as v17


PART_ID_TO_FORCE = 3          # Patch1 (CANDIDATE_PARTS[0])
Z_GATE_CAPTURE_THRESHOLD = 0.5  # eval 모드 결정론적 z_gate[3]가 이 값 이하로 처음 내려오는 시점 캡처
MAX_CAPTURE_EPOCHS = 700


def capture_checkpoint_at_z_gate_crossing(part_id=PART_ID_TO_FORCE,
                                           threshold=Z_GATE_CAPTURE_THRESHOLD,
                                           max_epochs=MAX_CAPTURE_EPOCHS):
    """
    run_training()과 동일한 학습 절차(동일 시드/동일 하이퍼파라미터/동일 train_step)를 재현하되,
    eval 모드 결정론적 z_gate[part_id]가 threshold 이하로 처음 내려오는 epoch에서
    model.state_dict()를 캡처해서 반환한다.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    data, node_registry = v17.build_bpillar_section()

    TARGET_MP = 7_421_470
    target_mps = {0: TARGET_MP}

    weights = {
        'w_phys':      10.0,
        'w_collision':  5.0,
        'w_order':      1.0,
        'w_mass':       2.0,
        'w_smooth':     0.5,
        'w_anchor':     0.02,
        'w_sat':        0.01,
        'w_sparse':     v17.W_SPARSE,
    }

    # 검증 스크립트는 소규모(93 nodes) forward-only 비교이므로 CPU로 고정 --
    # 캡처된 state_dict를 이후 Phase 2에서 새 모델에 로드할 때 device mismatch를 피하기 위함.
    device = torch.device('cpu')
    data = data.to(device)

    model = v17.CGDN(
        in_channels=8, hidden_channels=128, num_layers=4, heads=4,
        edge_dim=4, max_displacement=50.0, num_parts=5, init_log_alpha=2.0,
    ).to(device)

    thick_param_ids = {id(p) for p in model.thickness_decoder.parameters()}
    gate_param_ids  = {id(model.log_alpha)}
    main_params  = [p for p in model.parameters()
                    if id(p) not in thick_param_ids and id(p) not in gate_param_ids]
    thick_params = list(model.thickness_decoder.parameters())
    gate_params  = [model.log_alpha]
    lr = 1e-3
    weight_decay_val = 1e-4
    optimizer = torch.optim.AdamW(
        [{'params': main_params,  'name': 'main'},
         {'params': thick_params, 'name': 'thickness_decoder'},
         {'params': gate_params,  'name': 'gate_params', 'lr': 0.0}],
        lr=lr, weight_decay=weight_decay_val)

    alda_state = v17.make_alda_state()
    pruning_state = v17.make_pruning_state(num_parts=5, candidate_parts=v17.CANDIDATE_PARTS)

    x = data.x
    edge_index_cpu = data.edge_index.cpu()
    base_t_cpu = x[:, 6:7].cpu()
    part_ids = x[:, 4]
    section_ids = x[:, 5]

    v17.verify_thickness_gradient(x[:, :2], x[:, 6:7], x[:, 7:8], data.edge_index)
    target_area, _ = v17.compute_section_area(x[:, :2].cpu(), base_t_cpu, edge_index_cpu)
    collision_spec = v17.build_collision_spec(x[:, :2], x[:, 6:7], part_ids, section_ids)

    stage2_done = False
    gate_stage_done = False

    print(f"--- Phase 1: run_training과 동일한 절차로 z_gate[{part_id}] <= {threshold} "
          f"crossing 시점 캡처 ---")

    for epoch in range(max_epochs):
        if (not stage2_done) and epoch == v17.STAGE2_EPOCH:
            for group in optimizer.param_groups:
                if group.get('name') == 'thickness_decoder':
                    for p in group['params']:
                        optimizer.state.pop(p, None)
                    group['lr'] = 0.3 * lr
            stage2_done = True

        if (not gate_stage_done) and epoch == v17.GATE_ACTIVE_EPOCH:
            for group in optimizer.param_groups:
                if group.get('name') == 'gate_params':
                    group['lr'] = 1e-2
            gate_stage_done = True

        info = v17.train_step(model, data, optimizer, target_mps, target_area,
                               epoch, max_epochs, weights, curriculum=True,
                               curriculum_ratio=(0.2, 0.7), collision_spec=collision_spec,
                               alda_state=alda_state)

        # [수정] run_training/results_v17_1의 실제 프루닝 판정 기준과 동일하게,
        # 결정론적(eval) z_gate가 아니라 train_step이 반환한 학습 시점(stochastic) z_gate로
        # 갱신되는 EWMA(ewma_z)를 crossing 기준으로 사용한다.
        # 이유: 결정론적 z_gate(HardConcrete 기댓값)는 stochastic 샘플의 EWMA보다 항상 높게
        # 나옴(클램프된 노이즈 샘플의 평균은 log_alpha가 나타내는 결정론적 점보다 더 자주 0쪽으로
        # 쏠림). 실제 pruning_state['ewma_z']가 임계값(thresh_low=0.08)과 비교되는 대상이므로,
        # 이 실험도 동일한 지표를 기준으로 crossing 시점을 잡아야 results_v17_1과 정합적이다.
        if info['gate_active']:
            v17.update_pruning_state(pruning_state, info['z_gate'])

        z_val = pruning_state['ewma_z'][part_id].item()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch:04d}: ewma_z[{part_id}] = {z_val:.4f} "
                  f"(raw z_gate[{part_id}]={info['z_gate'][part_id]:.4f})")

        if z_val <= threshold:
            print(f"\n[Capture] epoch {epoch}: ewma_z[{part_id}] = {z_val:.4f} "
                  f"<= {threshold} -- state_dict capture")
            captured_state = copy.deepcopy(model.state_dict())
            return captured_state, data, target_mps, target_area, collision_spec, epoch, z_val

    # threshold를 끝까지 못 넘으면 마지막 epoch 상태를 그대로 사용(참고용)
    print(f"\n[Warning] {max_epochs}epoch 동안 ewma_z[{part_id}] <= {threshold} 도달 못함. "
          f"마지막 epoch 상태로 진행합니다(참고용, 임계값 미도달 상태에서의 forced 테스트).")
    captured_state = copy.deepcopy(model.state_dict())
    return captured_state, data, target_mps, target_area, collision_spec, max_epochs - 1, z_val


def _forward_and_metrics(model, data, target_mps, collision_spec, z_gate_override=None):
    """
    model.eval() 상태에서 forward 1회 + Mp err/l_collision 계산.
    z_gate_override가 주어지면 t_final을 t_raw = t_final/z_gate로 역산한 뒤
    t_final_forced = t_raw * z_gate_override로 재계산해서 사용한다(coords는 z_gate와 무관하므로
    natural/forced 동일하게 재사용).
    """
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    join_pairs = data.join_pairs if hasattr(data, 'join_pairs') else None

    fix_x_mask = x[:, 2].bool().unsqueeze(1)
    fix_y_mask = x[:, 3].bool().unsqueeze(1)
    part_ids = x[:, 4]
    section_ids = x[:, 5]
    fy = x[:, 7].unsqueeze(1)

    target_mp_node = torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)
    for section in torch.unique(section_ids):
        section_mask = (section_ids == section)
        target_mp_node[section_mask] = target_mps[int(section.item())]

    with torch.no_grad():
        new_coords, _, t_final, _, z_gate, _ = model(
            x, edge_index, edge_attr, target_mp_node,
            fix_x_mask, fix_y_mask, join_pairs, thickness_gate=1.0, gate_active=True)

        z_gate_used = z_gate
        if z_gate_override is not None:
            part_ids_local = part_ids.long()
            z_gate_node = z_gate[part_ids_local].unsqueeze(-1).clamp(min=1e-6)
            t_raw = t_final / z_gate_node
            z_gate_node_forced = z_gate_override[part_ids_local].unsqueeze(-1)
            t_final = t_raw * z_gate_node_forced
            z_gate_used = z_gate_override

        pred_mp_sections = []
        for section in torch.unique(section_ids):
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

            pred_mp_section = v17.calculate_mpl(coords_section, t_section, fy_section, edge_index_section)
            pred_mp_sections.append(pred_mp_section.item())

        pred_mp_total = float(np.sum(pred_mp_sections))
        target_mp_total = float(sum(target_mps.values()))
        mp_rel_err = abs(pred_mp_total - target_mp_total) / target_mp_total

        l_collision = v17.compute_collision_loss_v5(
            new_coords, t_final, part_ids, section_ids, collision_spec, z_gate=z_gate_used)

    return {
        'pred_mp': pred_mp_total,
        'mp_rel_err': mp_rel_err,
        'l_collision': l_collision.item(),
        'z_gate': z_gate_used.detach().cpu().numpy(),
    }


def run_validation_experiment():
    (state_dict, data, target_mps, target_area, collision_spec,
     captured_epoch, captured_z) = capture_checkpoint_at_z_gate_crossing()

    model = v17.CGDN(
        in_channels=8, hidden_channels=128, num_layers=4, heads=4,
        edge_dim=4, max_displacement=50.0, num_parts=5, init_log_alpha=2.0,
    )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\n--- Phase 2: Natural vs Forced(Part{PART_ID_TO_FORCE}=z=0) 비교 "
          f"(captured epoch={captured_epoch}, z_gate[{PART_ID_TO_FORCE}]={captured_z:.4f}) ---")

    natural = _forward_and_metrics(model, data, target_mps, collision_spec, z_gate_override=None)

    z_gate_natural = torch.as_tensor(natural['z_gate'])
    z_gate_forced = z_gate_natural.clone()
    z_gate_forced[PART_ID_TO_FORCE] = 0.0
    forced = _forward_and_metrics(model, data, target_mps, collision_spec, z_gate_override=z_gate_forced)

    print(f"\n[Natural]  z_gate={natural['z_gate'].round(4).tolist()}")
    print(f"           pred_mp={natural['pred_mp']:,.0f} N·mm | Mp err={natural['mp_rel_err']*100:.4f}% "
          f"| l_collision={natural['l_collision']:.6f}")

    print(f"\n[Forced: Part{PART_ID_TO_FORCE} z=0]  z_gate={forced['z_gate'].round(4).tolist()}")
    print(f"           pred_mp={forced['pred_mp']:,.0f} N·mm | Mp err={forced['mp_rel_err']*100:.4f}% "
          f"| l_collision={forced['l_collision']:.6f}")

    mp_feasible = forced['mp_rel_err'] < 0.02
    collision_feasible = forced['l_collision'] < 0.05
    is_feasible = mp_feasible and collision_feasible

    print(f"\n--- Feasibility Assessment (Forced case) ---")
    print(f"  Mp err < 2%        : {mp_feasible}  ({forced['mp_rel_err']*100:.4f}%)")
    print(f"  l_collision < 0.05 : {collision_feasible}  ({forced['l_collision']:.6f})")
    print(f"  OVERALL: {'PASS (feasible)' if is_feasible else 'FAIL (infeasible)'}")

    if is_feasible:
        print(f"\n[결론] z_gate[{PART_ID_TO_FORCE}]≈{captured_z:.2f} 시점에서 이미 Part{PART_ID_TO_FORCE}를 "
              f"완전히 제거해도 feasibility가 유지됨 -- pruning thresh_low(0.08)가 과도하게 "
              f"보수적일 가능성이 있는 근거.")
    else:
        print(f"\n[결론] z_gate[{PART_ID_TO_FORCE}]≈{captured_z:.2f} 시점에서 Part{PART_ID_TO_FORCE}를 제거하면 "
              f"feasibility가 깨짐 -- 현재 임계값(0.08)이 필요한 수준이며 섣불리 완화하면 안 됨.")


if __name__ == "__main__":
    run_validation_experiment()
