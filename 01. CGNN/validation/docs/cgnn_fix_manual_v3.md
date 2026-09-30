# 📋 [Prompt for Claude-Code Agent]

## 1. Role & Objective
* **Role:** Expert Structural Engineering & PyTorch Deep Learning Developer.
* **Objective:** Refactor the `build_simple_section` function in `pna_solver_validate_v6.py` to model a realistic **5-part B-Pillar cross-section** based on provided PDF specifications. You must also update the downstream visualization functions (`visualize_results`, `visualize_epoch_snapshots`) to properly render 5 parts instead of 3.
* **Why:** The current model uses a simplified 3-part section (Outer, Reinf, Inner) with 90 nodes. The real B-Pillar consists of 5 parts (#00, #03, #06, #07, #08) with specific High-Strength Steel (HSS) yield strengths (fy), thicknesses (t), and complex hat-shape geometries. Expanding to 150 nodes (30 per part) will enable high-fidelity reverse-engineering.

## 2. Engineering Specifications (From PDF)
* **Total Width:** 160.0 mm
* **Nodes per Part:** 30 nodes (Total 150 nodes)
* **Fixed Flange Area:** $X \le 15\%$ (24mm) and $X \ge 85\%$ (136mm)
* **Part 0 (#00 Outer Hat):** base_y=31.0, t=2.30, fy=1470.0 MPa, Hat-shape (Offset +50.0)
* **Part 1 (#03 Inner Plate):** base_y=26.0, t=1.60, fy=980.0 MPa, Flat
* **Part 2 (#06 Inner Hat):** base_y=29.0, t=1.60, fy=1470.0 MPa, Hat-shape (Offset +35.0)
* **Part 3 (#07 Patch 1):** base_y=24.0, t=1.40, fy=980.0 MPa, Flat
* **Part 4 (#08 Patch 2):** base_y=22.0, t=1.60, fy=440.0 MPa, Flat

## 3. Implementation Steps

### Step 1: Replace Data Builder Function
Replace the existing `build_simple_section` with the following `build_bpillar_section`:

```python
def build_bpillar_section():
    part_configs = [
        (0, 31.0, 2.30, 1470.0, True),   # #00 Outer
        (1, 26.0, 1.60,  980.0, False),  # #03 Inner Plate
        (2, 29.0, 1.60, 1470.0, True),   # #06 Inner Hat
        (3, 24.0, 1.40,  980.0, False),  # #07 Patch 1
        (4, 22.0, 1.60,  440.0, False),  # #08 Patch 2
    ]
    
    num_nodes = 30
    total_width = 160.0
    dx = total_width / (num_nodes - 1)

    nodes = []
    node_registry = {}
    idx = 0
    
    for part_id, y_base, t_val, fy_val, is_hat_shape in part_configs:
        for i in range(num_nodes):
            x_coord = i * dx
            fix = 1.0 if (x_coord <= total_width * 0.15 or x_coord >= total_width * 0.85) else 0.0
            
            if is_hat_shape:
                if fix == 1.0:
                    y_coord_node = y_base
                else:
                    height_offset = 50.0 if part_id == 0 else 35.0 
                    y_coord_node = y_base + height_offset
            else:
                y_coord_node = y_base

            nodes.append([x_coord, y_coord_node, fix, fix, float(part_id), 0.0, t_val, fy_val])
            node_registry[(part_id, i)] = idx
            idx += 1

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
        for i in range(num_nodes - 1):
            u = node_registry[(part_id, i)]
            v = node_registry[(part_id, i + 1)]
            add_edge(u, v, part_id)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr_list, dtype=torch.float32)
    join_pairs = torch.zeros((0, 2), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, join_pairs=join_pairs), node_registry
```

### Step 2: Update Main Block Initialization
In the `if __name__ == "__main__":` block:
1. Change `build_simple_section()` to `build_bpillar_section()`.
2. Update the target Mp for the new heavier section: Change `TARGET_MP = 2_500_000` to `TARGET_MP = 35_000_000` (Based on PDF showing approx 3.3E7 N·mm).

### Step 3: Update Visualization Functions for 5 Parts
Because the arrays are now expanded to 5 parts, you MUST update both `visualize_results` and `visualize_epoch_snapshots` to prevent index out-of-range errors.

Find these lines in BOTH visualization functions:
```python
    part_colors = {0: '#FF5722', 1: '#FFAA00', 2: '#4CAF50'}
    part_names  = {0: 'Outer(P0)', 1: 'Reinf(P1)', 2: 'Inner(P2)'} # (Or similar names)
    
    for part_id in range(3):
```

**Replace them with:**
```python
    part_colors = {0: '#FF5722', 1: '#FFAA00', 2: '#4CAF50', 3: '#2196F3', 4: '#9C27B0'}
    part_names  = {0: '#00(Outer)', 1: '#03(Plate)', 2: '#06(Inner)', 3: '#07(Patch1)', 4: '#08(Patch2)'}
    
    for part_id in range(5):
```

### Step 4: Execute & Verify
Run the updated `pna_solver_validate_v6.py`.
Verify that the `Nodes` output is `[150, 8]` and that the generated UI charts (Panel 2 and snapshots) correctly plot all 5 layers with their respective colors and geometries (2 Hat-shapes, 3 Flat plates).