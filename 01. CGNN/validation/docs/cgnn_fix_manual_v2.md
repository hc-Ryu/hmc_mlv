# 📋 [Prompt for Claude-Code Agent]

## 1. Role & Objective
* **Role:** Expert Structural Engineering & PyTorch Deep Learning Developer.
* **Objective:** Refactor the `compute_edge_mp_pna` function and related analytic blocks in `pna_solver_validate_v5.py`. You will implement a **"Thick Edge (2D Plate) Model"** to eliminate the discontinuity problem that occurs when the Plastic Neutral Axis (PNA) crosses horizontal edges.
* **Why:** The current 1D edge model causes extreme numerical jumps (step-functions) in net force when a horizontal edge is strictly above or below the PNA, resulting in large Indirect Terms and ~10% gradient errors. By projecting the actual thickness onto the Y-axis, we turn the edge into a 2D vertical span. This allows for a continuous, proportional distribution of tension and compression, driving the equilibrium residual and gradient error to absolute `0`.

## 2. Mathematical Formulation (The "Thick Edge" Model)
Instead of splitting logic by cases, EVERY edge is treated as a vertical span $H$ based on its projected Y-thickness.

1. **Geometry Projection:**
   * Length: $L = \sqrt{(x_u - x_v)^2 + (y_u - y_v)^2}$
   * Projected Y-Thickness: $t_y = t \cdot \frac{|x_u - x_v|}{L}$
   * Top boundary: $y_{top} = \max(y_u, y_v) + \frac{t_y}{2}$
   * Bottom boundary: $y_{bot} = \min(y_u, y_v) - \frac{t_y}{2}$
   * Vertical span: $H = \max(y_{top} - y_{bot}, 1e-12)$ (to prevent division by zero)

2. **Continuous PNA Split Ratio ($\alpha$):**
   * $\alpha$ represents the fraction of the edge's area that is in **Tension** (above PNA).
   * $\alpha = \text{clamp}\left(\frac{y_{top} - y_{pna}}{H}, 0.0, 1.0\right)$

3. **Force and Moment (Vectorized & Continuous):**
   * Area $A = L \cdot t$
   * Net Force = $\sum A \cdot f_y \cdot (2\alpha - 1.0)$
   * Centroid of Top part = $y_{top} - \frac{\alpha \cdot H}{2}$
   * Centroid of Bottom part = $y_{bot} + \frac{(1 - \alpha) \cdot H}{2}$
   * Moment ($M_p$) = $\sum A \cdot f_y \cdot \left[ \alpha \cdot (\text{Centroid}_{top} - y_{pna}) + (1 - \alpha) \cdot (y_{pna} - \text{Centroid}_{bot}) \right]$

## 3. Implementation Steps

### Step 1: Replace `compute_edge_mp_pna`
Replace the entire `compute_edge_mp_pna` function in the file with this new, mathematically elegant version:

```python
import torch

def compute_edge_mp_pna(coords, t, fy, edge_index, n_iter=50):
    mask = edge_index[0] < edge_index[1]
    u, v = edge_index[0][mask], edge_index[1][mask]

    y_u, y_v = coords[u, 1], coords[v, 1]
    x_u, x_v = coords[u, 0], coords[v, 0]
    
    L = torch.sqrt((x_u - x_v) ** 2 + (y_u - y_v) ** 2)
    t_e  = t[u].squeeze(-1)
    fy_e = fy[u].squeeze(-1)
    
    # 1. Projected Y-thickness
    dx = torch.abs(x_u - x_v)
    t_y = t_e * (dx / (L + 1e-12))
    
    y_max = torch.maximum(y_u, y_v)
    y_min = torch.minimum(y_u, y_v)
    
    y_top = y_max + t_y / 2.0
    y_bot = y_min - t_y / 2.0
    H = torch.clamp(y_top - y_bot, min=1e-12)
    
    Area_fy = L * t_e * fy_e

    # 2. Forward 1: Bisection for y_pna
    with torch.no_grad():
        y_lo = coords[:, 1].min().clone() - 5.0
        y_hi = coords[:, 1].max().clone() + 5.0
        for _ in range(n_iter):
            y_mid = 0.5 * (y_lo + y_hi)
            alpha = torch.clamp((y_top - y_mid) / H, 0.0, 1.0)
            net_force = torch.sum(Area_fy * (2.0 * alpha - 1.0))
            
            if net_force > 0: # Tension > Compression -> Move PNA up
                y_lo = y_mid
            else:
                y_hi = y_mid
        y_pna = 0.5 * (y_lo + y_hi)

    # 3. Forward 2: Calculate Mp
    alpha = torch.clamp((y_top - y_pna) / H, 0.0, 1.0)
    
    centroid_top = y_top - (alpha * H) / 2.0
    centroid_bot = y_bot + ((1.0 - alpha) * H) / 2.0
    
    m_top = alpha * (centroid_top - y_pna)
    m_bot = (1.0 - alpha) * (y_pna - centroid_bot)
    
    mp_total = torch.sum(Area_fy * (m_top + m_bot))

    return mp_total, y_pna
```

### Step 2: Unify Analytic Gradient Blocks
Because the new Thick Edge model is fully differentiable and continuous, you must update the gradient calculation blocks in `ImplicitPNASolver.backward`, `validate_backward`, and `run_training` (the analytic `g_an` part). 

Replace the old `m_above`, `m_below`, `m_cross` logic in those sections with the exact same vectorized logic used in Step 1 (Calculate `t_y`, `H`, `y_top`, `y_bot`, `alpha`, and calculate `mp_direct` or `mp_an_val` using `m_top` and `m_bot`).

*Important:* In the gradient blocks, `alpha` must be calculated using the differentiable `y_top` and `y_bot` (which require grad), but `y_pna` is treated as a detached constant. Use `torch.clamp` as it is automatically differentiable.

### Step 3: Update `compute_y_pna_ref` and Equilibrium Checks
Update `compute_y_pna_ref` to use the new `t_y` and `H` logic for its bisection.
Update the Equilibrium Residual check in `validate_forward` and `validate_backward` to use:
`net_force_eq = torch.sum(Area_fy * (2.0 * alpha - 1.0))`

### Step 4: Execute & Verify
Run the updated script. You must verify:
1. `Equilibrium residual` is exactly or near `0.0`.
2. `net_force [Edge 평형 잔차]` is near `0.0`.
3. `mean |indirect term|` is fundamentally `0.0` (not $10^2$ anymore).
4. `Err(A-An)%` and `Err(A-FD)%` drop to near absolute `0.000%` across the board.

---
**[End of Prompt]**

이 로직은 딥러닝과 구조역학을 결합할 때 발생하는 가장 큰 병목인 "이산화(Discretization)에 의한 불연속성"을 해결하는 매우 강력한 솔루션입니다. 적용 후 완전히 깨끗해진 오차율 지표를 보실 수 있을 겁니다!