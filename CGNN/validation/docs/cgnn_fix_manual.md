# 📋 [Prompt for Claude-Code Agent]

## 1. Role & Objective
* **Role:** Expert Structural Engineering & PyTorch Deep Learning Developer.
* **Objective:** Refactor the `ImplicitPNASolver` in the existing `pna_solver_validate_v4.py` file. Convert the current **Node-based (Point-mass)** calculation of the Plastic Neutral Axis (PNA) and Plastic Moment ($M_p$) to an **Edge-based (Line-segment)** continuous calculation.
* **Why:** The current node-based discretization causes a force equilibrium residual, resulting in a non-zero "Indirect Term" during Implicit Function Theorem (IFT) backward propagation, leading to ~10% gradient error. Edge-based integration will reduce this residual to near zero (`1e-8`).

## 2. Mathematical Formulation (Edge-based)
For each edge $E$ connecting nodes $u$ and $v$ with length $L$:
* **Case 1 (Entirely Above PNA):** $y_{min} \ge y_{pna}$
  * Force = $+L \cdot t \cdot f_y$
  * Moment = $L \cdot t \cdot f_y \cdot |(y_u + y_v)/2 - y_{pna}|$
* **Case 2 (Entirely Below PNA):** $y_{max} \le y_{pna}$
  * Force = $-L \cdot t \cdot f_y$
  * Moment = $L \cdot t \cdot f_y \cdot |(y_u + y_v)/2 - y_{pna}|$
* **Case 3 (Crossing PNA):** $y_{min} < y_{pna} < y_{max}$
  * Split ratio $\alpha = (y_{max} - y_{pna}) / (y_{max} - y_{min})$
  * $L_{above} = \alpha \cdot L$, $L_{below} = (1 - \alpha) \cdot L$
  * Force = $(L_{above} - L_{below}) \cdot t \cdot f_y$
  * Moment = $t \cdot f_y \cdot [L_{above} \frac{y_{max} - y_{pna}}{2} + L_{below} \frac{y_{pna} - y_{min}}{2}]$

## 3. Implementation Steps

### Step 1: Add the Edge-based Core Function
Insert the following core calculation logic into `pna_solver_validate_v4.py` (outside the class, or as a static method).

```python
import torch

def compute_edge_mp_pna(coords, t, fy, edge_index, n_iter=40):
    # 1. Prevent duplicate edges (undirected to directed)
    mask = edge_index[0] < edge_index[1]
    u = edge_index[0][mask]
    v = edge_index[1][mask]
    
    y_u, y_v = coords[u, 1], coords[v, 1]
    x_u, x_v = coords[u, 0], coords[v, 0]
    
    L = torch.sqrt((x_u - x_v)**2 + (y_u - y_v)**2)
    t_e = t[u].squeeze(-1)
    fy_e = fy[u].squeeze(-1)
    
    y_max = torch.maximum(y_u, y_v)
    y_min = torch.minimum(y_u, y_v)
    
    # [ Forward 1: Bisection for y_pna ]
    y_lo, y_hi = coords[:, 1].min(), coords[:, 1].max()
    for _ in range(n_iter):
        y_mid = 0.5 * (y_lo + y_hi)
        
        f_above = torch.where(y_min >= y_mid, L, 0.0)
        f_below = torch.where(y_max <= y_mid, -L, 0.0)
        
        cross_mask = (y_min < y_mid) & (y_max > y_mid)
        alpha = torch.where(cross_mask, (y_max - y_mid) / (y_max - y_min + 1e-8), 0.0)
        L_above = alpha * L
        L_below = (1.0 - alpha) * L
        f_cross = torch.where(cross_mask, L_above - L_below, 0.0)
        
        net_force = torch.sum((f_above + f_below + f_cross) * t_e * fy_e)
        
        if net_force > 0:
            y_lo = y_mid
        else:
            y_hi = y_mid
            
    y_pna = 0.5 * (y_lo + y_hi)
    
    # [ Forward 2: Calculate Mp ]
    m_above = torch.where(y_min >= y_pna, L * ((y_max + y_min)/2.0 - y_pna), 0.0)
    m_below = torch.where(y_max <= y_pna, L * (y_pna - (y_max + y_min)/2.0), 0.0)
    
    cross_mask = (y_min < y_pna) & (y_max > y_pna)
    alpha = torch.where(cross_mask, (y_max - y_pna) / (y_max - y_min + 1e-8), 0.0)
    L_above = alpha * L
    L_below = (1.0 - alpha) * L
    m_cross = torch.where(cross_mask, 
                          L_above * (y_max - y_pna)/2.0 + L_below * (y_pna - y_min)/2.0, 
                          0.0)
    
    mp_total = torch.sum((m_above + m_below + m_cross) * t_e * fy_e)
    
    return mp_total, y_pna
```

### Step 2: Update `ImplicitPNASolver.forward`
Refactor the `forward` method in `ImplicitPNASolver` to use the new edge-based logic instead of the flat node-based bisection.
* Input `edge_index` is now **required** in the forward pass.
* The method should call `mp_pred, y_pna = compute_edge_mp_pna(...)`.
* Save tensors for backward as before.

### Step 3: Align Analytic Validation Methods
Update `compute_y_pna_ref` and the analytic validation parts inside `validate_forward` and `validate_backward` to utilize `compute_edge_mp_pna`. 
* Ensure that the `mp_analytic` calculated independently matches `mp_solver` exactly.
* The expected behavior is that the printed `Equilibrium residual` and `sum(t·fy·s)` (Indirect Term magnitude) will drop significantly (close to `1e-8`), confirming that the discretization error is resolved.

### Step 4: Execute and Verify
Run `pna_solver_validate_v4.py`. Check the stdout to confirm:
1. `max |indirect term|` is $\approx 0$.
2. Gradient errors (`Err(A-An)%` and `Err(A-FD)%`) are well below $1.0\%$.

---
**[End of Prompt]**

이 가이드라인은 코드 내에 분산되어 있던 수학적 한계와 해결책을 Claude가 파악하기 쉽게 모듈화한 것입니다. 그대로 전달하시면 훌륭하게 리팩토링을 수행할 것입니다! 추가적으로 수정하고 싶은 제약조건이 있다면 편하게 말씀해 주세요.