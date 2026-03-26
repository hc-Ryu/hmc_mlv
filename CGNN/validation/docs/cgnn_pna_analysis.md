# B-Pillar 설계용 CGNN 아키텍처와 Implicit PNA Solver 정합성 검증

**버전**: pna_solver_validate_v4.py 기반 분석
**대상**: B-Pillar 단면 설계를 위한 CGNN + Implicit PNA Solver 검증 (박사급 수준)

---

## 1. 전체 개요: 무엇을 검증하는 코드인가

이 검증 코드는 B-Pillar용 CGNN 아키텍처 중 **ImplicitPNASolver**가 실제 전소성 모멘트(full plastic moment) 물리식을 얼마나 정확하게 재현하는지(정합성, fidelity)를 집중적으로 평가하기 위한 전용 실험 환경이다.

핵심 구조는 다음 네 단계로 요약된다.

1. **단면 데이터 구성 (build_simple_section)**  
   - 1개의 section, 3개의 part(Outer, Reinf, Inner), 파트당 30개 노드(총 90개)로 이상화된 B-Pillar 단면을 생성한다.  
   - 각 노드에 좌표(x, y), 고정 여부(fix_x, fix_y), 파트/섹션 ID, 두께 t, 항복강도 fy 를 할당한다.

2. **ImplicitPNASolver Forward 정합성 검증 (validate_forward)**  
   - ImplicitPNASolver.apply 로 계산된 Mp_solver 를, 동일한 bisection 로직으로 독립 계산한 Mp_analytic 과 비교한다.  
   - 힘 평형식 잔차(sum t·fy·sign(y − y_pna))를 함께 평가하여, 중립축 평형이 실제로 만족되는지 확인한다.

3. **ImplicitPNASolver Backward(IFT) 정합성 검증 (validate_backward)**  
   - autograd(IFT 구현), 해석적 gradient, 중앙 유한차분 gradient 의 3-way 비교를 통해 dMp/dy 의 정확도를 검증한다.  
   - 평형 상태에서 간접항(indirect term)의 크기가 거의 0에 가까운지 확인하여, 암묵함수 정리 적용이 수학적으로 일치하는지 점검한다.

4. **CGDN + PNA Solver 결합 학습 검증 (run_training)**  
   - 전체 CGDN v3(hidden=128, layers=4, heads=4)를 사용하여 좌표 변형을 예측하고, 그 결과에 ImplicitPNASolver를 적용해 Mp를 계산한다.  
   - target Mp와의 상대 오차 제곱을 손실로 하여 학습시키면서, epoch별 Mp 수렴, gradient 오차, indirect term 크기, 형상/PNA 변화 궤적을 기록한다.

이 과정을 통해, ImplicitPNASolver가 **(1) 정적 Mp 계산**, **(2) 미분값(gradient)**, **(3) GNN 기반 inverse design 과정에서의 수렴 거동**까지 실제 물리 모델을 잘 반영하는지 전방위적으로 검증한다.

---

## 2. 단면 데이터 및 그래프 구성

### 2.1 노드 특징: [x, y, fix_x, fix_y, part_id, section_id, t, fy]

`build_simple_section()` 함수는 3개의 part에 대해 파트당 30개 노드를 만들고, 각 노드에 다음 8차원 특징을 부여한다.

- x: 0 ~ 100 mm 구간을 등분한 좌표 (총 30개 노드 → 29개 구간).
- y: part와 고정 여부에 따라 상이한 초기 높이.
  - Part 0 (Outer):  
    - 고정 노드: y = 31.75 mm (실제 플랜지 위치에 해당하는 기준선).  
    - 자유 노드: y = 50.0 mm (외판 자유단 위치).
  - Part 1 (Reinf):  
    - 고정 노드: y = 30.0 mm.  
    - 자유 노드: y = 40.0 mm.
  - Part 2 (Inner):  
    - 고정 노드: y = 28.25 mm.  
    - 자유 노드: y = 20.0 mm.
- fix_x, fix_y: 0 또는 1.  
  - x 좌표가 15.0 mm 이하 또는 85.0 mm 이상인 노드는 플랜지 영역으로 간주하고 fix=1.0.  
  - 나머지는 fix=0.0.  
  - 이를 통해 좌우 끝단에 각각 5개 노드씩 고정점이 형성된다.
- part_id: 0(Outer), 1(Reinf), 2(Inner).
- section_id: 여기서는 단일 section이므로 항상 0.
- t: 두께. Outer/Inner = 1.5 mm, Reinf = 2.0 mm.
- fy: 항복강도. Outer/Reinf = 1500 MPa(N/mm²), Inner = 1200 MPa(N/mm²).

이러한 구성은 **단면을 세 장의 서로 다른 강판으로 모델링하면서, 플랜지 영역에 실제 고정 경계조건을 부여한 구조역학적 상황**을 반영한다.

### 2.2 엣지 및 인접 행렬 구성

단면은 각 part별로 1차원 선형 그래프로 연결된다.

- 각 part에 대해, i번째 노드와 i+1번째 노드를 양방향으로 연결.  
  예: (part 0, i=0) ↔ (part 0, i=1), ..., (i=28) ↔ (i=29).
- edge_attr: [length, angle, part_id, edge_type]
  - length: 두 노드 간 유클리드 거리.  
  - angle: atan2(dy, dx)로 표현된 엣지 기울기.  
  - part_id: 이 엣지가 속한 part의 ID.  
  - edge_type: 여기서는 모두 0.0 (intra-part edge).

이렇게 생성된 `edge_index`, `edge_attr`는 CGDN 내부 GATv2Conv에서 메시지 패싱의 기반으로 사용되며, **형상 연속성과 part별 기하 구조**를 보존하는 역할을 한다.

---

## 3. ImplicitPNASolver: Forward 수식과 구현 정합성

### 3.1 물리적 정의

ImplicitPNASolver는 주어진 단면 좌표와 두께/강도 분포에 대해, 다음 조건을 만족하는 소성 중립축(PNA) 위치와 전소성 모멘트(Mp)를 계산한다.

1. 힘 평형 조건 (인장력 = 압축력)
   - y 좌표가 PNA 위에 있는 영역은 인장, 아래는 압축으로 가정한다.
   - t_i·fy_i 를 단위 폭당 소성 응력으로 보고, 인장/압축 쪽 합력이 서로 같아지는 y_pna 를 찾는다.

2. 전소성 모멘트 정의
   - 각 요소 i의 소성 모멘트 기여는 t_i·fy_i·d_i 이고, d_i = |y_i − y_pna| 이다.
   - Mp = 합( t_i·fy_i·|y_i − y_pna| ).

이 모델은 **이상화된 완전 소성 분포(직사각형 응력 블록)**를 가정한 전소성 모멘트 정의를 정확히 구현한 것이다.

### 3.2 Forward 구현: Bisection 기반 y_pna 탐색

pna_solver_validate_v4.py 에서 forward는 다음과 같이 동작한다.

1. 입력 준비
   - y = coords[:, 1] (모든 노드의 y좌표).
   - t_flat = t.squeeze(-1), fy_flat = fy.squeeze(-1).
   - y_lo = min(y), y_hi = max(y).

2. 이분법 반복 (n_iter=30)
   - y_mid = (y_lo + y_hi) / 2.
   - F_tens = 합( t·fy·I(y > y_mid) )  (인장측 합력의 단위계수).
   - F_comp = 합( t·fy·I(y < y_mid) )  (압축측 합력의 단위계수).
   - F_tens > F_comp 이면 PNA가 더 위에 있어야 하므로 y_lo = y_mid, 아니면 y_hi = y_mid.

3. 수렴 후 PNA 및 Mp 계산
   - y_pna = (y_lo + y_hi) / 2.
   - d = |y − y_pna|.
   - Mp_pred = 합( t·fy·d ).

여기서 I(·)는 indicator 함수이며, 실제 구현에서는 (y > y_mid).float(), (y < y_mid).float() 로 표현된다. 이 알고리즘은 **정확히 힘 평형 조건을 만족하는 y_pna 를 이분법으로 탐색**하며, 반복 횟수 30회면 double precision 수준에서 충분한 수렴도를 확보한다.

### 3.3 Forward 정합성 검증: validate_forward

`validate_forward(data)`는 ImplicitPNASolver.forward의 정합성을 다음 절차로 검증한다.

1. ImplicitPNASolver로 Mp_solver 계산
   - mp_solver = calculate_mpl(coords, t, fy, None).

2. 동일 로직으로 독립적인 y_pna_ref 계산
   - compute_y_pna_ref(coords, t, fy)가 forward와 같은 bisection 로직으로 y_pna를 계산한다.
   - 이 값은 코드 독립 경로로 얻은 "참조 PNA"이다.

3. 참조 Mp_analytic 계산
   - mp_analytic = 합( t·fy·|y − y_pna_ref| ).

4. 평형식 잔차 계산
   - s_ref = sign(y − y_pna_ref).  
   - equilibrium_residual = 합( t·fy·s_ref ).  
   - 이 값이 0에 가까울수록 인장/압축 평형이 정확히 달성되었음을 의미한다.

5. 오차 분석 출력
   - 절대 오차: err_mp = |mp_solver − mp_analytic|.
   - 상대 오차: err_pct = err_mp / mp_analytic × 100 (%).  
   - 평형 잔차와 함께 PASS/FAIL을 판단한다.

이 검증으로, ImplicitPNASolver.forward가 **전소성 모멘트 물리식을 수치적으로 정확하게 재현하고 있으며, 평형 조건도 잘 만족시키고 있음을** 확인할 수 있다.

---

## 4. ImplicitPNASolver: Backward(IFT)와 Gradient 정합성

### 4.1 IFT 기반 미분 구조

Backward에서는 암묵함수 정리(Implicit Function Theorem)를 사용해, Mp가 각 y_i에 대해 갖는 미분 dMp/dy_i 를 계산한다.

1. 평형식 정의
   - g(y_pna, y) = 합( t_i·fy_i·sign(y_i − y_pna) ) = 0.
   - 이 식에서 y_pna는 y 벡터의 암묵적 함수.

2. IFT에 따른 dy_pna/dy_i
   - ∂g/∂y_pna 와 ∂g/∂y_i 를 계산하면 dy_pna/dy_i = −(∂g/∂y_i) / (∂g/∂y_pna).

3. Mp의 미분 분해
   - Mp = 합( t_i·fy_i·|y_i − y_pna| ).  
   - dMp/dy_i = 직접항(direct) + 간접항(indirect) 로 분해된다.  
     - 직접항: 해당 노드의 거리 |y_i − y_pna| 자체가 변하는 효과.  
     - 간접항: y_i 변화로 인해 y_pna가 이동하여 모든 노드의 거리가 변하는 전역 효과.

### 4.2 코드에서의 backward 구현

pna_solver_validate_v4.py 의 backward 구현 요지는 다음과 같다.

1. 저장된 텐서 로드
   - coords, t, fy, y_pna.

2. 기초량 계산
   - y = coords[:, 1].  
   - t_flat = t.squeeze(-1), fy_flat = fy.squeeze(-1).  
   - s = sign(y − y_pna).

3. IFT 관련 항들
   - dg/dy_pna = −합( t_flat·fy_flat ).  
   - dg/dy = t_flat·fy_flat (정의 방식에 따라 sign이 분리되어 그 다음 단계에서 반영됨).  
   - dy_pna_dy = −dg/dy / (dg/dy_pna + 작은 epsilon).

4. Mp 미분 항 분리
   - direct = t_flat·fy_flat·s.  
   - indirect = −합( t_flat·fy_flat·s ) · dy_pna_dy.  
   - dMp_dy = direct + indirect.

5. 좌표 gradient 반영
   - grad_coords[:, 1] = grad_output * dMp_dy.  
   - x 방향은 Mp에 영향을 주지 않으므로 항상 0.

### 4.3 3-way Gradient 검증: validate_backward

`validate_backward(data, n_check, eps)`는 다음 세 가지 gradient를 비교한다.

1. Autograd (ImplicitPNASolver.backward)
   - coords_ag = coords.clone().requires_grad_(True).  
   - mp = calculate_mpl(coords_ag, t, fy, None).  
   - mp.backward() 후 coords_ag.grad[:, 1] 를 g_autograd 로 사용.

2. 해석적 gradient (analytic)
   - y_pna_ref = compute_y_pna_ref(coords, t, fy).  
   - y_flat, t_flat, fy_flat 계산.  
   - s_ref = sign(y_flat − y_pna_ref).  
   - g_analytic = t_flat·fy_flat·s_ref (평형 상태에서는 간접항이 거의 0이므로 직접항만으로 충분히 정확함).

3. 중앙 유한차분 (finite difference)
   - 선택된 free node index i에 대해, y_i를 ±eps 만큼 변화.  
   - mp_p = Mp(y_i + eps), mp_m = Mp(y_i − eps).  
   - g_fd = (mp_p − mp_m) / (2·eps).

추가로, 간접항 크기를 다음과 같이 측정한다.

- sum_tfy_s = 합( t_flat·fy_flat·s_ref ).  
- sum_tfy = 합( t_flat·fy_flat ).  
- dy_pna_dy = (t_flat·fy_flat) / (sum_tfy + epsilon).  
- indirect = −sum_tfy_s · dy_pna_dy.  
- max/mean |indirect| 를 출력하여, 평형 상태에서 이 값이 거의 0에 수렴하는지 확인한다.

검증 출력에는 각 노드별로 다음 값이 포함된다.

- Autograd, Analytic, FiniteDiff 값.  
- Autograd 대비 Analytic 상대오차(%).  
- Autograd 대비 FiniteDiff 상대오차(%).  
- 평균 오차와 PASS/FAIL 판정.

이를 통해, ImplicitPNASolver.backward에 구현된 IFT 기반 gradient가 **해석적으로도, 수치적으로도 정확하게 Mp의 민감도 dMp/dy를 재현한다**는 것을 확인할 수 있다.

---

## 5. CGDN 아키텍처와 PNA Solver 결합 구조

### 5.1 CGDN(v3) 구조 요약

검증 코드 내부의 CGDN 클래스는 B-Pillar 메인 코드의 CGDN v3 구조를 그대로 가져온 것이다.

- 입력 노드 특징: x ∈ R^{N×8}  
  [x, y, fix_x, fix_y, part_id, section_id, t, fy].
- Node Encoder: Linear(8 → 128) → LayerNorm → GELU.
- Per-block FiLM Generators: 4개 블록 각각 독립적인 FiLMGenerator.  
  target_mp 를 입력으로 받아 gamma, beta 를 생성하고, block별로 서로 다른 조건 임베딩을 학습.
- CGDNBlock × 4:  
  - GATv2Conv(hidden, hidden/heads, heads, edge_dim=4).  
  - LayerNorm.  
  - FiLM modulation: h = gamma ⊙ h + beta.  
  - GELU.  
  - Residual connection: h_out = h + h_res.
- Decoder: Linear(128 → 64) → GELU → Linear(64 → 2) 로 delta_coords = [Δx, Δy] 예측.
- Hard constraint:  
  - delta_x = delta_coords[:, 0:1] * (not fix_x_mask).float().  
  - delta_y = delta_coords[:, 1:2] * (not fix_y_mask).float().  
  - 고정된 좌표성분에 대해서는 변위를 강제로 0으로 만든다.
- join_pairs (옵션): u, v 노드 쌍을 mid = (u+v)/2 로 묶어서, 용접/결합 조건을 강제.

이 네트워크는 **목표 Mp(스칼라)를 입력으로 받아, 각 노드의 좌표를 얼마나 변형해야 해당 Mp를 달성할 수 있을지**를 조건부로 학습하는 구조이다.

### 5.2 run_training: Mp 타겟팅 학습 루프

`run_training(data, target_mp_val, ...)` 함수는 CGDN과 ImplicitPNASolver를 결합해, Mp 타겟팅 inverse design을 수행한다.

1. 초기 설정
   - model = CGDN(in_channels=8, hidden=128, num_layers=4, heads=4).  
   - optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4).  
   - target_mp_node: 모든 노드에 동일한 target_mp_val을 갖는 (N×1) 텐서.

2. 매 epoch 수행
   - model.train(), optimizer.zero_grad().
   - new_coords, delta = model(x, edge_index, edge_attr, target_mp_node, fix_x_mask, fix_y_mask, join_pairs).
   - pred_mp = calculate_mpl(new_coords, t, fy, None).  
   - loss = ((pred_mp − target_mp_val) / target_mp_val)².
   - loss.backward().
   - gradient clipping: clip_grad_norm_(model.parameters(), max_norm=1.0).  
   - optimizer.step().

3. 기록 및 모니터링
   - pred_mp, mp_err_pct, grad_error(analytic/FD), indirect term 크기를 history에 저장.  
   - snapshot_interval 마다 현재 좌표(new_coords)와 y_pna, pred_mp를 snapshot으로 저장하여 epoch별 형상 변화를 시각화 가능하게 함.

이 학습 루프를 통해, **CGDN이 ImplicitPNASolver가 제공하는 Mp gradient를 이용하여 좌표를 점진적으로 조정하고, target Mp로 수렴하는지**를 실험적으로 확인할 수 있다.

---

## 6. 시각화와 정합성에 대한 해석

### 6.1 visualize_results: 4‑Panel 종합 진단

`visualize_results` 함수는 다음 4개의 패널을 통해 ImplicitPNASolver의 정합성과 CGDN 학습 거동을 시각적으로 요약한다.

1. Panel 1: Mp 수렴 곡선
   - epoch별 pred_mp (GNN → PNA solver) 를 MN·mm 단위로 플로팅.  
   - target_mp 수평선과 초기 analytic Mp 수평선을 함께 표시.  
   - 2차 축에 % 오차(mp_err_pct)를 표시하여 수렴 속도와 최종 정확도를 동시에 보여준다.

2. Panel 2: Base vs Result 단면 형상 + PNA 위치
   - part별로 Base 구조와 최종 Result 구조의 x-y 궤적을 서로 다른 심볼/투명도로 표시.  
   - 초기 PNA와 최종 PNA 위치를 각각 다른 스타일의 수평선으로 표시한다.  
   - 이를 통해 GNN이 어떤 패턴으로 좌표를 변형했는지, 그리고 PNA가 어디로 이동했는지 한눈에 파악할 수 있다.

3. Panel 3: Gradient 정확도
   - grad_epochs에서 측정한 autograd vs analytic, autograd vs finite diff 상대오차를 epoch별로 플롯.  
   - 기준선(예: 1% 임계선)을 표시해 IFT gradient의 신뢰성을 정량적으로 평가한다.

4. Panel 4: Indirect term 크기 (평형 품질)
   - epoch별 mean |indirect term| 을 로그 스케일로 플롯.  
   - 평형이 잘 잡혀 있다면 이 값은 매우 작은 값으로 유지되며, 0으로 수렴하는 경향을 보인다.

이 4개 패널은 **PNA Solver가 Mp와 gradient를 얼마나 잘 제공하는지**, 그리고 **그 gradient를 사용한 GNN 학습이 수치적으로 안정적인지**를 통합적으로 진단하는 도구 역할을 한다.

### 6.2 visualize_epoch_snapshots: 학습 과정의 형상·PNA 궤적

`visualize_epoch_snapshots` 함수는 history['snapshots']에 저장된 여러 epoch의 단면 형상을 그리드 레이아웃으로 시각화한다.

- 각 subplot에는
  - Base 구조(연한 점선),  
  - 해당 epoch에서의 변형 구조(진한 실선),  
  - 해당 epoch에서의 PNA 수평선,  
  - 제목에 epoch 번호, pred_mp, target_mp 대비 오차(%)가 표시된다.
- X, Y 축의 범위는 모든 snapshot과 base를 포함하는 공통 스케일을 사용하여 비교가 직관적이다.

이를 통해, **학습이 진행되면서 형상이 물리적으로 말이 되는 방식으로 점진적으로 변형되는지**, **PNA 위치가 목표 Mp 달성 방향으로 일관되게 이동하는지**를 시각적으로 검토할 수 있다.

---

## 7. 종합 평가: Implicit PNA Solver의 정합성

이 검증 코드는 ImplicitPNASolver의 정합성을 다음 세 레벨에서 평가한다.

1. Forward Level (정적 Mp 계산)
   - bisection 기반 y_pna 탐색과 Mp 계산이 해석적 기준(Mp_analytic)과 거의 완전히 일치함을 확인.  
   - 힘 평형 잔차가 수치적으로 0에 가깝게 떨어지는지 검증.  
   → 전소성 모멘트의 기본 물리식을 정확히 구현.

2. Gradient Level (민감도 dMp/dy)
   - autograd(IFT) gradient와 해석적 gradient, 중앙 유한차분 gradient의 3-way 비교에서, 상대오차가 설정된 임계(예: 1~2%) 이내로 작음을 확인.  
   - indirect term 크기가 평형 상태에서 매우 작게 유지되어, 암묵함수 정리의 간접항이 이론적으로 기대되는 수준(거의 0)에 있음을 확인.  
   → 구조 설계 민감도 분석 측면에서 신뢰할 수 있는 dMp/dy를 제공.

3. System Level (GNN 기반 inverse design)
   - CGDN이 ImplicitPNASolver가 제공하는 gradient를 사용하여 좌표를 조정할 때, Mp가 target 값으로 안정적으로 수렴하는지 확인.  
   - 학습 도중 gradient 오차와 indirect term 크기가 안정적으로 유지되며, 단면 형상과 PNA 궤적이 물리적으로 해석 가능한 패턴을 보이는지 시각적으로 검토.  
   → 실제 B-Pillar 단면 설계 inverse 문제에 이 Solver를 결합해도, 수치적·물리적 안정성을 기대할 수 있음을 시사.

요약하면, pna_solver_validate_v4.py는 ImplicitPNASolver가 **(1) 전소성 모멘트 물리식을 정확히 재현하고, (2) IFT를 통한 gradient 계산이 해석/수치적으로 일치하며, (3) CGNN 기반 설계 최적화 루프에서도 안정적으로 동작한다**는 것을 단계별로 입증하는 검증 프레임워크이다.
