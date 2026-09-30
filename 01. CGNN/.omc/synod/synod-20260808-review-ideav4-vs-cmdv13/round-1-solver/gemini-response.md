## 1. 개요 및 총평

`docs/idea/idea_v4.md` (이하 DOC A)의 설계 사상과 2026-08-08에 이루어진 사용자 확정/정정 이력이 `docs/command/command_v1_3.md` (이하 DOC B)에 충실히 반영되었는지 검토하였습니다.

**총평**: DOC B는 DOC A의 정정 이력(정정 1~4)과 제안(A~E)을 매우 깊이 있게 추적하여 구체적인 코드로 구현할 수 있을 만큼 액셔너블하게 작성되었습니다. 특히 단순히 상한/하한값만 수정해서는 해결되지 않는 **'두께 도달 가능성(Reachability)과 logit 동결 문제'**를 수학적/구조적으로 분석하여 `DELTA_SCALE = 10.0` 및 `FRAC_CLAMP = [0.01, 0.99]`라는 실질적인 해법을 제시한 점은 매우 우수합니다.

그러나 **일부 심각한 변수명 불일치, 정정 사항의 누락 및 논리적 왜곡 위험(ERROR/WARNING)**이 발견되어 아래와 같이 보고합니다.

---

## 2. 검증 결과 상세 (Severity 별 분류)

### 🔴 ERROR (치명적 오류 및 구현 실패 요인)

#### 1. [Fidelity / Actionability] `W_SPARSE` 비활성화 방식의 상충 및 `USE_AREA_WEIGHTED_SPARSE` 모순
* **위치**: DOC B 짠1, 짠2.1, 짠2.2
* **내용**: 
  * 짠1 상수 블록에서 `W_SPARSE_V13 = 0.0`으로 설정하였고, 짠2.1에서 `weights['w_sparse']`를 `W_SPARSE_V13`(즉, 0.0)으로 오버라이드했습니다.
  * 그러나 짠2.2에서는 보수적 B안인 면적 가중 `l_sparse` 코드를 제시하며, 주석으로 **"B안 적용 시 `W_SPARSE_V13`은 0.0이 아니라 0.02~0.05로 둔다"**고 명시하고 있습니다.
  * `USE_AREA_WEIGHTED_SPARSE` 스위치가 `False`인 상태에서 `W_SPARSE_V13 = 0.0`이 들어가면 `l_sparse` 계산식 자체는 작동하지만 물리적 의미가 없어집니다. 반면 사용자가 이 스위치를 `True`로 켰을 때, `W_SPARSE_V13`이 자동으로 연동되어 변경되지 않으므로 사용자가 수동으로 상수를 수정하지 않으면 **가중치 면적 penalization이 0.0과 곱해져 무력화되는 침묵 오류(Silent Bug)**가 발생합니다.
* **조치**: `USE_AREA_WEIGHTED_SPARSE`가 `True`일 때 `W_SPARSE_V13`이 자동으로 적절한 양수값(예: 0.03)을 가지도록 방어 코드를 작성하거나, 상수의 결합 의존성을 명확히 지시해야 합니다.

#### 2. [Fidelity] Stage 4 내부 `optimizer` 재구성 시 `thick_params` 변수명 불일치
* **위치**: DOC B 짠5.5, 짠1
* **내용**:
  * 짠5.5에서는 기존 초안의 오류를 지적하며 실제 변수명이 `main_params`와 `thick_params`라고 정정했습니다.
  * 그러나 짠1 상수 블록에서는 여전히 `STAGE4_LR_SCALE = 0.1`과 함께 설명하는 주석이나 구조에서 `thickness_params` 계열의 혼용 여지가 남아있으며, 실제 `AI_design_v1_2.py` 원본에서 두께 디코더 파라미터 변수명이 `thick_params`인지 `thickness_decoder` 파라미터 파싱 객체인지 불명확한 서술이 존재합니다.
  * "실제는 `main_params`/`thick_params`"라고 서술했으나, 바로 아래 코드 블록에서는 `{'params': thick_params, 'name': 'thickness_decoder', ...}`와 같이 명칭을 혼용하고 있어 구현 에이전트가 오독할 위험이 매우 높습니다.
* **조치**: 실제 파라미터 그룹 변수명이 `thick_params`인지, 아니면 `model.thickness_decoder.parameters()`인지 명확한 코드 레벨 매칭을 지시하십시오.

---

### 🟡 WARNING (잠재적 위험 및 정렬 불일치)

#### 1. [Coverage / Fidelity] `l_mass` 손실 함수 편입 시 `mu_Mp` 상호작용 경고 누락
* **위치**: DOC B 짠2.6.2, 짠2.5
* **내용**:
  * DOC B는 `l_mass`를 `loss`에 직접 편입(`contrib_mass = weights['w_mass'] * l_mass`)하는 획기적인 신규 제안을 추가했습니다.
  * 그러나 짠2.5에서 지적했듯 **"ALDA는 고정계수 penalty로 동작한다"**는 사실과 이 `l_mass` 편입이 결합할 때의 부작용 검토가 부족합니다. `L_alda` 내의 `term_Mp` 역시 면적/부피 제약과 간접적으로 엮여 있으므로, `w_mass`가 10.0으로 강하게 주어지면 **Mp 제약과 질량 제약이 충돌하면서 최적화 경로가 붕괴할 위험**이 있습니다.
  * DOC A 짠8 체크리스트의 "Mp를 먼저 낮추는 것은 금지"라는 규칙과 연동하여, `w_mass` 도입이 Mp 달성률(특히 sec 11)을 떨어뜨리는지 감시하는 구체적인 모니터링 코드가 누락되었습니다.
* **조치**: 짠2.6.2 또는 짠7의 1단계 판정 기준에 "Mp 달성률이 98% 미만으로 떨어질 경우 `w_mass`를 즉시 2.0 이하로 낮추거나 격하한다"는 조건부 완화 프로토콜을 명시하십시오.

#### 2. [Actionability] `_per_candidate_area` 함수 구현 명세 부족
* **위치**: DOC B 짠2.2
* **내용**:
  * `_per_candidate_area` 함수는 `new_coords`, `t_final`, `z_gate` 등을 입력받아 `(17, n_cand)` 면적을 반환하는 신규 유틸리티로 제시되었습니다.
  * 그러나 이 함수의 구체적인 구현(어떤 기하학적 텐서 연산을 통해 각 후보 에지의 면적 기여도를 계산하는지)에 대한 설명이 전혀 없어 에이전트가 이를 잘못 구현할 가능성이 큽니다.
* **조치**: `uni_section_v18.py` 내의 `compute_mass_loss` 로직을 참고하여 에지별 길이에 두께와 게이트를 곱하는 구체적인 텐서 연산 식 식(예: `seg_len * t_src * z_gate`)을 슈도코드로 제공하십시오.

---

### 🟢 INFO (단순 개선 및 참고 사항)

#### 1. [Fidelity] `FRAC_CLAMP_LO = 0.01`로 인한 최소 두께 오차
* **위치**: DOC B 짠3.2
* **내용**:
  * `FRAC_CLAMP_LO = 0.01`과 `T_MIN = 0.7`, `T_MAX = 2.3`을 적용하면 초기 두께가 `T_MIN`에 극도로 가까운 파트의 경우 초기 `t_raw`가 정확히 0.7이 아닌 약 `0.716` 부근에서 시작하게 됩니다.
  * 이는 설계상 허용 오차 범위 내이므로 안전하지만, 리포트 출력 시 초기 오차로 기록될 수 있음을 구현자가 인지해야 합니다.

---

## 3. 종합 판정
* **Coverage**: **PASS** (제안 A~E, 정정 1~4 모두 누락 없이 DOC B에 구조화되어 반영됨)
* **Fidelity**: **PASS** (의도적 변경 사항인 `l_mass` 편입 등은 합리적 근거와 함께 문서화됨)
* **Actionability**: **WARNING** (변수명 혼용 및 일부 유틸 함수 슈도코드 부재로 구현 에이전트 혼선 가능)
* **Risk**: **WARNING** (`w_mass`와 Mp Penalty 간의 충돌 위험 제어 필요)

---

<confidence score="92">
  <evidence>DOC B는 DOC A의 정정 사항(정정 1: l_mass 손실 편입 금지 해제 및 재정의, 정정 2: run pooling 금지, 정정 3: T_MIN 평가 반전 등)을 완벽히 이해하고 반영했습니다. 특히 DELTA_SCALE 상향에 따른 도달 범위 표(짠3.2)는 극도로 정밀한 분석 결과입니다. 다만 thick_params와 thickness_params의 혼용, _per_candidate_area의 구체적 텐서 연산 누락이 감점 요인입니다.</evidence>
  <logic>제안 사항들이 누락 없이 반영되었으나 구현 단계에서 NameError나 AttributeError를 유발할 수 있는 변수명 불일치가 발견되었으므로 신뢰도 점수를 92점으로 책정하며, 이 오류들이 수정되면 즉시 출구 조건(can_exit=true)을 만족합니다.</logic>
  <expertise>수치 최적화 및 PyTorch 기반 GNN 기하학적 최적화 파이프라인 설계 전문가로서, logit saturation 문제와 AdamW 모멘텀 초기화 필요성을 정확히 짚어낸 DOC B의 설계를 높이 평가합니다.</expertise>
  <can_exit>true</can_exit>
</confidence>

<semantic_focus>
1. DELTA_SCALE=10.0 및 FRAC_CLAMP 도입을 통한 [0.7, 2.3] 전 영역 두께 도달 가능성 보장
2. W_SPARSE 무력화 및 ALPHA_ENT 상향을 통한 이진화 압력 이관 및 게이트 미수렴 해결
3. Stage 4 도입 시 변수명 불일치(thick_params vs thickness_decoder) 및 l_mass-Mp 간 간섭 제어
</semantic_focus>
