# idea_v1.md — 17층 B-pillar 3D 형상 구조 최적화 구체화 아이디어

기반: `idea_v0.md` (요구사항 1~7), `initial_section.md` (17섹션 등방 스케일링/Mp 검증),
`uni-section/code/uni_section_v18.py`, `AI_design_v0.py`.

`/synod idea` 세션 요지: Claude(Validator) conf 78, Gemini(Architect, pro/high) conf 92→95(Critic).
**OpenAI(gpt4o/o3)는 이번 세션 동안 API rate limit(재시도 3회 모두 실패)으로 응답을 받지 못해 제외됨** —
Prosecutor/Explorer 관점 없이 Claude+Gemini 2자 합의로 도출된 초안이므로, 실제 구현 전 반대 관점
검증(엣지 케이스, 학습 불안정성)을 별도로 한 번 더 거치는 것을 권장한다.

---

## 0. 가장 중요한 발견: 현재 코드는 두께 불변을 보장하지 않는다

`uni_section_v18.py`의 `CGDN.forward()` (line 358-371)를 실제로 읽으면:

```python
part_ids_local    = x[:, 4].long()
section_ids_local = x[:, 5].long()
...
composite_key = section_ids_local * max_parts + part_ids_local   # ← (section, part) 조합으로 그룹핑
_, inverse = torch.unique(composite_key, return_inverse=True)
...
group_mean  = group_sum / group_count.clamp(min=1)   # 그룹(=한 섹션의 한 파트) 내부만 평균
delta_t_part = group_mean[inverse].unsqueeze(-1)
```

즉 현재 구조는 이미 `section_id`를 노드 feature(x[:,5])로 갖고 있고 pooling도 하고 있지만,
**pooling 단위가 "섹션별 파트"이지 "전체 섹션에 걸친 동일 파트"가 아니다.** 이 상태로 17섹션을
그대로 이어붙이면 outer/inner part의 두께가 섹션마다 독립적으로 학습되어 요구사항 4("outer part,
inner part는 모든 section에 걸쳐서 연결되기 때문에 절대 두께가 바뀌지 않음")를 정면으로 위반한다.

**필요한 최소 수정 (1줄 개념, 구현은 §3 참고):**
```python
composite_key = logical_part_id   # section_id 항을 제거 — 연속 파트는 전 섹션에서 동일 key
```
연속 파트(outer/inner)는 `logical_part_id`가 17섹션 전체에서 동일해야 pooling이 전 섹션에 걸쳐
일어나 두께가 자동으로 하나의 스칼라로 수렴한다. 반대로 재등장 patch는 새 `logical_part_id`를
받아야 독립적으로 pooling된다 — 이것이 요구사항 4와 5를 **동시에** 만족시키는 유일한 지점이다.

---

## 1. 실행 아키텍처: Autoregressive Sliding-Window Curriculum (ASWC)

- 생성 순서는 요구사항 1대로 17→1 top-down 고정.
- 17섹션을 한 번에 하나의 계산 그래프로 backprop하면 메모리/그래디언트 불안정 위험이 크고,
  완전 순차(섹션별 독립 학습)는 인접 연속성(요구사항 6)을 반영할 수 없다. 따라서 **윈도우 크기
  W=3의 sliding window**로 절충한다: 섹션 k를 최적화할 때 [k, k+1] (이미 확정된 위쪽 1개 섹션)까지만
  그래디언트를 흘리고, 위쪽 섹션의 좌표는 `detach()`한 상태로 continuity loss의 타깃으로만 사용한다.
  (완전한 3-섹션 윈도우로 양방향 backprop하면 이미 확정된 섹션까지 흔들릴 위험이 있으므로, 위쪽은
  detach, 아래쪽만 학습 대상으로 삼는 비대칭 윈도우를 권장.)
- 3단계 curriculum:
  1. **Global Coarse Fit**: 좌표는 freeze, `thickness_decoder`와 스케일 팩터만 목표 Mp 트렌드에
     맞춰 대략 학습 (§0 수정 적용 후 — 이 단계에서 연속 파트 두께가 수렴하는지 먼저 확인).
  2. **Sequential Refinement**: 17→1 순서로 CGDN을 섹션별로 학습하되, thickness_decoder 중 연속
     파트에 해당하는 부분은 **이 단계 종료 시점에 freeze**(요구사항 4, §3 참고).
  3. **Global Relaxation**: 좌표(coordinate decoder)만 전체 재학습, thickness_decoder는 계속 freeze.

---

## 2. Part 삭제/재등장의 그래프 인코딩

요구사항 5("17th에 있던 reinforce part가 16th에서 삭제, 15th에서 재등장하면 새 파트가 됨")를
동적 그래프가 아니라 **사전 정의된 rule-based 스케줄**로 처리한다 (Claude 지적, Gemini도 "Disputed
Claims"에서 동의): 학습 루프가 매 스텝 topology를 바꾸는 것은 정적 배치 그래프(PyG 등) 구조와
충돌하므로, 전처리 단계에서 17섹션 × part 존재/삭제 여부를 사람이 먼저 정의(또는 목표 Mp 곡선에서
규칙적으로 파생)한 뒤, 그 결과로 각 섹션의 `part_ids`/`logical_part_id` 텐서를 미리 구성한다.

- `logical_part_id` 할당 규칙:
  - Outer Hat(0), Inner Plate(1), Inner Hat(2): 17섹션 전체 동일 ID — 항상 pooling 그룹에 포함.
  - Patch1/Patch2류: 연속 구간에서는 동일 ID 유지, 삭제→재등장 시 **새 ID 발급**(예: 원래 3이면
    재등장 시 5, 6... 순차 부여). ID 발급은 전처리 단계 규칙표에서 관리 (JSON/CSV로 저장 권장,
    코드에 하드코딩 금지 — 목표 Mp 랜덤 샘플링(§4)마다 재실행 가능해야 함).
  - `build_collision_spec()` (line 559)이 이미 `section_ids`별로 파트 쌍을 순회하는 구조이므로,
    신규 ID 발급 시 이 함수의 입력 `part_ids`/`section_ids`만 갱신하면 충돌 검사 로직은 그대로 재사용 가능.

---

## 3. 두께 불변 구현 (CGDN 최소 수정안)

`CGDN.forward()`의 pooling 키를 파트 카테고리에 따라 분기한다:

```python
# continuous_part_ids: 학습 시작 전 고정된 set (예: {0, 1, 2})
is_continuous = torch.isin(part_ids_local, continuous_part_ids_tensor)
composite_key = torch.where(
    is_continuous,
    logical_part_id_local,                                  # 전 섹션 공통 pooling
    section_ids_local * max_parts + logical_part_id_local,  # 신규/비연속 파트는 섹션별 독립
)
```

- `logical_part_id_local`은 §2에서 정의한 전처리 산출물(연속 파트는 물리 part_id와 동일, 재등장
  patch는 새 ID)을 담은 노드 feature로 별도 컬럼(x[:, N])에 추가한다.
- Stage 2(Sequential Refinement) 종료 시, `thickness_decoder`의 파라미터 중 연속 파트에 대응하는
  출력 채널(혹은 그 채널로 흐르는 gradient path)을 `requires_grad_(False)`로 고정. `S_PROTECT`
  (line 183, 현재 `[0, 1, 2]`)가 이미 pruning 보호용으로 존재하므로 이 리스트를 재사용해 "protect ==
  freeze 대상"으로 삼을 수 있다.
- Stage 3(Global Relaxation)에서는 freeze가 유지되므로 좌표만 움직이고 두께는 절대 변하지 않는다
  — 이것이 Gemini critic 결과(conf 95)와 Claude 지적이 합의한 지점이다.

---

## 4. 목표 Mp 분배와 연속성/자유도 균형

- 요구사항 7의 "±50% 랜덤"을 섹션별 독립 샘플링하면 이웃 섹션끼리 비현실적으로 급변할 수 있다.
  **공간 상관 Gaussian Process**로 17섹션에 걸친 편차 벡터를 샘플링:
  `α ~ GP(0, Σ), Σ_ij = σ² exp(-(i-j)²/2ℓ²)`, σ≈0.25(±50%에 대응), ℓ≈3(3~4섹션 스무딩).
  `target_Mp(k) = initial_Mp(k) × (1 + α_k)` — `initial_Mp(k)`는 `initial_section.md` §2 표의 실측값
  (k=17: 21,637,199 → k=1: 55,297,613, 이미 단조증가 확인됨) 사용.
- 형상 연속성 loss는 정확한 좌표 일치가 아니라 **완화된 Chamfer/hinge 형태**로 설계해 "과도한 제약
  금지" 요구를 만족한다:
  `L_shape(k,k-1) = max(0, ChamferDist(P_k/s(k), P_{k-1}/s(k-1)) - δ)` (δ≈2mm, 스케일 정규화 후 비교).
  가중치는 학습 초반 크게(형태 붕괴 방지) → 후반 sigmoid decay로 축소(로컬 Mp 만족 우선).
  단, `AI_design_v0.py`에 이미 `compute_section_continuity_loss`/`compute_shape_continuity_loss`
  (line 642, 665)가 구현되어 있으므로, **새 loss를 만들기 전에 이 두 함수가 5-part hat 구조에 그대로
  적용 가능한지 먼저 확인**하고, 안 되면 위 Chamfer-hinge로 대체/확장한다 (Gemini critic: "Disputed /
  Unverified" — 코드 감사 우선).
- Mp 목표의 실현 가능성은 각 섹션에 대해 `compute_edge_mp_pna`/`calculate_mpl`을 재호출해 사후
  검증하는 폐루프를 학습 루프 바깥에 둔다(사전 커브 검증, 매 epoch는 아님).

---

## 5. 미해결/추가 검증 필요 (2자 합의 세션의 한계)

1. `AI_design_v0.py`와 `uni_section_v18.py`의 part 구성이 실제로 동일한지 미확인 — 다르면 continuity
   loss 이식 시 part_id 매핑 재정의 필요.
   -> 다른게 맞음 둘다 신경쓰지 말고 C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\initial_section.md 이거 따르면 됨.
2. Sliding window의 비대칭 detach 전략이 실제로 gradient 누적 오류 없이 동작하는지는 최소 재현
   실험(2~3섹션)으로 먼저 검증 필요.
   -> 꼭 필요하면 하는데, 그렇지 않으면 full code로 직행
3. OpenAI(Explorer/Prosecutor) 관점 부재 — 특히 "GP 스무딩된 target Mp가 실제로 collision loss와
   충돌하는 극단 케이스"는 이번 세션에서 검증되지 않았다. 구현 착수 전 재시도 권장.