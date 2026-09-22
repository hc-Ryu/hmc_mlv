# command_v12.md — 다단계 위상 경량화 및 그래프 수술(Cascade Optimization) 아키텍처

**목적:** 
기존 v11의 '존재 게이트(L0) 기반 위상 경량화'를 계승하되, 부재 삭제 시 발생하는 기하학적 허공(Gap) 문제를 해결하기 위해 **"중단-수술-재시작(Stop-Snap-Resume)"** 기반의 연쇄 최적화(Cascade Optimization) 파이프라인을 구축한다.

---

## 1. 핵심 설계 철학 (Core Philosophy)

1.  **완전한 물리적 실현성 (Feasibility):** 프루닝된 부재가 남긴 기하학적 빈 공간을 방치하지 않고, 인접 부재를 밀착(Snap)시켜 양산 및 후처리가 즉시 가능한 폐단면(Closed Section)을 도출한다.
2.  **연쇄 최적화 (Cascade Optimization):** 형상이 급변하는 '접합(Snapping)' 이벤트 직후의 물리적 쇼크(Mp 급락 등)를 방지하기 위해, 수술된 형상을 새로운 초기값(Base)으로 삼아 학습 환경을 **완전 초기화(Hard Restart)** 한다.
3.  **영구적 삭제 (Hard Pruning):** 한 번 도태 판정을 받은 부재는 재시작 시 영구적으로 배제되며 다시 살아나지 않는다.

---

## 2. 2-Cycle 학습 워크플로우 (Stop-Snap-Resume)

전체 최적화 루프는 단일 `run_training`이 아닌, 상태를 감시하는 **아우터 루프(Outer Loop)** 체제로 개편된다.

### Cycle 1: 위상 탐색 및 프루닝 (Topology Exploration)
*   **작동 방식:** v11과 동일하게 노드, 두께, 존재 게이트($z$)를 동시 최적화하며 희소화 어닐링($\lambda_0$)을 진행한다.
*   **중단 트리거 (Trigger):** 특정 부재 $i$의 존재 확률 $z_i$가 임계값($\tau = 0.1$) 이하로 떨어져 $N$ 에폭(예: 20 에폭) 동안 연속 유지되면, 해당 부재를 '완전 삭제 대상'으로 확정하고 **Cycle 1 학습을 즉시 강제 종료(Break)** 한다.

### Event: 그래프 수술 및 기하학적 접합 (Graph Surgery & Snapping)
Cycle 1이 중단되면 현재의 `new_coords`를 바탕으로 다음의 수술을 진행한다.
1.  **부재 삭제 분기:**
    *   **Case A (외곽 부재 삭제):** 최외곽 부재(예: c)가 삭제될 경우, 남은 부재(a, b)는 이미 결합되어 있으므로 별도의 좌표 이동 없이 c만 데이터에서 소거한다.
    *   **Case B (중간 부재 삭제):** 샌드위치 구조의 중간 부재(예: b)가 삭제될 경우, 공중에 뜬 외곽 부재(c)의 y좌표를 안쪽 부재(a)의 플랜지 y좌표 위치로 강제 평행이동(Shift)하여 밀착시킨다.
2.  **데이터 리빌딩 (Data Rebuild):** 삭제된 부재의 노드와 엣지를 모델 및 `Data` 객체에서 완전히 덜어낸 새로운 그래프 데이터를 생성한다 (차원 축소). 

### Cycle 2: 재최적화 (Hard Restart & Refinement)
*   **완전 초기화:** 수술이 완료된 밀착 도면을 새로운 `base_coords`로 설정한다. 옵티마이저(`AdamW`), 에폭(Epoch) 카운터, 커리큘럼 스케줄러를 **모두 0으로 리셋**한다.
*   **충돌 환경 리셋:** 변경된 형상과 부재 수를 바탕으로 `build_collision_spec()`을 다시 호출하여 앵커와 간극(Clearance) 기준을 새로 짠다.
*   **재학습:** 새로운 환경에서 타겟 $M_p$를 향해 다시 학습을 시작한다. (이때 초기 커리큘럼이 다시 적용되므로 자연스러운 웜업(Warm-up)과 기하학적 정렬이 이루어진다.)

---

## 3. 손실 함수 및 아키텍처 세부 명세 (v11 계승)

Cycle 1과 Cycle 2의 내부를 도는 모델(CGDN)은 v11에서 확립된 다음의 원칙을 그대로 따른다.

*   **존재 게이트 (L0 Hard-Concrete):** $t_{final} = z_i \cdot t_{raw}$ 구조를 유지하여 $z_i \to 0$일 때 완전한 0 두께를 보장한다. (단, Cycle 2에서는 이미 삭제된 부재는 아예 네트워크에 입력되지 않는다.)
*   **질량 손실 (One-sided Mass Loss):** 
    $$L_{mass} = \max(0, \frac{Area - Target}{Target})^2 + \lambda_0(\text{epoch}) \cdot \text{mean}(z)$$ 
    단방향 페널티와 희소화 보상을 결합하여 부재 삭제의 동력을 제공한다.
*   **충돌 위반 게이팅 (Collision Violation Gating):** 
    $$Violation_{ij} = z_i \cdot z_j \cdot (\text{relu}(-gap_{ij}))^2$$
    사라져가는 부재가 억울하게 충돌 페널티를 발생시키지 않도록 방어한다.
*   **필수 부재 보호:** 뼈대를 구성하는 `Outer Hat`, `Inner Plate` 등은 $z=1$ 로 고정하여 단면의 폐합(Closed-loop)을 영구적으로 유지한다.

---

## 4. 구현 가이드 (Implementation Notes)

*   **`run_cascade_training` (아우터 루프 래퍼 함수 신설):** 
    기존 `run_training`을 내부에 품고, 반환된 상태(Status)를 체크하여 수술(Surgery)과 재시작(Restart)을 통제하는 마스터 함수를 신설한다.
*   **`perform_graph_surgery(data, dead_part_id)` 유틸리티 구현:**
    PyTorch Geometric의 `Data` 객체를 입력받아, 특정 `part_id`에 해당하는 노드와 엣지를 마스킹하여 삭제하고, 조건에 따라 남은 파트의 y좌표를 Shift하는 로직을 독립된 함수로 구현한다. 
*   **모델 동적 인스턴스화:** 
    부재 수가 5개에서 4개(또는 3개)로 줄어들면 `CGDN` 모델의 `n_parts` 파라미터도 변해야 하므로, 수술 후에는 모델 객체 자체를 새로 인스턴스화(`CGDN(n_parts=4, ...)`)하여 이전 학습의 잔재(Momentum)를 완벽히 끊어낸다.