제시해주신 20260401_yj.py(이하 v3)의 최신 학습 로직과 pna_solver_validate_v9.py(이하 validation)의 구조를 결합하기 위한 command.md입니다.

이 명령은 validation 모델의 단일 섹션(Single-section) 구조와 초기 노드 배치 로직을 100% 보존하면서, v3의 고급 물리 손실함수, 커리큘럼 학습, 설계 제한 영역(Keep-out) 로직만 이식하는 것을 목표로 합니다.

📄 파일명: command_migrate_loss_logic.md
Markdown
# Validation 모델에 v3 손실함수 및 학습 로직 이식 지시서

## 1. 개요
본 작업의 목적은 `pna_solver_validate_v9.py`의 물리적 검증 로직에 `20260401_yj.py`의 강력한 손실함수 체계(Collision, Smoothness, Keep-out, Curriculum Learning)를 통합하는 것입니다. 
**중요: 모든 기하학적 초기 설정(Single Section, 30 nodes 등)은 수정하지 않습니다.**

## 2. 코드 이식 단계

### Step 1: 손실 함수 모듈 이식
`yj` 코드에서 다음 함수들을 복사하여 `validation` 파일의 `calculate_mpl` 함수 아래에 추가하십시오.
* `compute_smoothness_loss_angle`: 각도 기반의 부드러운 형상 유지 로직.
* `compute_mass_loss`: 면적 기반 질량 계산.
* `compute_collision_loss_v3`: 다중 파트 간섭 방지(세그먼트 투영 방식).
* `compute_repulsive_keepout_loss`: 설계 제한 영역 척력 구현.

**[주의]** `compute_section_continuity_loss`와 `compute_shape_continuity_loss`는 층간 연속성을 위한 함수이므로, 단일 층인 validation 모델에서는 제외합니다.

### Step 2: 커리큘럼 학습 스케줄러 이식
`yj` 코드의 `get_curriculum_weights` 함수를 그대로 가져옵니다.

```python
def get_curriculum_weights(epoch, total_epochs, curriculum_ratio):
    # (v3 코드와 100% 동일하게 복사)
    ...
Step 3: 학습 루프(train_step) 업데이트
train_step 함수를 다음과 같이 재구성합니다.

가중치 정의: weights = {'w_phys': 10.0, 'w_smooth': 0.1, ...} 딕셔너리를 학습 함수 인자로 추가.

Loss 계산부 수정:

Python
# 통합 예시
l_smooth = compute_smoothness_loss_angle(new_coords, edge_index, edge_attr)
l_mass = compute_mass_loss(...)
l_collision = compute_collision_loss_v3(new_coords, part_ids, section_ids, parts_order_in_sections={0: [0, 1, 2]})
l_keepout = compute_repulsive_keepout_loss(new_coords, section_ids, keepout, height_multiplier=s_keepout)

total_loss = (weights['w_phys'] * l_phys * s_phys + weights['w_smooth'] * l_smooth * s_smooth + ...)
Step 4: 설계 제한(Keep-out) 영역 적용
pna_solver_validate_v9의 메인 실행부(if __name__ == "__main__":)에 다음 구성을 추가합니다.

Python
# 단일 섹션(0)에 대한 Keep-out 영역 정의
keepout = {
    0: [(x1, x2, y1, y2), (x3, x4, y3, y4)] # 실제 좌표 값 입력
}
3. 구현 지침
데이터 인터페이스: validation 코드의 데이터가 part_ids, section_ids 등을 포함하고 있는지 확인하십시오. 없다면 Data 객체 생성 시 section_id(0으로 고정)와 part_id 피처를 추가하십시오.

Curriculum 적용: training 루프 내에서 get_curriculum_weights를 호출하여 각 에폭마다 가중치를 동적으로 곱해줍니다.

Ghost Layer 관련 코드 제외: 현재 validation은 단일 섹션이므로 dynamic_thickness_v2.md에서 구현한 층별 삭제 로직은 제외하고, 기본적인 손실 함수들만 적용하여 안정적인 학습이 이루어지게 하십시오.

4. 기대 효과
기존: 단순히 PNA만 검증하던 모델.

변경 후: 충돌 방지, 각도 평활화, 특정 영역 회피(Keep-out)가 적용된 고도화된 최적화 모델로 업그레이드됨.


---

### 적용 시 체크포인트
1. **`compute_collision_loss_v3` 호출 시:** validation 코드는 1개 섹션이므로 `parts_order_in_sections` 인자를 전달할 때 `{0: [0, 1, 2]}`와 같이 단일 섹션 리스트로 전달해야 에러가 발생하지 않습니다.
2. **`keepout` 좌표:** validation 모델의 초기 형상(`coords`) 범위를 확인하여 `keepout` 사각형이 노드들의 좌표 범위 내에 위치하도록 설정하십시오.
3. **가중치(Weights):**
   처음에는 `w_phys`를 높게 설정하고 나머지 `s_` 가중치들이 붙는 커리큘럼 방식을 사용하여 초기에 형상이 붕괴되지 않도록 하십시오.