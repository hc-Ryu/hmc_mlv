# 단일 층(Top Section) 분리 및 2D 모델화 코드 수정 지시서

**[목적]**
`20260401_yj.py` 파일의 17개 Section 중 최상단(Section 16)의 데이터만 추출하여, `num_sections = 1`인 단일 층 2D 최적화 모델로 코드를 단순화합니다.

**[작업 대상 파일]**
`C:\Users\user\Documents\GitHub\hmc_mlv\CGNN\20260401_yj.py`

**[주의 사항]**
* 이 문서에 명시된 부분 외의 모델 아키텍처(CGDN), 물리 솔버(ImplicitPNASolver), Loss 함수 로직은 **절대 수정하지 않습니다.**
* Section 16은 원래 전체가 고정(`fix_x=1, fix_y=1`)되어 있었으나, 최적화를 위해 양 끝 플랜지만 고정되는 일반 경계조건으로 풀어줍니다.

---

## 1. 데이터 구성(Data Construction) 딕셔너리 수정
Section 16에 해당하는 파트 구성과 목표 $M_p$만을 남기고 전부 삭제(또는 주석 처리)합니다. 인덱스는 `0`으로 재할당합니다.

**[수정할 코드 블록 - In[7] Data Construction]**
```python
# 기존 0~16으로 정의된 parts_in_sections, parts_order_in_sections를 단일 층으로 변경
parts_in_sections = {
    0: [0, 4, 3, 2],  # 기존 Section 16의 파트 구성
}

parts_order_in_sections = {
    0: [0, 4, 3, 2],  # 기존 Section 16의 파트 계층 순서
}

num_sections = 1  # 17에서 1로 변경
```

## 2. 노드(Node) 좌표 생성 로직 단순화
기존의 복잡한 층간 보간(Interpolation) 로직을 제거하고, 상단부(upper_section)의 좌표를 그대로 사용하도록 강제합니다.

[수정할 코드 블록 - 노드 생성 for 루프]
기존 for i in range(num_nodes): 내부의 좌표 할당 로직을 아래와 같이 교체합니다.

```Python
for i in range(num_nodes):
            # 오직 upper_section(기존 최상단)의 좌표만 사용
            if part == 1:
                x_coord = upper_section[4][i][0]
                y_coord = upper_section[4][i][1]
            else:
                x_coord = upper_section[part][i][0]
                y_coord = upper_section[part][i][1]

            # 기존 Section 16은 전체가 고정되어 있었으나, 최적화를 위해 플랜지만 고정
            fix_x = 1.0 if (i in [0, 1, 2, 17, 18, 19]) else 0.0
            fix_y = 1.0 if (i in [0, 1, 2, 17, 18, 19]) else 0.0

            t_val  = 1.5 if part != 1 else 2.0
            fy_val = 1500.0 if part != 2 else 1200.0

            ## [x, y, fix_x, fix_y, part_id, section_id, t, fy]
            x[current_idx] = torch.tensor([x_coord, y_coord, fix_x, fix_y,
                                            float(part), float(section), t_val, fy_val])
            node_registry[(section, part, i)] = current_idx
            current_idx += 1
```
## 3. 층간 연결(Inter-section Edge) 제거 및 Keep-out Zone 수정
3D 연결성을 담당하던 Z축 엣지 생성 루프를 주석 처리하고, 제한 구역도 단일 층으로 변경합니다.

[수정할 코드 블록 - 엣지 및 Keepout]

```Python
# Inter-section (종방향: 3D 연속성) 엣지 생성 로직 삭제 또는 주석 처리
# for section in range(num_sections - 1):
#     next_section = section + 1
#     ... (생략) ...
#             add_edge(u, v, part_id=part, edge_type=1.0)

# Keep-out Zone을 단일 층(0)으로 변경 (기존 Section 16의 제약조건)
keepout = {
    0: [(0.0, 600.0, -60.0, -120.0), (0.0, 600.0, 120.0, 180.0)]
}
```

## 4. 학습 목표 (Training Target) 수정
기존 17개의 타겟 모멘트를 1개로 줄입니다.

[수정할 코드 블록 - In[8] Training]

```Python
# 기존 17개 섹션의 target_mps 딕셔너리를 단일 층(기존 16층 값)으로 대체
    target_mps = {
        0: 55645928  # 기존 Section 16의 목표 전소성 모멘트
    }
```

## 5. 시각화 (Visualization) 코드 호환성 보완
시각화 파트에서 하드코딩된 Section Number들을 동적으로 1개 층에 맞게 조절합니다.

[수정할 코드 블록 - 2D Subplot 시각화 루프 설정]

```Python
# 기존 하드코딩된 section_order 변경
# section_order = [5]  <- 이 부분을 아래와 같이 수정
section_order = list(range(num_sections - 1, -1, -1)) # [0]이 됨
n_rows = len(section_order)

fig, axes = plt.subplots(n_rows, 2, figsize=(48, 16 * n_rows), squeeze=False)

# ... 중략 ...
# draw_section_on_ax 호출 시 axes 배열 인덱싱 주의 (squeeze=False 적용)
    draw_section_on_ax(
        ax_base=axes[row_idx, 0],
        ax_def=axes[row_idx, 1],
        coords=base_coords_sec,
        # ... 후략
```