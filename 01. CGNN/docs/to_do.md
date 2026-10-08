# TO-DO

## [2026-10-08] run별 두께가 CSV export 단계에서 평균으로 희석되는 문제

### 배경
- `AI_design_v23_ref.py`는 학습 중 candidate 파트(Patch1–4)가 일부 섹션에서 삭제되면,
  삭제 섹션을 경계로 **연결 run마다 두께를 따로 학습**한다 (Phase C / Stage 4, `compute_segment_ids()` + `run_key` pooling, L2248–2271).
- 예: v23_ref의 Patch4 = run0(sec 0–1) 2.497, run1(sec 5–6) 2.500, run2(sec 8–14) 2.500 mm.
  v23_ref_yb는 Patch1(run 3개), Patch2·Patch3(run 2개), Patch4(run 2개)도 나뉨.

### 문제점
1. **export에서 run별 두께가 하나로 평균됨**
   - `export_final_section_csv()` (L3286–3296)는 CSV 형식이 파트당 두께 1개(구분 행 `point_idx=0`의 `t_mm`)만 담을 수 있어서,
     생존 노드 전체 t_raw의 **노드 수 가중 평균**을 기록한다. 섹션 수가 많은 run이 값을 지배한다.
   - run 간 편차 > 1e-3 mm이면 경고만 출력하고 그대로 저장한다 (실패 처리 아님).
2. **리포트와 후속 해석의 물성 불일치**
   - 리포트의 Mp/My(hard)는 모델 내부의 run별 t_raw로 계산된다.
   - CSV를 쓰는 후속 단계는 모두 평균 두께를 받는다:
     `02. post_processing`, `03. uq` 샘플링, FEM 입력, `check_part_overlap()`, `check_flange_gap_from_csv()`.
3. **기하 불일치**
   - 좌표는 run별 t_raw 기준 접합 오프셋 (t_a+t_b)/2로 생성된다. 여기에 평균 두께를 적용하면
     얇은 run은 겹침, 두꺼운 run은 간극으로 판정될 수 있다 (예: 1.0 / 2.5 mm → 1.75 mm).
4. **UQ 설계변수가 파트 단위**
   - `03. uq/sample/bpillar_samples_v23.py`는 `t_Patch4` 하나(기본값 = 평균 2.499328)를 Patch4가 있는 모든 섹션에 동일하게 적용한다.
   - run이 실제로 별도 부품(별도 블랭크)이면 두께 공차도 독립 → 변수를 run별로 나눠야 한다.
5. **(참고) 학습 중 평균 구간**: run 내부 섹션끼리는 하나의 두께로 평균되고 (run = 부품 1개, 의도된 동작),
   Phase A/B(분할 전)에는 전 섹션 파트 단위 pooling.

### 현재 v23 영향
- 거의 모든 run이 T_MAX = 2.5 mm 근처에 포화(차이 ~0.003 mm) → **v23 결과에는 실질적 영향 없음**.
- 구조적으로 막혀 있는 것은 아니므로, 두께가 상한에서 벗어나는 버전부터는 문제가 드러난다.
- 별도 확인 필요: 대부분의 두께가 T_MAX에 붙어 있음 → 상한이 사실상 활성 제약 (T_MAX 설정/두께 페널티 재검토).

### 해야 할 일
- [ ] **CSV에 run별 두께 기록** — 둘 중 하나 선택
  - (a) 데이터 행의 `t_mm`(현재 항상 0.0)에 노드/run별 두께 기록, 리더는 행 값이 있으면 우선 사용
  - (b) run마다 별도 파트 블록으로 export (예: Patch4_run0/1/2) — 기존 리더 형식과 호환,
        단 블록 순서로 파트를 식별하는 코드(`PART_NAMES` 7개 고정, `FLANGE_PAIRS`/`FOLLOWERS` 등) 수정 필요
  - 권장: (b)
- [ ] CSV를 읽는 코드 일괄 수정: `AI_design_v23*.py`의 검사 함수(`_read_template_csv_by_pid`, `check_part_overlap`, `check_flange_gap_from_csv`),
      `02. post_processing/post_processing.py`, `03. uq/sample/bpillar_samples_v23.py`, FEM 입력 생성부
- [ ] **UQ 설계변수 run별 분리**: `t_Patch4_run0/1/2` 등 → v23_ref 기준 N = 24, PDD(m=4, S=1) 계수 97개, 샘플 291개(×3)
      (v23_ref_yb는 run 수가 더 많으므로 별도 계산)
- [ ] **export 경고 → 실패 처리**: run 간 편차가 기준(예: 0.05 mm)을 넘으면 export 중단 (조용한 평균 방지)
- [ ] 수정 후 검증: CSV 기반 Mp/My 재계산 결과가 리포트 hard 값과 일치하는지, 접합 간극 검사 통과하는지 확인
- [ ] T_MAX 포화 문제 별도 검토
