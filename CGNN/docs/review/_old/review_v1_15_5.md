# review_v1_15_5.md — 플랜지 접합부 공차(Gap) 검토

- 검토 대상: `AI_design_v1_15_5.py`, `reports/v1_15_5/final_section_v1_15_5.csv`, `reports/v1_15_5/AI_design_v1_15_5_report.md`
- 세션: `/synod review` (fairy_tail 프로젝트, Gemini flash/thinking-high conf95 + OpenAI o3/reasoning-medium conf87, Claude 코드 대조 검증)
- 초점: 결과 형상에서 파트 플랜지(FLANGE_PAIR_SPEC 20쌍) 접합부의 공차 발생 여부

## 1. 개요

v1.15.5는 v1.15.4 대비 물리/학습 로직 변경 없이 입력 좌표 CSV만 `initial_section_v2.csv → initial_section_v4.csv`로 교체한 버전이다. 독스트링(L9-26)은 v2.csv의 심각한 플랜지 표면 간극 문제(340개 조합 중 328개, 96.5%가 0.01mm 허용오차 초과)를 "v3.csv"가 floor별 강체 평행이동으로 해소했다고 서술하지만, 실제 코드가 로드하는 파일은 v4.csv다. 이 검토는 그 간극(gap)이 최종 형상에서 실제로 허용오차 이내인지, 그리고 이를 검증하는 파이프라인이 그 주장을 뒷받침하는지를 점검한다.

Gemini(Architect)·OpenAI(Explorer) 두 모델이 독립적으로 동일한 3가지 핵심 결함을 지적했고, Claude가 실제 소스 코드 대조 및 CSV 좌표 직접 재계산으로 검증했다. 1차 검증([검증 부재] 계열)에 이어 사용자 요청에 따라 **"왜 공차가 애초에 발생하는가"의 근본 원인**을 2차로 추적했으며, 그 결과 WARNING 2의 판정이 정정됐다(§2-2 참고).

## 2. 발견 사항

### 2-1. 검증 계층의 결함 (1차 검토)

### [ERROR] 1. 최종 형상의 플랜지 간극(양의 gap)을 검증하는 코드가 없음 — 확증됨

- `check_part_overlap()`(L946-1013)은 같은 floor에서 x가 근접한 두 파트 노드의 ±t/2 밴드가 겹치는 **침투 깊이**(`depth = min(up_i,up_j) - max(lo_i,lo_j)`)만 계산해 `depth > tol_mm`일 때만 위반으로 집계한다(L993-994). `depth`가 음수(밴드가 서로 떨어져 있음, 즉 양의 gap)인 경우는 애초에 위반 조건에 걸리지 않는다 — **접합부가 아무리 벌어져 있어도 이 함수는 통과시킨다.**
- 학습 중 계산되는 `compute_flange_penetration_diag()`(L1047-1069)는 `flange_gap0`을 반영한 실제 관통량을 계산하지만, 함수 docstring이 명시하듯 **진단 전용**(`detach()`, loss/역전파 미반영)이며 콘솔 epoch 로그(L3139)에만 출력된다. 최종 export 이후 이 값을 재계산하거나 하드 assert로 거는 코드는 없다(grep 결과 `compute_flange_penetration_diag` 호출은 L2842 한 곳, 학습 루프 내부뿐).
- 따라서 `AI_design_v1_15_5_report.md` §5의 "겹침 없음 — 모든 파트쌍이 허용 오차 이내"는 **침투(negative gap) 없음만 의미**하며, 플랜지가 목표 두께만큼 정확히 밀착됐는지(positive gap ≤ 0.01mm)는 이 리포트로 전혀 보장되지 않는다. 학습 손실이 `flange_gap0`을 목표로 유지하려는 soft 페널티일 뿐 hard 제약이 아니므로, 수렴이 불완전하면 정지 상태의 잔여 간극이 검증 없이 최종 CSV에 그대로 남을 수 있다.
- **권장**: export 직후 `FLANGE_PAIR_SPEC` 기반으로 최종 CSV의 실제 표면 거리와 목표 두께 합(t_a/2+t_b/2+flange_gap0)의 편차를 계산해 `FLANGE_ASSERT_TOL_MM` 이내인지 검사하는 `check_flange_gap()`을 `check_part_overlap()`과 나란히 추가하고, 결과를 리포트 별도 섹션으로 append할 것.

### [WARNING→INFO 정정] 2. `v3.csv` 서술과 실제 로드 파일 `v4.csv`의 불일치 — 실제 회귀는 데이터로 기각됨

- 독스트링 L9, L12-23은 오직 "v3.csv"만 플랜지 간극 수정본으로 서술한다. 그러나 실행 코드 L488-489는 `initial_section_v4.csv`를 로드한다(grep 확인, v3.csv에 대한 로드 코드는 존재하지 않음). 이 자체는 여전히 문서-코드 불일치(구성 관리 미흡)다.
- **2차 검증(직접 재계산)**: `initial_section_v4.csv`를 `identify_mating_flange_pairs()`와 동일한 로직(파트/floor/point_idx 그룹핑 + `gap0 = dist0 - (t_a+t_b)/2`)으로 직접 파싱·재계산한 결과, v3.csv와 **완전히 동일**했다 — 수정 대상 289쌍(Outer-Plate/Plate-Inner/Plate-Patch1/Plate-Patch2)은 전부 정확히 `0.0000mm`, Inner-Patch2 3-way 샌드위치 51쌍(17floor×3쌍)은 정상적인 `1.6000mm`(Plate 두께, 결함 아님)로 v2.csv(328/340 위반)의 문제가 실제로 해소된 상태임을 확인했다. 즉 v4.csv는 v3.csv의 기하학적 내용을 그대로 유지한 채 파일명만 바뀐 것으로 판단된다 — **회귀는 발생하지 않았다.**
- **권장**: 문서-코드 불일치 자체는 여전히 구성 관리 리스크이므로 독스트링을 v4.csv 기준으로 갱신(v3=v4임을 명시)할 것을 권장하되, 심각도는 ERROR/WARNING이 아닌 INFO(문서 정합성 문제)로 하향한다.

### [WARNING] 3. `FLANGE_TAPER_WEIGHTS` 경계 노드 감쇠 한계

- `FLANGE_TAPER_N=2`, `FLANGE_TAPER_WEIGHTS=(1.0, 2/3, 1/3)`로 플랜지 노드 좌우 2개 이웃까지 오프셋을 전파한다. 그러나 파트 양 끝단(첫/마지막 point_idx) 근처의 플랜지 노드는 taper 범위 밖으로 나가는 이웃이 존재하지 않아, 코드가 명시하듯(L692-696 주석, 범위 밖은 가중치 0) 사실상 taper 폭이 짧아진다.
- 이는 학습(soft-mode)에서의 근사 한계이며, §1의 최종 검증 부재와 겹치면 경계 근처 노드에서 국소적 잔여 간극이 검증 없이 넘어갈 수 있다.
- **권장**: §1의 `check_flange_gap()`을 point_idx 단위로 세분화해 경계 노드에서 유독 큰 편차가 나오는지 별도로 리포트하면, 이 taper 한계가 실제 문제인지 판별 가능하다(현재는 진단 지표조차 없어 확인 불가).

### [INFO] 4. 리포트 문구가 검사 범위를 과대 대표

- §5의 "모든 파트쌍이 허용 오차 이내"라는 문구는 독자에게 플랜지 접합 품질까지 보증된 것으로 오인시킬 수 있다. `append_overlap_section_to_report()`(L1016-1044)가 생성하는 이 문구를 "겹침(interpenetration) 검사 결과이며 플랜지 간극(gap)은 별도"로 명확히 하는 것을 권장한다.

### 2-2. 공차가 발생하는 근본 원인 (2차 추적 — 사용자 요청)

"검증이 안 되고 있다"는 것은 공차가 **왜** 생기는지에 대한 답이 아니라, 생겨도 걸러지지 않는다는 파이프라인 결함일 뿐이다. 아래는 실제로 공차가 어디서 발생/보존되는지에 대한 메커니즘 분석이다.

**[핵심] 학습이 간극을 0으로 "만드는" 것이 아니라, 입력에 있던 간극을 그대로 "보존"하도록 설계되어 있다.**

`forward()`의 플랜지 오프셋 로직(L1287-1329)은 손실 함수 기반 gradient 수렴이 아니라, 매 forward pass마다 다음을 **결정론적·해석적(closed-form)**으로 계산한다:

```
gap0 = dist0(초기 좌표) - (t_a0+t_b0)/2      # identify_mating_flange_pairs(), 학습 시작 전 1회 계산
move = ±(Δt_a 또는 Δt_b)/2 × normal          # Δt = t_raw(현재) - t_raw0(Δ=0일 때)
```

주석(L1288-1289)이 명시하듯 목표는 "**초기 표면 간극 gap0을 학습 내내 보존**"하는 것이다 — 즉 gap을 줄이는 최적화 목표 자체가 존재하지 않는다. 두께가 변해도 표면 간극이 gap0으로 유지되도록 노드를 정확히 밀어낼 뿐이다.

**따라서 최종 결과물의 공차 크기는 결국 "입력 CSV(`initial_section_v4.csv`)가 floor0(또는 각 floor) 시점에 이미 갖고 있던 gap0 값"과 거의 1:1로 대응한다.** 이번 검토에서 v4.csv를 직접 재계산해 gap0이 289쌍 모두 정확히 0.0000mm임을 확인했으므로(§2-1의 WARNING 2 정정 참고), 이 메커니즘은 "이미 거의 완벽한 입력을 그대로 유지"하는 방향으로 작동하고 있다 — 결과가 양호한 것은 신경망이 간극을 학습으로 닫아서가 아니라, **애초에 입력이 닫혀 있었기 때문**이다.

**[WARNING] 5. 그럼에도 잔여 공차가 발생할 수 있는 두 개의 구체적 경로**

- **3-way 접합 노드에서의 오프셋 중첩**: `FLANGE_PAIR_SPEC` 주석(L732-733)이 명시하듯 Plate의 point_idx 22/23/24는 Plate-Inner 쌍(L724-725)과 Plate-Patch2 쌍(L729)에 **동시에** 속한다. 각 쌍의 오프셋(`move_a`/`move_b`)은 그 쌍 고유의 법선(`flange_normal`, 서로 다른 방향)을 따라 계산되고 `index_add`(L1320-1323)로 같은 노드에 **가산**된다. 각 쌍 단독으로는 목표 gap0을 정확히 복원하는 해석적 정답이지만, 두 법선이 평행하지 않은 한 한 노드에 두 개의 서로 다른 방향 보정이 겹쳐 더해지면 두 목표 gap을 "동시에" 정확히 만족시키는 것은 기하학적으로 보장되지 않는다 — 이는 학습 근사 오차가 아니라 **오프셋 공식 자체의 구조적 한계**다.
- **Export 시 run별 두께의 파트 단위 평균화**: `export_final_section_csv()`(L2261) `t_part = float(t_raw_cpu[pmask_all].mean())`는 파트 전체(모든 floor/run 포함)의 평균 `t_raw`를 그 파트의 대표 두께로 CSV 구분 행에 기록한다. 좌표(`new_coords`) 자체는 forward() 내부에서 **노드별 실제 t_raw**로 정확히 오프셋되므로 좌표는 내적으로 일관되지만, CSV에 최종 기록되는 "파트 두께" 값은 candidate 파트가 여러 run(pruning으로 분리된 구간)으로 나뉘어 run별 두께가 다를 경우 그 평균값이 될 뿐, 어느 run의 실제 두께와도 정확히 일치하지 않을 수 있다. 이번 v1_15_5 리포트(§4)에서는 Patch2의 두 run이 우연히 둘 다 2.300mm(T_MAX 포화)라 드러나지 않았지만, run별 두께가 갈리는 다른 조건에서는 "좌표는 맞는데 표기된 두께와 안 맞는" 형태의 명목상 공차가 발생할 수 있는 구조적 위험이다.
- **권장**: (a) 다중 참여 노드(여러 FLANGE_PAIR_SPEC 항목에 동시에 속하는 point_idx)를 사전에 식별해, 해당 노드에서는 여러 목표 gap의 가중 최소자승 등으로 오프셋을 한 번에 푸는 방식을 검토할 것. (b) `export_final_section_csv()`가 candidate 파트에 대해서는 run(segment) 단위로 별도 두께를 기록하도록 수정하거나, 최소한 run 간 두께 편차가 있을 때 경고를 출력할 것.

## 3. 결론

이번 v1.15.5의 표면적 결과가 양호한 것은 검증 코드가 잘 작동해서가 아니라, 입력 CSV(`initial_section_v4.csv`)가 이미 거의 완벽한 밀착 형상이고 학습 메커니즘이 "간극을 줄이는" 것이 아니라 "입력 간극을 그대로 보존"하도록 설계되어 그 좋은 초기 상태가 그대로 이어졌기 때문이다(§2-2, 확증). 파이프라인 구조상 이를 사후에 검증하는 코드는 침투(overlap) 검사 하나뿐이며 이는 간극(양의 gap) 문제를 원천적으로 포착하지 못한다(ERROR 1, 확증) — 즉 입력이 나빴다면 이번에도 걸러지지 않고 그대로 통과했을 것이다. `check_flange_gap()` 형태의 별도 하드 검증 추가, 3-way 노드 오프셋 중첩 처리, run별 두께 기록 정합화가 우선 권고 사항이다.

## 4. 신뢰도 및 숙의 과정

<details>
<summary>숙의 과정</summary>

- **Gemini (Architect, conf 95, can_exit=true)**: overlap/gap 비대칭성을 아키텍처 결함으로 규정, 최종 export 단계 결정론적 스냅핑(projection) 후처리 도입 제안.
- **OpenAI (Explorer, conf 87, can_exit=false — v4.csv 실제 내용을 직접 열람하지 못해 가정 포함이라고 스스로 명시)**: 동일 3개 결함을 edge-case 관점에서 확인, taper 경계 노드 한계를 추가로 세분화.
- **Claude (Validator, 1차)**: `check_part_overlap()`(L946-1013), `compute_flange_penetration_diag()`(L1047-1069) 및 그 호출부(L2842, L3139)를 직접 대조해 ERROR 1을 코드 근거로 확증. 두 모델 간 이견 없음(Critic/Defense 라운드 없이 조기 수렴 처리).
- **Claude (2차, 사용자 요청 — 근본 원인 추적)**: `identify_mating_flange_pairs()`의 gap0 계산 로직을 파이썬으로 재구현해 `initial_section_v2/v3/v4.csv`를 직접 파싱·재계산, v3.csv와 v4.csv가 gap0 기준으로 완전히 동일함을 수치로 확인(WARNING 2 → INFO 정정). `forward()`의 오프셋 공식(L1287-1329)을 대조해 "gap을 0으로 만드는" 것이 아니라 "입력 gap0을 보존"하는 설계임을 확증. `export_final_section_csv()`(L2186-2266)의 run-평균 두께 기록 방식을 확인해 §2-2 WARNING 5를 코드 근거로 도출.

**신뢰도**: Gemini 95 · OpenAI 87 · Claude 코드+데이터 검증(1차·2차) 일치.

**최종 신뢰도: 93%**

</details>
