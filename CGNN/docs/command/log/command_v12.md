# command_v12.md — uni_section_v11 → v12 개선 지시서 (Cascade Stop-Snap-Resume)

> 출처: /synod idea 세션 (Claude Validator + Gemini flash/high + OpenAI gpt4o, Solver 라운드 실 병렬 교차검증)
> 기반: `docs/idea/idea_v12.md`(Cascade 아이디어) + `docs/idea/idea_v1.md`(v11 실패 진단) + `results_v11` 실측
> 목표: v11의 **소프트 게이트 프루닝**을 계승하되, 도태된 부재를 **실제로 삭제(Hard Pruning)**하고
>       인접 부재를 밀착(Snap)시킨 뒤 **완전 재시작(Hard Restart)**하는 연쇄 최적화 파이프라인 구축.

## 0. 판정 요약표

| idea_v12 제안 | 판정 | 비고 |
|---------------|------|------|
| Cascade Stop-Snap-Resume (하드 프루닝+수술+재시작) | **채택** | 소프트 게이트의 잔존 질량·수치 특이성 제거 효과 확실 |
| Cycle 1 트리거 `z_i<0.1, 20 epoch 지속` | **조건부 채택** | ⚠️ **선결 과제 필수**: v11 게이트가 고착되어 z가 안 떨어짐 → §1 없이는 영원히 미발동 |
| v11 손실(one-sided mass + λ0·mean(z)) 계승 | **수정 채택** | v11에서 실패한 구조 그대로면 트리거 불가 → §1로 교체 |
| Graph Surgery + Y-Snap + 차원 축소 | **채택** | 5-part 토폴로지 전용 규칙 명세 필요(§2) |
| CGDN 재인스턴스화 + optimizer 리셋 | **수정 채택** | 생존 파트 파라미터·좌표는 **warm-start 이식**(형상 이력 보존) |

> **핵심 결론(양 모델 만장일치):** Cascade는 타당하나, **v11 프루닝 학습 버그(§1)를 먼저 고치지 않으면
> Cycle 1 트리거가 절대 발동하지 않는다.** `uni_section_v12.py`가 이미 §1 수정을 구현했으므로,
> 본 지시서는 **v12.py를 프루닝-정상화 베이스라인으로 삼고 그 위에 Cascade(§2~§4)를 얹는** 것을 권장한다.

---

## 1. [선결 과제 · 최우선] v11 프루닝 학습 정상화

results_v11 실측: **E[z]가 0.990→0.991로 전 구간 고착**(게이트 미학습), **면적 +36% 증가**(경량화 실패).
아래를 적용하지 않으면 `z_i<0.1`이 발생하지 않아 Cascade 전체가 죽는다. (근거: idea_v1.md §5, v11 로그)

### 1.1 물리엔 결정론적 게이트 (variance trap 제거)
학습 중 물리 솔버(PNA Mp/면적/충돌)에 **stochastic 샘플이 아닌 결정론적 게이트**를 전달한다.
```python
z_gate = clamp(stretch(sigmoid(log_alpha)), 0, 1)      # 물리용(결정론) → t_final = z_gate · t_raw
z_open = sigmoid(log_alpha - beta*log(-gamma/zeta))    # 희소화(L0 활성확률)용
```
- stochastic 샘플을 매 스텝 `t_final`에 곱하면 Mp/충돌이 요동쳐 학습이 발산한다(v11 loss 1→5).

### 1.2 feas_gate 데드락 제거 + 질량 하향 압력
v11의 `contrib_sparsity = λ0·feas_gate·mean(z)`에서 **feas_gate 곱을 삭제**한다.
Mp 불만족 시 희소화가 0이 되어 게이트가 학습되지 않는 순환 데드락을 끊는다. 질량은 제약식으로:
```
L = w_phys·asymHuber(Mp)                       # 강건 앵커, w_phys=30 (v11 10)
  + μ_m·relu(area/target − 1)                  # Augmented-Lagrangian 상한(10 epoch마다 feasible이면 μ_m+=0.5)
  + w_area·(area/area_init)                     # ★ 하향 압력(Stage3만) — one-sided cap이 없던 v11의 근본 결함
  + λ_s·mean(z_open[{3,4}])                     # 프루닝 후보만 희소화
  + w_coll·Σ z_i z_j·violation²                 # v11 유지(사라지는 부재 충돌 무력화)
```
- **하향 항 `w_area·area/area_init`이 없으면 Mp를 면적 키워 맞추므로 경량화가 불가능**(v11 +36%의 직접 원인).

### 1.3 초기화·스케줄
- `init_log_alpha`: 2.5(z≈1 고착) → **2.0**(z≈0.96, 유연) 또는 idea_v12의 0.5 계열. 단 Stage1·2엔 게이트 **동결(z=1)**.
- `part_gate.log_alpha` **전용 optimizer 그룹 lr=1e-2, wd=0**; 매 스텝 `log_alpha.clamp_(-8,8)`.
- **β 어닐링** 0.7→0.1(Stage3): 게이트 이진화 경화.
- **더블 쇼크 제거**: 두께 언락(epoch≈90)과 프루닝 시작(epoch≈150)을 **다른 시점**에.

> 이 §1은 `uni_section_v12.py`에 이미 구현됨. Cascade 착수 전 **먼저 v12.py로 1회 학습**하여
> "특정 후보 파트의 z가 실제로 0.1 아래로 떨어지는지" 확인(검증 Gate 1). 안 떨어지면 λ_s/μ_m/w_area 튜닝.

---

## 2. [Graph Surgery] B-Pillar 5-Part 토폴로지 전용 수술·밀착 규칙

### 2.1 토폴로지 스택 및 삭제 케이스
```
Y≈60  [0] Outer Hat   (fy1470)  ─ PROTECTED (z=1)
Y≈45  [2] Inner Hat   (fy1470)  ─ Active  ⚠ 구조재(고위험)
Y≈15~28 [1] Inner Plate(fy980)  ─ PROTECTED (z=1)
Y≈16  [3] Patch1      (fy980)   ─ Active  ← 주 삭제 후보
Y≈7   [4] Patch2      (fy440)   ─ Active  ← 주 삭제 후보
```
- **보호 파트({0,1})는 삭제 대상 아님** → 순수 외곽 삭제(Case A)는 **발생하지 않음**.
- 모든 삭제는 **Case B(샌드위치/보강재 삭제 + 밀착)**로 처리한다.
- ⚠️ **Validator 결정 포인트:** idea_v12는 보호를 {0,1}로만 두어 Inner Hat(2, 구조재)이 삭제 후보에 남는다.
  Inner Hat 삭제는 단면 강성 급락 위험이 크다. **1차 실행은 protect={0,1,2}로 후보를 패치 {3,4}로 한정**하고,
  Inner Hat 프루닝은 안정화 후 별도 검토를 권장(idea_v1.md와 정합). (idea_v12를 엄격히 따르려면 {0,1}이되,
  Part2 삭제 트리거 임계를 더 보수적으로.)

### 2.2 Y-Snap 규칙 (밀착)
- **[3] Patch1 삭제**: 하부 [4] Patch2의 y를 [1] Inner Plate 하면 플랜지 레벨로 끌어올려 밀착. (Patch2 부재 시 스냅 불필요)
- **[4] Patch2 삭제**: 데이터에서 제거만. 추가 스냅 불필요.
- **[2] Inner Hat 삭제**(허용 시): [2] 플랜지에 붙어 있던 노드를 [1] Inner Plate 상면 y로 스냅. [0] Outer Hat은 보호·고정이므로 이동하지 않음.

### 2.3 `perform_graph_surgery(data, dead_part_id)` 유틸리티 (신설)
1. `dead_part_id`에 속한 노드 ID 집합 검색.
2. §2.2 규칙에 따라 생존 인접 파트 노드의 y좌표 스냅(shift) 적용.
3. `data.x`에서 해당 노드 제거, `data.edge_index`에서 참조 엣지 필터링 후 **0..N_new-1 재색인**.
4. 새 `Data` 객체 반환(차원 축소). `node_registry`도 재구성.

---

## 3. [Cascade Loop] Stop-Snap-Resume 제어 (`run_cascade_training` 신설)

기존 `run_training`을 감싸는 **아우터 루프**를 신설한다.

### 3.1 Cycle 1 — 위상 탐색 + 트리거
- §1 정상화된 게이트로 노드+두께+z 동시 최적화(λ_s 어닐링).
- **트리거**: 후보 파트 i의 **eval 게이트 z_i(결정론) < τ=0.1**이 **연속 20 epoch** 유지 → 삭제 확정, `BREAK`.
- **연쇄 붕괴(avalanche) 방어**: 한 이벤트에서 **가장 z 낮은 단 1개 파트만** 삭제. 동시 다수 트리거여도 1개씩.

### 3.2 Event — 수술
- 삭제 직전 상태 `checkpoint_pre_surgery_part_{id}.pt` 저장.
- `perform_graph_surgery(data, dead_id)` 호출 → 밀착·차원축소된 새 data.
- `build_collision_spec(new_data)` **재호출**: 삭제 파트 관련 앵커 쌍 제거, 생존 파트 clearance 재산정
  (9-앵커 스킴이 고정 부재 수 전제이므로 **재빌드가 필수** — 하드 삭제가 앵커를 깨는 문제의 해법).

### 3.3 Cycle 2 — Hard Restart + Warm-start
- 수술된 좌표를 **새 base_coords**로 설정.
- `CGDN(n_parts=new_count)` **신규 인스턴스화**, optimizer/epoch/curriculum/β/μ_m **전부 리셋**.
- **Warm-start(형상 이력 보존)**: 생존 파트의 좌표·`t_raw`·(가능하면 파트 임베딩)를 새 모델에 **이식**.
  구조 및 optimizer 모멘텀만 끊고, 학습된 기하는 살린다(순수 리셋 시 수렴 지연).
- 새 환경에서 타겟 Mp로 재학습(초기 커리큘럼 웜업 자연 적용).
- 종료 조건: 더 이상 트리거 없이 Mp feasible 수렴, 또는 최대 Cycle 수(예: 3) 도달.

### 3.4 비미분 이벤트 처리
- 수술은 이산 이벤트 → autograd 체인 단절. **반드시 새 optimizer/scheduler 생성**, 이전 Adam m,v 폐기.
- 각 Cycle의 best(eval 게이트 기준 Mp feasible & 최소 면적) 체크포인트를 개별 저장.

---

## 4. 구현 순서 및 검증 게이트

1. **§1 프루닝 정상화 먼저** (또는 `uni_section_v12.py` 채택) → 단독 1회 학습.
   - **Gate 1**: Cycle 1 없이도 후보 파트 z가 학습 중 0.5→0.1로 하강하는가? (실패 시 §1.2 데드락/하향항 재점검)
   - **Gate 2**: 면적이 초기 대비 **감소**하는가? (증가하면 w_area↑ 또는 μ_m 스케줄 조정)
2. **§2 perform_graph_surgery 단위 테스트**: 임의 dead_part_id로 노드/엣지 수 정확 감소, 재색인 무결성,
   y-snap 좌표 정확성 확인(차원 불일치 에러 0).
3. **§3 run_cascade_training 통합**: 트리거→수술→재시작이 에러 없이 1 Cycle 완주하는가?
   - **Gate 3**: Cycle 2 재학습이 epoch 0부터 정상 기동, `build_collision_spec` 재빌드 후 충돌 항 정상.
4. **다중 Cycle**: 2개 이상 파트가 순차 삭제되어도 안정 수렴하는가? (avalanche 쿨다운 확인)
5. 각 단계 후 기존 `verify_thickness_gradient` 통과 확인.

---

## 5. 리스크 및 완화 (숙의 도출)

| 리스크 | 완화 |
|--------|------|
| **트리거 미발동**(v11 게이트 고착) | §1 선결(결정론 게이트·feas_gate 제거·하향항·lr 1e-2) — **최우선** |
| **연쇄 붕괴**(1개 삭제→나머지 z 급락) | 이벤트당 1개만 삭제 + 수술 후 N_cool≈30 epoch 트리거 비활성 |
| **형상 이력 상실**(순수 리셋) | 생존 파트 좌표·t_raw warm-start 이식(구조·모멘텀만 리셋) |
| **Inner Hat(2) 오삭제**(구조 붕괴) | 1차 protect={0,1,2}로 후보를 패치 {3,4}로 제한 |
| **차원 불일치·재색인 버그** | perform_graph_surgery 단위 테스트(Gate 2) 필수 |
| **하드 삭제가 9-앵커 깨뜨림** | 수술 후 build_collision_spec **재호출**로 앵커 재구성(§3.2) |

---

<details>
<summary>숙의 과정 (Synod idea)</summary>

**모델 기여**
- **Gemini (Architect, conf 95):** Cascade 장점(수치 안정·모멘텀 리셋·실질 경량화) 정리, B-pillar 스택 기반 Y-snap 규칙 정의, warm-start 이식·avalanche 쿨다운·단일 삭제 안전장치, PyG 재색인/재인스턴스화 절차, 검증 게이트.
- **OpenAI (Explorer, conf 75, can_exit=false):** "선결 과제 없이는 목표 달성 불가" 강조(신뢰도 낮춰 미해결 신호), 비미분 이벤트 체크포인트 전략, 다중 삭제 관리 필요성.
- **Claude (Validator):** ① Cascade 트리거가 v11의 **실패한 프루닝**에 의존 → §1을 최우선 선결로 승격. ② gpt4o의 "Outer/Inner Hat = Case A" 오분류 교정(보호 파트는 삭제 안 됨 → 전부 Case B). ③ Inner Hat(2) 구조재 오삭제 위험 → protect={0,1,2} 1차 권장. ④ `uni_section_v12.py`가 이미 §1을 구현 → Cascade의 프루닝-정상화 베이스라인으로 지정.

**해결된 쟁점**
1. Cascade 발동 조건 → v11 게이트 버그(feas_gate/variance/init) 선결이 전제.
2. 하드 삭제의 9-앵커 파괴 → 수술 후 build_collision_spec 재빌드로 해결.
3. 리셋 vs 이력 보존 → optimizer/구조만 리셋, 생존 파트 형상 warm-start.
4. 삭제 후보 범위 → 1차는 패치 {3,4}(Inner Hat 보호), 안정화 후 확대 검토.

**신뢰도**
- Gemini 95 / OpenAI 75(미해결 신호) → 핵심(선결 과제) 완전 일치. **종합 ≈ 86%.**
- 남은 불확실성: Y-snap 좌표 규칙의 물리적 타당성은 실제 수술 후 Mp 회복으로 검증 필요; Cascade 다중 Cycle 안정성은 실험 확인 대상.
- 주의: Gemini pro는 rate limit 회피 위해 flash 사용.
</details>
