# idea_v3 — 동적 파트 분할(Dynamic Part Splitting)을 AI_design_v1.py에 적용하기

작성일: 2026-08-05
참조: `docs/idea/idea_v2.md` (2-Stage 알고리즘 제안), `AI_design_v1.py`, `uni-section/code/uni_section_v18.py`

---

## 0. 문제 정의 (idea_v2 요약)

Candidate 파트(Patch 1 = part_id 3, Patch 2 = part_id 4)가 학습 중 특정 중간 섹션에서
z_gate=0으로 삭제되어 위/아래로 물리적으로 분리되면, 분리된 각 조각은 **서로 다른 파트**로
취급되어 **각각 독립적인 단일 두께**로 최적화되어야 한다 (예: 상단 조각 1.2t, 하단 조각 1.6t).

현재 코드가 이를 지원하지 못하는 이유:

- part_id는 정적 라벨(3, 4)이며, 끊어진 두 조각도 같은 part_id를 공유한다.
- 두께 pooling(`CGDN17.forward()`의 composite_key, AI_design_v1.py 약 line 172-189)은
  candidate 파트를 **섹션별 완전 독립**(`10_000 + section_id * max_parts + part_id`)으로
  묶는다. 즉 현재는 idea_v2가 말한 "로컬 풀링" 상태다:
  - 장점: 분리된 조각이 다른 두께를 가질 수는 있음.
  - 단점: **이어져 있는 구간조차 섹션마다 두께가 제각각** → 프레스 성형 관점에서 제조 불가.
- 반대로 candidate를 연속 파트처럼 전 섹션 글로벌 pooling으로 바꾸면, 분리된 두 조각이
  억지로 같은 두께를 강요받는다 (설계 자유도 손실).

핵심 난점(idea_v2 §2): 학습 중 z_gate가 0.4, 0.7 같은 연속값일 때 "끊어짐"을 미분 가능하게
판정할 수 없다. → **미분으로 풀지 말고, 이산 이벤트(DELETED 확정)를 트리거로 pooling 그룹을
재구성**하는 것이 정답이다. 다행히 현재 코드에 필요한 재료가 전부 있다:

| 필요 재료 | 현재 코드 위치 |
|---|---|
| 삭제 확정 이산 상태 | `pruning_state['state'] == STATE_DELETED`, (17, 2) 텐서 |
| 히스테리시스(깜빡임 방지) | `update_pruning_state_multi()` — EWMA + confirm_epochs=20 |
| 그룹별 pooling 메커니즘 | `composite_key` → `torch.unique(return_inverse)` → `scatter_add_` |
| 스테이지 경계 | `ASWC_STAGE1_END=50`, `ASWC_STAGE2_END=200`, `ASWC_STAGE3_END=250` |

---

## 1. 설계 원칙

1. **미분 가능성 보존**: 그룹 라벨(segment id) 계산은 전부 `torch.no_grad()` / 정수 연산으로
   수행하고, gradient는 기존과 동일하게 pooling된 delta_t를 통해서만 흐른다. composite_key는
   원래부터 비미분 정수 인덱스이므로 그래프를 건드리지 않는다.
2. **재넘버링 없는 라벨링**: part_id 자체(노드 feature 4열)는 절대 바꾸지 않는다. 새 파트 ID를
   노드에 써넣는 대신, **pooling 시점에 composite_key만 다르게 계산**한다. (idea_v2의
   "part_id=5, 6 동적 재할당"을 문자 그대로 구현하면 collision_spec, 시각화, PART_COLORS 등
   part_id에 의존하는 모든 코드가 깨진다 — key 레벨에서만 분할하는 것이 최소 침습.)
3. **DELETED만 절단점으로 인정**: z_gate가 낮아도 PENDING 상태면 아직 이어진 것으로 취급.
   DELETED는 히스테리시스+confirm_epochs를 통과한 비가역 확정이므로, 분할 이벤트도 비가역이
   되어 학습 안정성이 보장된다 (그룹이 매 epoch 진동하지 않음).

---

## 2. 제안 아키텍처: 3-Phase 두께 pooling 스케줄

idea_v2의 2-Stage 제안을 현재 ASWC 스테이지에 맞게 3단계로 구체화한다. 핵심은 candidate
파트의 pooling 단위가 학습 진행에 따라 **"파트 전체" → "연결 성분(run)"** 으로 세분화되는 것.

```
Phase A (epoch < GATE_ACTIVE_EPOCH):        candidate도 글로벌 pooling (파트당 1 두께)
Phase B (GATE_ACTIVE ~ 분할 이벤트 전):      동일 — "패치는 하나의 두께" 제약 유지 (idea_v2 Step 1)
Phase C (DELETED 발생 이후):                DELETED 섹션을 절단점으로 1D CCL → run별 독립 pooling
                                            (idea_v2 Step 2+3)
```

- 현재 코드의 "섹션별 완전 독립 pooling"은 **제거**한다. 이것이 꿀렁꿀렁 두께 문제의 원인이며,
  Phase C의 run별 pooling이 그 역할("재등장 시 새 두께")을 더 올바르게 대체한다: 재등장한
  조각은 DELETED 구간 건너편의 새로운 run이므로 자동으로 새 두께 그룹이 된다.
- Phase C 진입은 고정 epoch가 아니라 **이벤트 구동**: `update_pruning_state_multi()`가
  반환하는 `newly_deleted` 마스크가 True를 낸 순간부터 해당 파트의 pooling이 run 단위로
  세분화된다. (confirm_epochs=20 + GATE_ACTIVE_EPOCH 이후이므로 실질적으로 Stage 2 중후반
  ~ Stage 3에서 발생 — idea_v2의 "200 epoch 이후 Stage 3 진입 시점" 제안과 자연스럽게 일치.)

### 왜 "분할 후 재병합"은 없는가
DELETED는 현재 상태기계에서 비가역이므로(DELETED → ALIVE 전이 없음) run 라벨도 세분화만
된다. 단조 세분화(monotone refinement)라 두께 변수의 정체성이 학습 내내 안정적이다.

---

## 3. 구현 상세

### 3.1 segment_id 계산 — 1D CCL (신규 함수)

이미지 CCL의 1D 특수화는 cumsum 한 줄이면 된다. 섹션 축(0~16)을 따라 "생존(non-DELETED)
구간의 연속 run"에 고유 번호를 붙인다.

```python
@torch.no_grad()
def compute_segment_ids(pruning_state):
    """candidate 파트별 1D Connected Component Labeling.
    반환: seg_id (NUM_SECTIONS, len(CANDIDATE_PARTS)) long —
          같은 run에 속한 (section, part)는 같은 값, DELETED 칸은 -1.
    원리: deleted 마스크의 cumsum은 절단점을 지날 때마다 1씩 증가하므로
          생존 칸들의 run 번호가 된다."""
    deleted = (pruning_state['state'] == STATE_DELETED)          # (17, n_cand) bool
    run_id = torch.cumsum(deleted.long(), dim=0)                 # 절단점 누적 카운트
    seg_id = torch.where(deleted, torch.full_like(run_id, -1), run_id)
    return seg_id
```

예 (idea_v2의 예시 재현): Patch 1의 state가 `[A,A,A,D,D,A,A, ...]`이면
`seg_id = [0,0,0,-1,-1,2,2,...]` → 상단 run(0)과 하단 run(2)이 서로 다른 pooling 그룹.

### 3.2 CGDN17.forward()의 composite_key 교체

현재 (line 177-181):

```python
composite_key = torch.where(
    is_continuous,
    part_ids_local,                                          # 전 섹션 공통
    10_000 + section_ids_local * max_parts + part_ids_local, # candidate: 섹션별 독립  ← 제거
)
```

변경안 — forward()에 `segment_ids` 인자(shape (17, n_cand), 기본값 None)를 추가:

```python
if segment_ids is None:
    # Phase A/B: candidate도 파트 단위 글로벌 pooling ("패치는 하나의 두께")
    composite_key = part_ids_local
else:
    # Phase C: run 단위 pooling — key = 10_000 + part_id * MAX_RUNS + seg_id
    cand_col = candidate_col_of(part_ids_local)              # part_id 3→0, 4→1 매핑
    seg_of_node = segment_ids[section_ids_local, cand_col]   # 노드별 run 번호
    composite_key = torch.where(
        is_continuous,
        part_ids_local,
        10_000 + part_ids_local * (NUM_SECTIONS + 1) + seg_of_node.clamp(min=0),
    )
```

- `seg_of_node == -1`(DELETED 칸의 노드)은 어차피 `t_final = t_raw * z_gate`에서
  z_gate≈0이 곱해져 두께가 0이 되므로, clamp로 인접 run에 흡수돼도 물리적으로 무해하다.
  (엄밀함을 원하면 별도 dummy 그룹 key를 부여해도 되지만 필수 아님.)
- run 최대 개수는 `ceil(17/2)=9`이므로 `NUM_SECTIONS + 1` stride면 key 충돌 없음.
- `torch.unique(return_inverse)` → `scatter_add_` 이하 로직은 **한 줄도 안 바뀐다**.
  그룹 개수만 달라질 뿐 gradient 경로가 동일하므로 flatten/unflatten 같은 shape 함정도 없다.

### 3.3 train_step_multi / run_training_multi 연결

```python
# run_training_multi() 루프 내부, train_step 호출 전:
newly = ...  # train_step_multi 안에서 update_pruning_state_multi()가 반환 (현재는 버려짐)
if newly.any():
    split_active = True
    seg_ids = compute_segment_ids(pruning_state).to(device)
    print(f"[Dynamic Split] epoch {epoch}: (section,part) {newly.nonzero().tolist()} DELETED 확정 "
          f"→ run 단위 pooling 재구성 (그룹 수 변화 로그)")
model(... , segment_ids=seg_ids if split_active else None)
```

- 현재 `update_pruning_state_multi()`의 반환값 `newly_deleted`가 `train_step_multi()`에서
  버려지고 있다 (line 696) — 이것을 info dict로 올려보내 분할 트리거로 쓴다.
- `segment_ids`는 epoch마다 다시 계산할 필요 없이 `newly_deleted.any()`일 때만 갱신.

### 3.4 분할 직후 두께 워밍업 (선택이지만 권장)

분할 순간 pooling 그룹이 바뀌면 delta_t 평균이 점프해 loss spike가 날 수 있다. 두 가지 완화책:

1. **자연 완화(공짜)**: 분할 전 글로벌 pooling이므로 두 run의 초기 두께가 동일 → 분할 직후
   두께 불연속은 0에서 시작하고, 이후 gradient가 서서히 벌린다. 사실상 spike가 작다.
2. **명시적 완화**: 분할 이벤트 후 ~10 epoch 동안
   `delta_t_run = delta_t_global + λ(epoch) * (delta_t_run - delta_t_global)`,
   λ를 0→1 램프. (1로 충분하면 생략.)

### 3.5 Stage 3와의 상호작용

- 현재 Stage 2 종료 시(`epoch >= ASWC_STAGE2_END`) **연속 파트만** 두께 detach된다
  (line 655-658) — candidate run들의 두께는 Stage 3에서도 계속 학습 가능하므로,
  idea_v2 Step 3("분할된 파트 독립 최적화")이 별도 코드 없이 성립한다. ✔
- 단, Stage 3(250 epoch)까지 남은 구간이 50 epoch뿐이므로, 분할이 늦게(예: epoch 230)
  확정되면 독립 최적화 시간이 부족할 수 있다. → **분할 이벤트가 ASWC_STAGE2_END 이후에
  발생하면 max_epochs를 (마지막 분할 epoch + 60) 정도로 연장**하는 adaptive extension을
  권장한다. 또는 confirm_epochs를 앞당기도록 sparse loss 스케줄을 조정.

### 3.6 부수 수정 체크리스트

| 위치 | 수정 내용 |
|---|---|
| `run_static_dry_run()` | segment_ids=None 경로 검사에 "candidate 파트도 (분할 전) 전 섹션 동일 두께" 검사 **추가** (현 정정 주석과 반대가 됨에 유의 — 분할 전 글로벌 pooling으로 바뀌므로) |
| `train_step_multi()` line 696 | `newly_deleted` 반환값을 info로 전달 |
| `visualize_training_multi()` panel 8 | candidate 파트 두께를 run별 라인으로 분리 표시 (예: `Patch1-run0`, `Patch1-run2`) — 분할 성공의 직접 증거 |
| `plot_gate_heatmap()` | DELETED 확정 칸에 'X' 마커 오버레이 + run 경계선 표시 |
| history | `'num_thickness_groups'` 추가 — 그룹 수가 계단식으로 증가하는지 추적 |
| 3D plotly | run별로 legendgroup 분리하면 조각별 두께를 hover에 표기 가능 |

---

## 4. 검증 계획

1. **단위 테스트 (학습 없음)**: `pruning_state['state']`를 인위적으로
   `[A,A,A,D,D,A,A,A,D,A,...]`로 세팅 → `compute_segment_ids()` 출력이 손 계산과 일치,
   composite_key 그룹 수 = 연속파트 3 + run 수 인지 assert.
2. **스모크 테스트**: `--epochs 250`으로 실행, 특정 (section, part)가 DELETED 되도록
   sparse 가중치를 키운 시나리오에서:
   - 분할 로그 발생 확인, `num_thickness_groups` 계단 증가 확인.
   - 분할된 두 run의 최종 두께가 유의미하게 다른지(> 0.05mm) 확인.
   - 이어진 run 내부에서는 두께 spread < 1e-4 (Stage 0 검사와 동일 기준) 확인 —
     "꿀렁꿀렁 두께" 문제가 해소되었다는 직접 증거.
3. **회귀 테스트**: DELETED가 전혀 발생하지 않는 시나리오(sparse 가중치 축소)에서
   candidate 파트가 끝까지 파트당 1 두께를 유지하고 Mp 수렴이 기존과 동등한지 확인.

---

## 5. 리스크 및 열린 질문

- **글로벌 pooling 전환의 부작용**: 현재 코드는 섹션별 독립 pooling 덕분에 Mp를 섹션 단위로
  맞추기 쉬웠다. 분할 전 글로벌 pooling으로 바꾸면 candidate 두께의 자유도가 줄어
  초반 Mp 오차가 커질 수 있다 → 좌표(delta_coords)와 연속 파트 두께가 흡수해야 하며,
  이것이 안 되면 절충안으로 "Phase B에서 글로벌 평균 + 소폭 섹션별 편차 허용
  (편차에 L2 패널티)" 같은 soft 제약을 검토.
- **분할 후 Mp 재분배**: 상단 run이 얇아지고 하단 run이 두꺼워지는 idea_v2의 기대 거동은
  `l_phys`가 섹션별로 걸려 있으므로 자동으로 유도된다. 별도 loss 불필요할 것으로 예상되나
  스모크 테스트에서 확인 필요.
- **Patch1/Patch2 동시 분할**: seg_id가 파트 열별로 독립 계산되므로 상호 간섭 없음.
- **재등장(re-emergence) 시나리오**: 현 상태기계에서 DELETED는 비가역이므로 "꺼졌다가 다시
  켜지는" 재등장은 발생하지 않는다. 만약 향후 재등장을 허용하려면 seg_id 안정성(라벨이
  뒤바뀌는 문제)을 다시 설계해야 함 — 본 문서 범위 밖.

---

## 6. 요약

| idea_v2 제안 | idea_v3 구현 매핑 |
|---|---|
| Step 1: 글로벌 풀링으로 위상 탐색 | Phase A/B — composite_key를 part_id 단위로 (현행 섹션별 독립 pooling 제거) |
| Step 2: CCL + part_id 동적 재할당 | `compute_segment_ids()` (cumsum 1D CCL), part_id 재할당 대신 composite_key 레벨 분할 |
| Step 3: 분할 파트 독립 최적화 | run별 pooling 그룹이 자동으로 독립 변수화, Stage 3 detach 정책 그대로 활용 |
| 트리거 시점 | 고정 epoch 대신 `newly_deleted` 이벤트 구동 (히스테리시스 확정과 동기화) |

최소 침습 원칙: 노드 feature(8열), z_gate 게이팅, pruning 상태기계, loss 구성은 전부 그대로.
바뀌는 것은 **composite_key 계산 한 곳 + segment_ids 배관 + 시각화**뿐이다.
