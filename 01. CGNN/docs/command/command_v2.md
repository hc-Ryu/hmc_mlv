# command_v2.md — AI_design_v1_1.py: 동적 파트 분할(Dynamic Part Splitting) 구현 지시서

작성일: 2026-08-05 (/synod idea 세션 synod-20260805-234140-0a3ec3 합의안)
참조: `docs/idea/idea_v3.md`(설계), `docs/idea/idea_v2.md`(배경), `AI_design_v1.py`(기반 코드),
`uni-section/code/uni_section_v18.py`(불변 — import만)

> **앵커 규칙**: 본 문서의 모든 위치 지정은 **함수/변수 이름 기준**이다. 라인 번호는 참고용일 뿐
> 코드 변경으로 어긋날 수 있으므로, 구현 에이전트는 반드시 이름 매칭으로 위치를 찾을 것
> (/synod critic 라운드에서 Gemini·OpenAI 공통 지적).

---

## §0. 목표 및 산출물

`AI_design_v1.py`를 복사해 **`AI_design_v1_1.py`** 를 만들고, candidate 파트(part_id 3, 4)의
두께 pooling을 다음 3-Phase 스케줄로 교체한다. `uni_section_v18.py`와 `AI_design_v1.py` 원본은
수정하지 않는다.

| Phase | 조건 | candidate pooling 단위 |
|---|---|---|
| A/B (분할 전) | DELETED 확정이 하나도 없음 | **파트당 1 두께** (전 섹션 글로벌 — "하나의 패치 = 하나의 두께") |
| C (분할 후) | `pruning_state`에 DELETED 존재 | **연결 run당 1 두께** (DELETED 섹션을 절단점으로 1D CCL) |

즉, 현행 "candidate 섹션별 완전 독립 pooling"(`10_000 + section_id * max_parts + part_id`)은
**제거**된다 — 이것이 두께가 섹션마다 진동하는(제조 불가) 원인이었다.

산출물:
- `AI_design_v1_1.py`
- `tests/test_segment_ids.py` (CCL 단위 테스트, 학습 불필요)
- 기존과 동일 경로의 결과물 + `reports/figures/` 내 run별 두께 시각화

---

## §1. 신규 함수 `compute_segment_ids()` — 1D CCL

SECTION 2(pruning 유틸) 블록, `update_pruning_state_multi()` 아래에 추가:

```python
@torch.no_grad()
def compute_segment_ids(pruning_state):
    """candidate 파트별 1D Connected-Component Labeling.
    반환: seg_id (NUM_SECTIONS, len(CANDIDATE_PARTS)) long.
    생존(non-DELETED) 연속 run마다 0부터 시작하는 조밀한 run 번호, DELETED 칸은 -1."""
    deleted = (pruning_state['state'] == STATE_DELETED)            # (17, n_cand) bool, CPU
    alive = ~deleted
    # run 시작점: 살아있으면서 (첫 섹션이거나 바로 위 섹션이 DELETED)
    prev_deleted = torch.cat([torch.ones(1, deleted.shape[1], dtype=torch.bool), deleted[:-1]], dim=0)
    run_start = alive & prev_deleted
    run_id = torch.cumsum(run_start.long(), dim=0) - 1             # 살아있는 칸에서 0,1,2,...
    seg_id = torch.where(alive, run_id, torch.full_like(run_id, -1))
    return seg_id
```

주의사항:
- idea_v3 초안의 `cumsum(deleted)` 단순 버전이 아니라 **run_start 감지 버전**을 쓴다
  (Solver 라운드 Gemini 제안) — run 번호가 0부터 조밀하게 나와 키 stride 계산이 명확해진다.
- PENDING_DELETE는 **생존으로 취급**한다. DELETED(비가역 확정)만 절단점이다.
- `pruning_state['state']`는 CPU 텐서다(원본 `make_pruning_state_multi()` 참조). 반환값을
  forward에 넘기기 전 `.to(device)` 할 것.
- 전부 정수/불리언 연산 + `no_grad` — autograd 그래프에 절대 들어가지 않는다.

## §2. `CGDN17.forward()` 수정 — composite_key 교체

시그니처에 keyword 인자 추가: `segment_ids=None` (shape `(NUM_SECTIONS, len(CANDIDATE_PARTS))`
long 또는 None).

기존 composite_key 블록(§3.1 주석, `is_continuous` ~ `composite_key = torch.where(...)`)을
다음으로 교체:

```python
continuous_ids = torch.tensor(CONTINUOUS_PARTS, device=x.device)
is_continuous = torch.isin(part_ids_local, continuous_ids)

if segment_ids is None:
    # Phase A/B: candidate도 파트 단위 글로벌 pooling
    composite_key = part_ids_local
else:
    # Phase C: run 단위 pooling
    # part_id 3→열0, 4→열1 (CANDIDATE_PARTS=[3,4] 가정; 연속 파트 노드는 clamp로 안전 인덱싱 후 where로 무시됨)
    cand_col = (part_ids_local - CANDIDATE_PARTS[0]).clamp(min=0, max=len(CANDIDATE_PARTS) - 1)
    seg_of_node = segment_ids[section_ids_local, cand_col]        # (N,) — DELETED 칸 노드는 -1
    run_key = 10_000 + part_ids_local * (NUM_SECTIONS + 1) + seg_of_node
    # seg=-1 (DELETED 칸) 노드는 별도 dummy 그룹으로 격리 — 인접 키와의 충돌(-1 언더플로우) 방지
    run_key = torch.where(seg_of_node < 0, torch.full_like(run_key, 90_000 + part_ids_local), run_key)
    composite_key = torch.where(is_continuous, part_ids_local, run_key)
```

이하 `torch.unique(return_inverse)` → `scatter_add_` → `group_mean[inverse]` 로직은 **한 줄도
수정하지 않는다**. 필수 준수 사항:

- **키 충돌 금지 검증**: `seg_id ∈ [0, 8]` (17섹션 최대 run 수 = ⌈17/2⌉ = 9), stride
  `NUM_SECTIONS+1 = 18`이므로 파트 간 키 겹침 없음. **`seg_id = -1`을 clamp로 인접 run에 흡수하지
  말고 위처럼 dummy 키(90_000대)로 격리할 것** — critic 라운드에서 `-1` 언더플로우가 인접 파트
  키와 충돌할 수 있다는 지적이 CONFIRMED됨. dummy 그룹 노드는 z_gate≈0이라 t_final≈0이므로
  물리적으로 무해하다.
- **gradient 흐름**: composite_key/segment_ids는 정수 상수. gradient는 기존과 동일하게
  pooling된 `delta_t`를 통해서만 흐른다. `.detach()` 추가 금지(불필요), `requires_grad` 설정 금지.
- 파트 수·텐서 차원은 분할 후에도 **불변**이다. 바뀌는 것은 `torch.unique`가 만드는 그룹 개수뿐
  이므로 옵티마이저/파라미터 재구성은 필요 없다 (critic 라운드 Gemini의 "동적 차원 확장 대응"
  지적은 기각 — 이 아키텍처에는 해당 없음).

## §3. 학습 루프 배관 — 이벤트 구동 트리거

### 3.1 `train_step_multi()`
- `update_pruning_state_multi(...)`의 반환값 `newly_deleted`(현재 버려짐)를 지역 변수로 받아
  반환 dict에 `"newly_deleted": newly_deleted` 추가.
- 시그니처에 `segment_ids=None` 인자 추가, model 호출에 그대로 전달.

### 3.2 `run_training_multi()`
epoch 루프에 상태 변수 `seg_ids = None` 유지. 매 epoch, train_step 결과에서:

```python
if info['newly_deleted'].any():
    seg_ids = compute_segment_ids(pruning_state).to(device)
    n_runs = [int((seg_ids[:, c].max().item()) + 1) for c in range(len(CANDIDATE_PARTS))]
    split_epochs.append(epoch)
    print(f"[Dynamic Split] epoch {epoch}: DELETED 확정 "
          f"{info['newly_deleted'].nonzero().tolist()} → run 수 {n_runs} (pooling 재구성)")
```

- `seg_ids`는 `newly_deleted.any()`일 때만 재계산 (DELETED 비가역 → 라벨 단조 세분화,
  재병합 없음).
- 다음 epoch의 `train_step_multi()` 호출에 `segment_ids=seg_ids` 전달. (같은 epoch 내 재호출
  불필요 — 한 epoch 지연은 무해.)

### 3.3 Adaptive Epoch Extension
분할이 늦게 확정되면 Stage 3 독립 최적화 시간이 부족하다. epoch 루프를 `while` 또는 가변
상한으로 바꾸고:

```python
if split_epochs and epoch == max_epochs - 1:
    needed = split_epochs[-1] + 60
    if needed > max_epochs:
        max_epochs = needed
        print(f"[Adaptive Extension] 마지막 분할 epoch {split_epochs[-1]} → max_epochs={max_epochs}")
```

- 상한 안전장치: `max_epochs`는 절대 `초기값 + 120`을 넘지 않게 clamp (무한 연장 방지).
- history 배열은 append 방식이므로 연장에 자동 대응. `visualize_training_multi()`의
  stage 경계선은 상수 그대로 두어도 무방.

### 3.4 Warmup 램프 — **구현하지 않음**
idea_v3 §3.4의 "분할 직후 L2 warmup 패널티"는 **채택하지 않는다**. 분할 전 글로벌 pooling
덕분에 두 run은 동일 두께에서 출발하므로(불연속 0에서 시작) 패널티는 수학적으로 무의미하고,
오히려 필요한 분기를 지연시킨다 — critic 라운드에서 Gemini(conf 95)·OpenAI 양쪽 모두 동의한
결론. 스모크 테스트에서 loss spike가 실측되면 그때 재검토한다(§6 회귀 기준 참조).

### 3.5 Pre-split soft 편차 허용 — **구현하지 않음 (hard-global 채택)**
"글로벌 평균 + 섹션별 편차 패널티" soft 절충안(idea_v3 §5)은 기본 구현에서 제외한다.
물리적으로 한 파트인 동안 두께 불일치를 허용하는 것은 제조 일관성 원칙과 최소 침습 원칙에
어긋난다는 것이 critic 라운드 합의. 단, §6-3 회귀 테스트에서 초기 Mp 수렴이 v1 대비 유의미하게
악화되면(수렴 epoch 2배 이상 등) soft 방식을 후속 v1.2 과제로 승격한다.

## §4. Stage 0 dry-run 수정 — `run_static_dry_run()`

- 기존 "연속 파트만 두께 불변" 검사에 **candidate 파트 검사 추가**: `segment_ids=None`으로
  호출되는 dry-run에서는 candidate도 파트당 두께 spread < 1e-4여야 한다 (Phase A/B 글로벌
  pooling의 직접 검증). 기존 주석("candidate는 검사 대상 아님")은 v1.1에서 반대로 뒤집히므로
  주석도 갱신할 것.
- 추가 검사: 인위 pruning_state(예: 섹션 5~7 DELETED)로 `compute_segment_ids()` → forward에
  주입 → 같은 run 내 두께 spread < 1e-4, 서로 다른 run은 **독립 그룹**(unique 그룹 수 =
  연속 3 + 살아있는 run 수 + dummy 그룹 수) 확인.

## §5. 시각화/로깅 수정

| 대상 | 수정 |
|---|---|
| `history` | `'num_thickness_groups'` 추가(매 epoch, forward의 `num_groups` 또는 seg_ids로 계산) — 분할 시 계단식 증가가 성공의 1차 증거 |
| `plot_gate_heatmap()` | DELETED 확정 칸에 'X' 텍스트 오버레이(`pruning_state` 전달 필요), run 경계에 세로선 |
| `visualize_training_multi()` panel 8 | candidate 두께를 run별 라인으로 분리(`Patch1-run0` 등). 분할 전 구간은 단일 라인 → 분할 epoch에서 갈라지는 형태 |
| `plot_sections_3d_plotly_multi()` | candidate trace의 legendgroup을 `f'{PART_NAMES[pid]}-run{seg}'`로 분리, hover에 run 번호·두께 표기 |
| 콘솔 로그 | 분할 이벤트/에포크 연장 로그(§3.2, §3.3). 텐서를 포맷 문자열에 직접 넣지 말 것(v1 교훈) |

## §6. 검증 계획 (acceptance criteria)

1. **단위 테스트** `tests/test_segment_ids.py` (torch만 필요, 수 초):
   - state `[A,A,A,D,D,A,A,A,D,A,...]` → 손계산 seg_id 일치, DELETED 칸 -1.
   - 전부 ALIVE → 모든 칸 0 (run 1개). 전부 DELETED → 모든 칸 -1.
   - 경계: 섹션 0이 DELETED로 시작 / 섹션 16이 DELETED로 끝나는 경우.
   - composite_key 시뮬레이션: 파트 3 run과 파트 4 run 키가 절대 겹치지 않음을 전수 확인.
2. **스모크 테스트** (`--epochs 250`, sparse 가중치를 키워 삭제 유도):
   - `[Dynamic Split]` 로그 발생, `num_thickness_groups` 계단 증가.
   - 분할된 두 run의 최종 두께 차이 > 0.05mm.
   - 같은 run 내부 두께 spread < 1e-4 ("꿀렁꿀렁" 해소의 직접 증거).
   - loss/Mp 오차에 분할 epoch 전후 지속적 발산 없음(일시 spike ≤ 2 epoch 회복 허용).
3. **회귀 테스트** (sparse 가중치 축소, DELETED 미발생 시나리오):
   - candidate가 끝까지 파트당 1 두께 유지, seg_ids는 끝까지 None.
   - 최종 Mp 상대오차가 v1 결과와 동등 수준(≤ 2% feasibility 기준 동일 통과). 초기 수렴 속도가
     현저히 악화되면 §3.5 후속 과제 트리거.
4. `--dry-run-only` 통과 (§4의 신규 assert 포함).

## §7. 함정 경고 (구현 에이전트 필독)

- **키 언더플로우**: `seg_id=-1`을 산술 키에 그대로 넣으면 인접 파트 키와 충돌 — 반드시 dummy
  키로 격리 (§2, critic 라운드 CONFIRMED).
- **디바이스 불일치**: pruning_state는 CPU, forward는 CUDA일 수 있음 — seg_ids `.to(device)` 필수.
- **shape 함정**: `segment_ids` 인덱싱은 `[section_ids_local, cand_col]` fancy indexing —
  브로드캐스트로 (N,N)이 생기지 않는지 dry-run assert로 확인 (v1의 flatten/unflatten 교훈과 동일
  계열).
- **연속 파트 노드의 cand_col**: 연속 파트 노드도 cand_col 계산을 거치지만(clamp) 최종
  `torch.where(is_continuous, ...)`로 무시된다 — clamp를 빼먹으면 part_id 0~2에서 음수 인덱스
  런타임 에러.
- **GATE_ACTIVE_EPOCH 이전**: 게이트 학습 전에는 DELETED가 나올 수 없으므로 Phase A가 자동 보장
  — 별도 epoch 분기 코드를 만들지 말 것 (이벤트 구동이 유일한 전환 경로).
- 라인 번호로 위치 찾지 말 것 — 함수명/주석 텍스트 매칭 사용.

---

## §8. 숙의 요약 (Synod 세션 synod-20260805-234140-0a3ec3)

- 모드 idea | Gemini pro→**flash 폴백**(pro rate limit) thinking high | OpenAI gpt4o.
- Solver: Gemini(conf 100) — CCL run_start 벡터화·키 공식·에포크 연장·warmup 제안 / OpenAI(conf 85)
  — 문서 구조·키 충돌·soft pooling 경고 / Claude(Validator, conf 88) — 최소 침습 매핑 검증.
- Critic: Gemini(conf 95)·OpenAI(conf 75) 모두 can_exit=true로 수렴 → Defense 라운드 생략.
- 합의로 뒤집힌 항목: warmup L2 패널티 **제외**, soft 편차 허용 **제외**(hard-global 채택),
  seg_id=-1 **dummy 키 격리**(clamp 흡수안 폐기), 라인 번호 앵커 **금지**.
- 기각된 비평: "분할 시 텐서 차원 동적 확장 대응 필요"(Gemini critic) — unique 기반 pooling이라
  차원 불변, 해당 없음.
- 최종 신뢰도: **89%** (Trust 가중 평균; Gemini T≈1.6, OpenAI T≈1.1, Claude T≈1.4).
