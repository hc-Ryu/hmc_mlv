# command_v1.md — 17-Section B-Pillar 구조 최적화 구현 작업지시서

기반: `docs/idea/idea_v0.md`(요구사항 원문), `docs/idea/idea_v1.md`(사용자 주석 반영, ASWC 원안).

**[2026-07-28 /synod debug 세션 — §1~§4 전면 재작성]** 이전 버전(§1~§4)은 이름은 "Autoregressive
Sliding-Window Curriculum"이었지만 실제 지시 내용이 17섹션을 한 그래프로 합쳐 매 epoch 전체를 한 번에
학습하는 **joint(동시) 방식**이었다. 이는 idea_v0.md 요구사항 1·2("17th section부터 시작해 위→아래로
순차적으로 생성")를 위반하며, idea_v1.md §1이 원래 제안했던 "섹션 k 학습 시 [k,k+1](나중에 k+2로 확장)
윈도우만 그래디언트를 흘리고 이미 확정된 위쪽 섹션은 detach"라는 진짜 순차 설계와도 달랐다. 이번
정정에서 §1~§4를 idea_v1.md 원안대로, 사용자가 명시한 **윈도우 [k, k+1, k+2]** 기준으로 다시 쓴다.
(이 정정 이전에 joint 방식으로 구현됐던 `AI_design_v1.py`는 별도 세션에서 재작성 필요 — 이 문서 §7 참고.)

**사용자가 확정한 제약 (반드시 준수):**
1. `AI_design_v0.py`는 참고하지 말 것 — part 구성이 다르다. **`initial_section.md`만 단일 기준**으로 따른다.
2. 기본 경로는 **17섹션 전체를 바로 구현하는 실전 코드**다. 축소(2~3섹션) 검증은 강제 선행 단계가
   아니라, §6에 정의된 조건에서만 예외적으로 실행한다.
3. Part 삭제/재등장은 **사람이 미리 짜는 JSON 스케줄이 아니다** — "학습 중 목표 Mp를 맞추다 보니
   특정 섹션에서 그 파트가 자연히 켜지거나 꺼지는 것"이어야 한다(§3.3).
4. **[신규]** 섹션 생성은 **17th(최상단/원본)부터 시작해 1st(최하단)까지 순차적으로** 진행해야 하며,
   임의 시점에 활성 그래디언트를 갖는 섹션은 **최대 3개(윈도우 [k, k+1, k+2])**뿐이어야 한다. 이미
   확정된(윈도우를 벗어난) 섹션은 `.detach()`되어 continuity loss의 상수 타깃으로만 남는다.

### 인덱싱 규약 (반드시 확인)

idea_v0.md는 **1-indexed, k=17이 최상단/원본, k=1이 최하단**으로 서술한다. 그런데 코드
(`AI_design_v1.py`의 `scale_factor()`)는 0-indexed `section_id`를 쓰며, 실제 스케일 공식을 보면
`section_id=16 → s=1.0(원본, 최상단)`, `section_id=0 → s=1.6(최하단)`이다. **이전 세션에서 코드
주석에 "section_id 0=최상단"이라고 잘못 적어놓은 문서화 버그가 있었다** — 실제로는 반대다
(`section_id=16`이 최상단). 이 문서와 앞으로의 구현은 **idea_v0.md의 1-indexed k(17=최상단..1=최하단)를
1차 기준으로 쓰고**, 코드에서 0-indexed `section_id`가 필요할 때는 `section_id = k - 1`로 명시적으로
변환한다. 새 구현에서는 `section_id` 변수명 대신 `k`를 직접 순회 인덱스로 쓸 것을 권장(혼동 방지).

---

## 1. 목표/범위

`uni-section/code/uni_section_v18.py`의 단일 섹션 구조(`build_bpillar_section()`, `CGDN`,
`run_training()` 등)를 **그대로 재사용**해, 17개 섹션을 **k=17→1 순서로 하나씩** 생성한다. 매 순간
활성 그래디언트를 갖는 섹션은 최대 3개(슬라이딩 윈도우)뿐이며, 이미 지나간(확정된) 섹션은 다시
학습되지 않는다. 핵심 요구사항:
(a) outer/inner part 두께는 17섹션 전체에서 절대 불변(§3.2),
(b) candidate part(patch) 존재/삭제는 섹션마다 독립적으로 학습된 게이트가 결정(§3.3),
(c) 인접 섹션 간 형상 연속성을 과도하지 않은 loss로 유지(§4),
(d) 목표 Mp를 공간 상관 있게 17섹션에 분배(§5, 변경 없음).

새 파일 `uni_section_v18_sequential.py`에 구현한다(`uni_section_v18.py` 원본은 수정하지 않고 import).

---

## 2. 실행 아키텍처: Sliding-Window Sequential Generation

### 2.1 바깥 루프 — 섹션을 하나씩 확정

```python
TOP_K, BOT_K = 17, 1     # idea_v0.md 1-indexed 표기 그대로 사용
WINDOW = 3               # [k, k+1, k+2]

sections = {k: None for k in range(TOP_K, BOT_K - 1, -1)}   # SectionState 저장소
frozen_coords = {}        # k -> detached 최종 좌표 (continuity loss 타깃)
frozen_thickness_continuous = None   # k=17 완료 후 채워짐, 이후 전 섹션 공통 상수

for k in range(TOP_K, BOT_K - 1, -1):        # 17 -> 1
    # 1) 이번 스텝에 살아있어야 하는 윈도우: {k, k+1, k+2} (범위를 벗어나면 제외)
    active_ks = [kk for kk in (k, k+1, k+2) if BOT_K <= kk <= TOP_K]

    # 2) 아직 생성되지 않은 섹션(k)만 새로 SectionState로 초기화
    if sections[k] is None:
        sections[k] = build_section_state(k, frozen_thickness_continuous)

    # 3) 윈도우 안(k, k+1)에 대해서만 순전파/역전파 반복 (§2.2)
    train_window(k, active_ks, sections, frozen_coords, frozen_thickness_continuous)

    # 4) 섹션 k 확정: detach, optimizer에서 제거, continuity 타깃으로 캐시
    finalize_section(k, sections, frozen_coords)

    # 5) k=17(최초 섹션) 완료 직후: 연속 파트 두께를 영구 상수로 고정 (§3.2)
    if k == TOP_K:
        frozen_thickness_continuous = sections[k].get_continuous_thickness().detach().clone()
```

- `k+1`, `k+2`는 **이미 확정된(위쪽) 섹션**이다 — 이번에 새로 학습하는 건 `k` 하나뿐이고, `k+1`/`k+2`는
  `frozen_coords`에서 읽어온 detach된 상수로만 continuity loss에 참여한다(양방향 역전파 없음, idea_v1.md
  §1의 "위쪽은 detach, 아래쪽만 학습 대상"과 동일).
- 맨 처음(`k=17`)엔 `k+1`, `k+2`가 존재하지 않으므로 continuity loss가 없다(자유롭게 원본 형상에서
  출발). `k=16`엔 `k+1=17`만 존재. `k<=15`부터 `k+1`, `k+2` 둘 다 존재.
- 맨 끝(`k=1`, `k=2`)엔 윈도우가 자연히 줄어든다(`active_ks`가 2개 또는 1개) — 별도 처리 불필요, 범위
  체크만으로 충분.

### 2.2 윈도우 내부 학습 — `uni_section_v18.run_training()` 재사용

`train_window(k, ...)`는 원본 `run_training()`/`train_step()`의 물리 loss 계산(`calculate_mpl`,
`compute_mass_loss`, `compute_collision_loss_v5` 등)을 **그대로 재사용**하되, 다음만 다르다:
- 최적화 대상 파라미터는 **섹션 k의 좌표/두께/게이트뿐**(k+1, k+2는 파라미터가 아니라 상수 텐서).
- loss에 continuity 항(§4)을 추가: `k`의 형상을 `frozen_coords[k+1]`(스케일 보정 후)에 가깝게.
- epoch 수는 섹션당 하이퍼파라미터(`N_EPOCH_PER_SECTION`, 예: 150~300 — §6에서 실측 후 조정)로,
  원본 `STAGE2_EPOCH=10`/`GATE_ACTIVE_EPOCH=11` 같은 epoch 상수는 **섹션 내부 상대 epoch 기준으로
  그대로 재사용**한다(섹션마다 자체 optimizer/스케줄을 새로 만들기 때문).

---

## 3. CGDN/게이팅: 섹션별 독립 파라미터

### 3.1 원본 그대로 재사용 가능한 부분

`build_bpillar_section()`, `CGDN`(forward 시그니처 동일), `compute_gates()`, `S_PROTECT=[0,1,2]`,
`CANDIDATE_PARTS=[3,4]`는 **원본 그대로** 섹션 하나짜리 인스턴스에 쓴다 — 이전 joint 방식에서 도입했던
`(17, len(CANDIDATE_PARTS))` 확장 `log_alpha`, `compute_gates_multi()` wrapper는 **더 이상 필요 없다**
(/synod design 세션 Gemini conf 95, OpenAI conf 71 공통 확인). 각 섹션이 독립적으로 학습되므로 원본의
1D `log_alpha`(shape `(5,)`) 하나를 섹션마다 새로 인스턴스화하면 그만이다.

### 3.2 연속 파트(outer/inner) 두께 고정 — k=17에서 1회만 학습, 이후 상수

```python
# k == TOP_K(17) 학습 종료 직후
frozen_thickness_continuous = {
    part_id: t_final[part_ids == part_id].mean().detach()
    for part_id in usv18.S_PROTECT   # [0, 1, 2]
}
```
- `k=16..1`의 `SectionState`를 만들 때, 연속 파트(0,1,2)의 노드 feature `t_val`(컬럼 6)에
  `frozen_thickness_continuous`를 그대로 대입하고, `CGDN.thickness_decoder`가 이 파트들에 기여하는
  텐서 부분을 **매 forward마다 detach**해서 그래디언트가 전혀 흐르지 않게 한다(원본 forward 로직은
  그대로 두고, 반환된 `t_final`에서 연속 파트 인덱스만 detach 후 override).
- 이렇게 하면 요구사항 4("절대 두께가 바뀌지 않음")가 **pooling 트릭이 아니라 단순 상수 대입**으로
  가장 직접적으로 보장된다.

### 3.3 Candidate 파트 게이팅 — 섹션마다 완전 독립, 사전 스케줄 없음

각 `SectionState`는 자신만의 `log_alpha`(shape `(5,)`, `S_PROTECT` 인덱스는 항상 1.0 고정)를 갖고,
원본 `make_pruning_state()`/`update_pruning_state()`를 **그대로** 섹션 단위로 돌린다. 섹션 k의 학습이
끝나면 이 게이트도 함께 확정(detach)되고, 다음 섹션(k-1)은 **완전히 새로운 `log_alpha`로 시작**한다 —
즉 k=16에서 patch가 꺼졌더라도 k=15는 그 사실을 전혀 모른 채 자기 목표 Mp만 보고 patch 존재 여부를
새로 결정한다. 이것이 idea_v0.md 요구사항 5("mp를 만족시키기 위해서" 섹션마다 자연 발생)를 **가장
단순하게** 만족시키는 방법이며, 재등장 시 "새 두께"도 별도 재넘버링 없이 자동 성립한다(각 섹션이 이미
독립 파라미터이므로).

**윈도우 안(k, k+1, k+2)에서 게이트 온도/스케줄 주의**: `k+1`, `k+2`가 확정되어 detach될 때, 그
시점의 HardConcrete temperature도 함께 고정해야 한다(계속 anneal되는 채로 두면 train/frozen 상태의
sparsity가 미묘하게 달라짐 — /synod debug 세션 Explorer 지적). `finalize_section()`에서
`sections[k].gate_temperature`를 그 시점 값으로 얼려서 `frozen_coords`와 함께 저장한다.

---

## 4. 인접 섹션 연속성 loss (윈도우 기반)

```python
def continuity_loss(coords_k, frozen_coords_k_plus_1, part_ids, scale_k, scale_k_plus_1, threshold=2.0):
    """섹션 k(학습 중)의 좌표를, 이미 확정된 k+1(위쪽, detach)의 좌표에 스케일 정규화 후 hinge loss로
    가깝게 유도. threshold(mm) 이내는 무패널티(요구사항 6 '과도한 제약 금지')."""
    ck  = coords_k / scale_k
    ck1 = frozen_coords_k_plus_1 / scale_k_plus_1        # 이미 detach된 상수
    ...  # 같은 part_id끼리 최근접 거리 hinge, uni_section_v18과 무관한 신규 함수
```
- `k=17`(첫 섹션)은 비교 대상이 없으므로 continuity loss 없이 시작.
- `k=16`은 `k+1=17`만 비교. `k<=15`부터는 `k+1`, `k+2` 둘 다 비교 가능하지만, **`k+1`과의 continuity에
  더 큰 가중치, `k+2`와는 완만한 가중치**(2단계 떨어진 섹션이라 직접 인접보다는 약한 제약)를 준다.
- candidate 파트가 `k+1`에서 z_gate≈0(사실상 삭제)이었다면, 그 좌표와 비교해도 무해하다(두께가
  이미 0에 가까워 물리적 영향 없음) — 필요하면 continuity 기여도에 `z_gate` 가중치를 곱해도 되나
  필수는 아니다.

---

## 5. 목표 Mp 생성 (Gaussian Process) — 변경 없음

```python
def generate_gp_target_mp(seed=None):
    """initial_section.md §2 실측 Area/Mp 표를 기준으로 GP 스무딩된 +/-50% 목표 생성.
    17개 값을 사전에 한 번에 생성해두고, 순차 루프의 각 k 스텝에서 그 중 k번째 값을 읽어 쓴다
    (목표 자체는 여전히 공간 상관 있게 미리 정해지며, 순차 학습 방식과 무관)."""
    initial_mp = torch.tensor([21_637_199, 23_286_535, 24_996_460, 26_766_975, 28_598_079,
                                30_489_775, 32_442_063, 34_454_943, 36_528_416, 38_662_482,
                                40_857_143, 43_112_398, 45_428_249, 47_804_695, 50_241_738,
                                52_739_377, 55_297_613], dtype=torch.float32)  # k=17..1 순서
    sections = torch.arange(17, dtype=torch.float32)
    sigma, ell = 0.25, 3.0
    K = sigma**2 * torch.exp(-(sections[:, None] - sections[None, :])**2 / (2 * ell**2))
    alpha = torch.distributions.MultivariateNormal(torch.zeros(17), covariance_matrix=K + 1e-6*torch.eye(17)).sample()
    target_mp = initial_mp * (1 + alpha)
    return target_mp   # index 0 -> k=17, index 16 -> k=1
```
사후 검증(realizability)은 이전과 동일하게 `compute_edge_mp_pna`/`calculate_mpl`로 학습 시작 전 1회.

---

## 6. 실행 순서 및 검증

### 6.1 Stage 0 — Static Shape/Index Dry-Run (필수, 학습 아님)

학습 루프 진입 전, `k=17` 섹션 하나만 생성해 gradient 없이(`torch.no_grad()`) 검증한다(순차 방식이라
17섹션 전체를 미리 만들 필요가 없음 — joint 방식 때와 달리 이 단계가 훨씬 가벼워진다):
- `build_section_state(17, None)`이 8열 노드 feature를 정상 생성하는지.
- `CGDN.forward()` 한 번 호출이 에러 없이 끝나는지, `log_alpha` shape이 `(5,)`인지.
- GPU 메모리 확인(윈도우가 최대 3섹션이므로 joint 방식보다 메모리 여유가 훨씬 큼 — OOM 위험 낮음).

### 6.2 Primary Path

Stage 0 통과 시, `k=17`부터 바로 순차 루프(§2.1)를 실행한다. 3~5섹션 toy 학습은 만들지 않는다(이미
매 스텝이 최대 3섹션이라 그 자체로 "축소" 성격을 갖는다는 점도 참고).

### 6.3 Defensive Fallback (조건부)

아래가 **실제로 관측된 경우에만** 특정 k 구간만 축소 재현(예: k=17,16,15 세 섹션만)으로 원인을 좁힌다:
1. 특정 섹션 학습 중 loss NaN/Inf.
2. `N_EPOCH_PER_SECTION` 내에 수렴하지 않음(원인 불명).
3. 섹션 전환(`finalize_section` → 다음 `k`) 직후 급격한 loss 스파이크.

### 6.4 Wall-clock 참고

순차 방식은 17번의 개별 학습(각 windows 최대 3섹션)이라, 이전 joint 방식(17섹션을 한 번에 250 epoch)
보다 전체 실행 시간이 길어질 수 있다(대략 "섹션당 epoch 수 × 17"에 비례). `N_EPOCH_PER_SECTION`은
처음엔 작게(예: 150) 시작해 실측 후 조정할 것 — 사용자가 이전에 확정한 "축소 학습 강제 금지" 원칙은
윈도우 크기(최대 3섹션)에 이미 반영되어 있으므로, epoch 수 자체는 자유롭게 실측 조정 가능하다.

---

## 7. 산출물 체크리스트

- [ ] `uni_section_v18_sequential.py` — `build_section_state`, `train_window`, `finalize_section`,
      `continuity_loss`, `generate_gp_target_mp`(§5 그대로), 순차 바깥 루프(§2.1).
- [ ] Stage 0 dry-run 통과 로그(k=17 섹션 하나 기준).
- [ ] 학습 산출물: 섹션별 체크포인트 또는 통합 체크포인트, k별 loss curve, 연속 파트 두께가 k=17에서
      결정된 후 k=16..1 전체에서 상수인지 확인하는 plot/assert(요구사항 4 최종 검증), **k별 candidate
      게이트 최종값**(어느 섹션에서 patch가 살아남았는지 표/히트맵 — 요구사항 5 검증).
- [ ] **[중요] 기존 `AI_design_v1.py`는 joint 방식으로 작성되어 있어 이 문서와 더 이상 맞지 않는다** —
      이 문서(§1~§5)에 따라 별도 세션에서 재작성 필요(시각화 코드 `visualize_training_multi`,
      `plot_sections_3d_plotly_multi`는 섹션별 결과를 모아 재사용 가능할 것으로 예상되나, 그 자체도
      "17섹션 동시 학습 히스토리" 가정을 "k별 순차 히스토리 병합"으로 바꿔야 함).
- [ ] (조건부) §6.3 트리거 시에만 축소 재현 로그.

<details>
<summary>숙의 과정</summary>

### /synod idea 세션 (command_v1.md 최초 작성, joint 방식으로 잘못 구현됨)
- Gemini/OpenAI/Claude가 9열 노드, composite_key 분기, 3-stage curriculum 등을 제안했으나, 실제로는
  "위쪽 detach + 섹션별 순차 학습"이라는 idea_v1.md 원안의 핵심을 반영하지 못하고 joint 배치 학습으로
  귀결됨 — 이후 여러 세션(2026-07-27~28)을 거쳐 뒤늦게 발견.

### /synod debug 세션 (2026-07-27, part 삭제/재등장 정정)
- 사전 JSON 스케줄 폐기, 학습된 게이트로 대체 — 이 결정 자체는 이번 순차 재작성에서도 유효(§3.3).

### /synod debug 세션 (2026-07-28, 이번 정정 — 순차 아키텍처로 §1~§4 전면 재작성)
- 사용자 지적: "17층을 위에서부터 하나씩 하라니까, 이건 전부 다 한번에 한거잖아" — joint 방식이
  idea_v0.md/idea_v1.md의 순차 요구사항을 어긴다는 근본 지적.
- Gemini(Architect, flash, conf 95): 섹션별 독립 파라미터(`SectionModule`), 윈도우 슬라이딩
  의사코드, k=17에서 연속 파트 두께 고정 후 영구 상수화, 섹션별 독립 `log_alpha`(사전 (17,2) 확장
  불필요) 제안 — 실제 채택.
- OpenAI(Explorer, o3, conf 71): 윈도우 이탈 시 optimizer state 정리(`optim.state.pop`) 필요성,
  게이트 temperature도 같이 얼려야 함(안 그러면 train/frozen sparsity 불일치), 마지막 "Global
  Relaxation" pass(전 섹션 좌표만 미세 조정) 필요 가능성 등 실무적 함정을 지적 — §3.3, §2.2 관련
  주의사항으로 반영.
- Claude(Judge): 두 안 모두 실제 `uni_section_v18.py` 함수/구조와 정합적이라 골격으로 채택. 인덱싱
  방향(코드의 `section_id` 0/16 배정이 문서 주석과 반대였던 사실)을 실제 `scale_factor()` 수식으로
  재검증해 이 문서 서두에 명시.

</details>
