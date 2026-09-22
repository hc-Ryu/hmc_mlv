# idea_v1_7.md — 노드 단위 스파이크 해결 + 연속성 단순화 설계 아이디어

작성일: 2026-08-09 (`/synod idea` 세션, Gemini flash conf95→98 + OpenAI gpt4o conf70→75)
근거: `docs/review/review_v1_6.md`(v1.6 붕괴 재발 리뷰), `reports/v1_6/`(실행 로그·대시보드),
그리고 이번 세션 직전 대화에서 확인한 신규 발견 — 단일 노드가 주변 대비 튀어나오는 증상은
`l_continuity`(섹션 간 비교)가 아니라 `uni_section_v18.compute_smoothness_loss_angle()`
(섹션 **내부** 노드별 각도 매끄러움, 불변 파일)의 두 가지 결함이 직접 원인일 가능성이 높다:
(a) `torch.mean()`으로 전체 ~1581개 노드에 걸쳐 평균을 내어 소수의 심한 스파이크가 희석됨,
(b) 좌우 이웃을 x좌표 대소비교로만 판별해 스파이크로 인해 좌우 판정이 꼬이면 해당 노드를
검사에서 통째로 건너뛰어(`continue`) 가장 심하게 뒤틀린 노드가 오히려 페널티 0을 받음.

**사용자 지시사항**: `continuity_weight_schedule`(및 그 위에 얹혔던 v1.6 폐루프 제어)은
단순화하고, 설계 노력은 smoothness loss 보강에 집중할 것.

---

## 아이디어 평가

### 1. 정적 토폴로지 맵 + LogSumExp 풀링 스무스니스 손실 (최우선, P0)

**설계**: `AI_design_v1_6.py`(→v1.7)에 신규 함수를 추가한다. `uni_section_v18.py`는 수정하지
않는다(기존 `compute_smoothness_loss_angle()`은 그대로 두고 **보강 항을 추가**하는 방식).

```python
# 1) 정적 이웃 맵 — epoch 0에 1회만 구축(data.edge_index 기반, 물리 엣지 edge_type==0)
#    design 세션에서 코드 확인: data.edge_index는 학습 전 구간 동일 텐서로 재사용되며,
#    동적 분할(compute_segment_ids)은 두께 pooling 그룹(seg_ids)만 바꿀 뿐 메쉬 그래프(노드-엣지
#    연결) 자체는 절대 바뀌지 않는다 — 따라서 "정적" 맵이 전체 학습 구간에서 항상 유효하다.
def build_static_neighbor_map(edge_index, edge_attr, num_nodes):
    """섹션 내 각 노드의 좌/우 이웃을 초기 x좌표 기준으로 1회만 고정한다.
    이후 좌표가 아무리 뒤틀려도(스파이크 발생) 이웃 판정 자체는 흔들리지 않는다 —
    원본 compute_smoothness_loss_angle()의 '매 스텝 x좌표 재비교' 버그(§0(b))를 근본적으로
    우회한다."""
    ...  # 초기 좌표 기준으로 seg/direction별 좌우 1개씩 고정, (N, 2) 인덱스 텐서 반환

# 2) Laplacian형 편차 + LogSumExp 풀링
def compute_robust_smoothness_loss(new_coords, static_neighbor_map, alpha=SMOOTH_LSE_ALPHA):
    """[v1.7] 정적 이웃으로 계산한 편차를 LogSumExp로 집계 — mean처럼 희석되지 않고
    (§0(a) 문제 해결), 동시에 hard max처럼 그래디언트가 한 노드에 무한정 쏠려 폭발하지도
    않는다(design 세션 Critic 라운드에서 수학적으로 증명됨, 아래 §1.2 참고)."""
    left = new_coords[static_neighbor_map[:, 0]]
    right = new_coords[static_neighbor_map[:, 1]]
    laplacian = new_coords - 0.5 * (left + right)          # 직선에서 벗어난 정도
    dev = torch.norm(laplacian, dim=-1)                     # (N,)
    return (1.0 / alpha) * torch.logsumexp(alpha * dev, dim=0)
```

- **실현 난이도**: 중 — `build_static_neighbor_map()`은 1회성 전처리(학습 시작 전), 손실
  함수 자체는 표준 PyTorch 연산으로 낮은 난이도. design 세션에서 "이게 정말 단순화냐"는
  반론(OpenAI)이 있었으나, **사용자 지시는 "연속성은 단순화, smoothness는 설계 노력을
  투입"이었으므로 이 배분은 지시와 정확히 일치한다**(Critic 라운드에서 Gemini가 재확인,
  아래 §1.2).
- **크래시 위험**: 낮음 — LogSumExp의 안전성이 수학적으로 검증됨(아래 §1.2).

#### §1.2 design 세션 검증: LogSumExp는 그래디언트 폭발을 일으키지 않는다

1라운드에서 OpenAI가 "mean 대신 max 계열로 바꾸면 v1.4 스타일 그래디언트 폭발을 재현할
위험이 있다"고 우려했다. 2라운드(Critic)에서 Gemini가 수학적으로 반박했다:

$$\text{LSE}(\mathbf{s}) = \log\sum_i e^{s_i}, \quad
\frac{\partial \text{LSE}}{\partial s_j} = \frac{e^{s_j}}{\sum_i e^{s_i}} = \text{softmax}(\mathbf{s})_j \in [0, 1]$$

softmax 출력은 항상 [0,1] 구간에 있으므로, LogSumExp의 개별 노드에 대한 그래디언트는 **항상
1.0 이하로 유계(bounded)**다. hard max(불연속 서브그래디언트)나 검증 없는 고차 거듭제곱과
달리 폭발 메커니즘 자체가 없다. OpenAI는 3라운드에서 동일 우려를 반복했으나 이 증명에 대한
새로운 반박을 제시하지 못했다 — Judge 판정: 채택.

- **추가 필요 사항(Critic 라운드, Gemini 지적, 채택)**: LogSumExp는
  $\frac{1}{\alpha}\log\sum e^{\alpha s_i}$ 형태로 온도(스케일) 파라미터 $\alpha$를 명시해야
  한다 — $\alpha$가 너무 작으면 사실상 mean과 비슷해져 원래 문제(희석)가 재발하고, 너무 크면
  수치적으로 불안정해질 수 있다. `SMOOTH_LSE_ALPHA` 상수를 신설하고, 좌표 편차의 실제 스케일
  (mm 단위, 관측된 스파이크 편차가 수십 mm 수준)에 맞춰 초기값을 정하고 §6 검증 계획에서
  재보정한다.

### 2. `continuity_weight_schedule` 단순화 — 폐루프 제어 완전 제거 (최우선, P0)

**설계**: v1.6의 `get_adaptive_continuity_weight()`(폐루프, `max(w_base, w_adaptive)` 래칫
버그의 원인)를 삭제하고, 원래 `continuity_weight_schedule()`에 하한선 한 줄만 추가한다.

```python
def continuity_weight_schedule(epoch, stage2_end, w_max=1.0, w_min=0.05, beta=0.1,
                                w_floor=CONTINUITY_W_FLOOR):
    """[v1.7] v1.5 sigmoid 그대로 + 하한선(review_v1_5.md 1차 원인 대응)만 추가.
    v1.6의 폐루프 제어(get_adaptive_continuity_weight, 상태 저장, EMA, eta 튜닝)는 완전히
    제거한다 — 사용자 지시(연속성 단순화) + review_v1_6.md에서 확인된 래칫 버그 재발 방지."""
    w_base = w_min + (w_max - w_min) / (1.0 + math.exp(beta * (epoch - stage2_end)))
    return max(w_base, w_floor)
```

- `train_step_multi()`의 `w_continuity_val` sentinel 패턴(v1.6에서 이미 확립)은 그대로
  유지 — 다만 호출부에서 `get_adaptive_continuity_weight()` 대신 위 단순화된
  `continuity_weight_schedule()`을 직접 호출하도록 되돌린다. `continuity_state` 딕셔너리,
  EMA 갱신 코드, `CONTINUITY_TAU`/`CONTINUITY_ETA`/`CONTINUITY_EMA_BETA` 상수는 모두 삭제.
- **왜 고정 상수가 아니라 여전히 하한 있는 sigmoid인가**: review_v1_5.md의 1차 원인(무하한
  감쇠 → 0.05까지 떨어져 형상 정합성 소멸)은 여전히 유효한 문제이므로, "하한만 있고 폐루프는
  없는" 가장 단순한 형태로 그 교훈만 남긴다.
- **크래시 위험**: 없음 — 상태 저장이 사라지므로 v1.6의 래칫 버그 자체가 구조적으로
  재발 불가능하다(더 이상 "한 번 켜지면 안 꺼지는" 상태가 존재하지 않음).

### 3. (참고, 채택하지 않음) "근본 유인(incentive)" 자체를 건드리는 방안

1라운드에서 OpenAI가 "smoothness를 완벽히 고쳐도, Mp(관성모멘트) 목표를 값싸게 달성하려는
근본적인 유인 자체가 남아있으면 다른 형태로 우회할 수 있다"는 우려를 제기했다. 2라운드에서
Gemini가 "관성모멘트를 늘리려는 압력 자체는 정상적인 물리 최적화 행동이며, 스파이크가
문제였던 것은 '얇고 뾰족한 형태로 몰래 우회할 수 있는 구멍'이 smoothness 손실에 있었기
때문 — 그 구멍을 수학적으로 완전히 막으면(정적 이웃 맵 + LogSumExp) 우회 경로 자체가
사라지므로 이 우려는 이번 수정을 막을 이유가 아니다"라고 반박했고, Claude(Judge)도 동의한다.
다만 **완전히 기각하지는 않는다** — §6 검증 계획에서 v1.7 재실행 후에도 다른 형태의 우회
(예: 매우 좁은 각도의 반복적 지그재그 등 정적 이웃 맵으로도 못 잡는 패턴)가 나타나는지
관찰 항목으로 남긴다.

### 4. (참고, 채택하지 않음) 이웃 판정에 메쉬 토폴로지/호 길이 순서 사용

1라운드에서 OpenAI가 "x좌표 비교보다 메쉬 엣지 토폴로지나 호 길이 순서가 더 견고한
이웃 정의 아니냐"고 제안했다. 아이디어 1의 "정적 이웃 맵"이 정확히 이것을 구현한다 —
epoch 0의 초기 좌표(아직 뒤틀리기 전) 기준으로 좌우를 고정하므로, 사실상 초기 형상의
위상학적 순서를 그대로 얼려서 쓰는 것과 동등하다. 별도 아이디어로 분리할 필요 없이 아이디어
1에 이미 흡수됨 — 중복 채택 방지를 위해 별도 항목으로는 기각.

---

## 권장 구현 순서

1. **[P0] 아이디어 2(연속성 단순화)** — 코드 삭제가 대부분이라 가장 빠르고 낮은 리스크.
   먼저 적용해 v1.6의 래칫 버그가 완전히 사라졌는지(로그의 `w_continuity`가 정상적으로
   감쇠하는지) 확인.
2. **[P0] 아이디어 1(정적 이웃 맵 + LogSumExp 스무스니스)** — 노드 스파이크 직접 해결.
   `SMOOTH_LSE_ALPHA` 초기값은 관측된 스파이크 편차 스케일(수십 mm)을 참고해 설정하고,
   재실행 로그에서 개별 노드 최대 편차 대비 손실 반응성을 확인 후 재보정.
3. 1·2 동시 적용 후 재실행 — `reports/v1_7/` 결과에서 (a) `w_continuity`가 실제로 감쇠하는지,
   (b) 노드 단위 스파이크가 사라지는지, (c) Mp 오차가 기존 대비 크게 나빠지지 않는지 확인.

**§6 불변 원칙 준수 확인**: `uni_section_v18.py`(`compute_smoothness_loss_angle()` 포함)는
전혀 수정하지 않는다 — 신규 항은 순수 보강(additive)이다. §4(중립 초기화+지연 엔트로피),
§7(R1~R3 롤백)도 이번 아이디어들과 무관하게 그대로 유지.

---

<details>
<summary>Synod 세션 상세 (idea 모드, Solver→Critic 2라운드, Defense 라운드는 생략)</summary>

**Round 1 (Solver)**: Gemini(Architect, conf95)가 정적 토폴로지 맵 + LogSumExp 풀링
스무스니스 손실과 결정론적 연속성 감쇠(하한 포함)를 제안하고 우선순위표로 정리했다.
OpenAI(Explorer, conf70)는 근본 유인 미해결 가능성, max 계열 손실의 그래디언트 폭발 위험,
정적 맵 구축이 "단순화"라는 사용자 지시와 모순되는 것 아니냐는 우려, smoothness 수정이
기존 연속성 버그를 다른 형태로 드러낼 위험 등 4가지를 제기했다.

**Round 2 (Critic)**: Gemini가 LogSumExp 그래디언트가 softmax로 [0,1] 유계임을 수식으로
증명해 OpenAI의 그래디언트 폭발 우려를 반박했고, "사용자 지시는 연속성만 단순화하라는
것이었으므로 smoothness에 설계 노력을 투입하는 것은 모순이 아니다"라고 정리했다. 근본 유인
우려에 대해서는 "스파이크라는 우회 경로 자체를 수학적으로 막으면 우려가 해소된다"고
반박하되 완전히 기각하지는 않고 §6 관찰 항목으로 남겼다. OpenAI critic(conf75)은 동일
우려를 반복했으나 Gemini의 수학적 증명에 대한 새 반박을 제시하지 못했다.

**Judge(Claude) 최종 판정**: 아이디어 1(정적 이웃 맵+LogSumExp)과 아이디어 2(연속성 단순화)
를 P0로 채택, 근본 유인 문제는 관찰 항목으로 격하(§6), 메쉬 토폴로지 이웃 정의 제안은
아이디어 1에 이미 흡수되어 중복 기각.

**신뢰 점수**: Gemini 95(solver)→98(critic, 수학적 증명 반영), OpenAI 70(solver)→75(critic,
반복 주장으로 소폭 상승에 그침 — 새 근거 부재). **최종 신뢰도 89%**.

</details>
