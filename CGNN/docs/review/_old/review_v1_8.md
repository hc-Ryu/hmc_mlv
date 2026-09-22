# review_v1_8.md — AI_design_v1_8.py 실행 결과 리뷰 (잔존 스파이크 원인 분석)

작성일: 2026-08-09 (`/synod review` 세션, Gemini flash conf92 + OpenAI o3 conf83, Critic 라운드로 수렴)
대상: `reports/v1_8/AI_design_v1_8_report.md`, `reports/v1_8/AI_design_v1_8_3d.html`,
`reports/v1_8/AI_design_v1_8_result.png`, `AI_design_v1_8.py`, `uni-section/code/uni_section_v19.py`

---

## 요약

**사용자 가설은 코드 근거로 타당하다.** Mp(소성모멘트)를 맞추기 위해 특정 노드를 PNA(중립축)에서
멀리 밀어내는 것이 "가장 값싼 지름길"이 되도록 손실 함수 구조 자체가 설계돼 있고, 이를 억제해야 할
smoothness 관련 손실(각도 기반 + v1.7의 LSE 보강)은 구조적·수치적으로 그 압력을 이기지 못한다.

v1.8은 review_v1_7.md가 지적한 두 원인 중 **①(LSE 온도 미보정, T=0.01→1.5mm)만 고쳤고, ②(정적
이웃 맵의 지역성 — 전역 자기교차 미탐지)는 그대로 남아 있다.** 게다가 ①을 고치는 과정에서 **온도
재보정으로 `l_smooth_lse`의 절댓값 자체가 줄었는데 가중치(`W_SMOOTH_LSE=0.25`)는 그대로 두어,
v1.7보다 오히려 smoothness 억제력이 약해졌을 가능성**이 있다(OpenAI 지적, Critic 라운드에서 검증됨).
추가로 두 모델과 Critic 라운드가 모두 독립적으로 수렴한 새로운 원인은 **DELETED(삭제 확정) 노드가
마스킹 없이 LSE 계산에 계속 포함되어, 물리적으로 무의미한 좌표 발산이 유효 노드에게 가야 할
그래디언트를 빼앗는 문제(Ghost Gradient Hijacking)**다.

---

## 발견된 문제

### [ERROR] 1. Mp/질량 관련 손실 가중치가 smoothness 관련 손실보다 20~40배 큼

- **근거(코드, `run_training_multi` 기본 `weights` 딕셔너리 및 손실 결합부)**:

  | 항목 | 가중치 | 집계 방식 |
  |---|---:|---|
  | `w_mass` | 10.0 | 목표 면적 초과분 이차 벌점 |
  | `w_area_floor` | 10.0 | 목표 면적 90% 미달 시 이차 벌점 |
  | ALDA(`L_alda_effective`) | gate≈1.0 배율 | Mp 직접 조정 |
  | `w_smooth` | 0.5 | `torch.mean()` — 전체 ~2992 물리 엣지 각도 오차 평균 |
  | `W_SMOOTH_LSE` | 0.25 | LogSumExp — 정적 이웃 노드 전체 중 최악값 위주 |

  Mp/면적을 직접 조정하는 항(10.0~10.0+ALDA)이 형상을 매끄럽게 유지하는 항(0.5+0.25=0.75) 대비
  **13~27배** 크다. 소성모멘트는 대략 `Σ(두께 × 항복강도 × |y - y_PNA|)`에 비례하므로, 면적(질량)을
  늘리지 않고 Mp를 올리는 가장 효율적인 방법은 소수 노드를 PNA에서 멀리 보내는 것이다 — 이 "지렛대
  전략"에 대한 대가는 smoothness 손실 0.75 수준밖에 안 되지만, 이를 안 쓰고 면적을 고르게 늘리면
  `w_mass`/`w_area_floor`(10.0)의 직격탄을 맞는다. 즉 옵티마이저 입장에서 스파이크 전략이 구조적으로
  더 싸다.

### [ERROR] 2. `compute_smoothness_lse`가 17섹션 전체를 단일 스칼라로 집계 — "한 번에 최악 1~2곳만" 억제(Whack-a-Mole)

- **근거(코드)**: `compute_smoothness_lse()`는 `neighbor_map`에 담긴 **모든 섹션의 모든 정적 이웃
  노드**(수백 개)의 `sq_deviation`을 `dim=0` 하나로 묶어 `logsumexp`를 계산한다. 섹션별 분리가 없다.
  LogSumExp는 온도가 스케일 대비 충분히 작으면 사실상 `max()`에 근접하는 연산이므로, 이 손실의
  그래디언트는 **그 순간 가장 심한 편차를 가진 노드(들)에게만 집중**되고 나머지 섹션에서 동시에
  자라는 스파이크는 방치된다. 가장 심한 스파이크가 완화되면 다음으로 심한 스파이크가 그래디언트를
  넘겨받는 순환이 발생 — 이것이 "가장 극단적인 단일 스파이크는 사라졌지만 완만한 수준의 스파이크는
  남아있다"(review_v1_7.md, 사용자 관찰)는 패턴과 정확히 일치한다.
- Gemini와 OpenAI가 Solver 라운드에서 독립적으로 이 메커니즘을 지목했고, Critic 라운드에서 서로
  교차검증했다.

### [ERROR] 3. DELETED 노드가 `compute_smoothness_lse`에 마스킹 없이 포함 — Ghost Gradient Hijacking

- **근거(코드 주석 및 로직)**: `compute_smoothness_lse()`의 독스트링은 "DELETED 파트에 속한 노드도
  그대로 포함한다"고 명시하며, 원본 `compute_smoothness_loss_angle()`과 동일하게 동작시키기 위한
  의도적 설계였다(v1.7 구현 세션에서 OpenAI의 마스킹 제안을 "범위 확장"으로 기각). 그러나 DELETED
  확정 노드는 `t_final≈0`이라 Mp/mass/collision 등 어떤 물리 손실로부터도 직접적인 억제를 받지 않는다
  — 좌표만 계속 최적화 대상으로 남아 자유롭게 발산할 수 있다.
- **메커니즘**: DELETED 노드의 좌표 편차가 LSE의 전역 `max`에 해당하는 값을 차지하면(가장 억제받지
  않는 노드이므로 가능성이 높다), 이 손실의 유한한 그래디언트 "예산" 전체가 물리적으로 무의미한
  DELETED 노드 쪽으로 흘러가고, 실제로 살아남아 제조에 반영될 노드(candidate 파트 중 7/17 섹션
  생존분 등)의 스파이크는 그래디언트를 받지 못한 채 방치된다.
- Gemini(Solver, Critic)와 OpenAI(Solver) 모두 독립적으로 이 가설에 도달했고, Critic 라운드
  (Gemini conf95, can_exit=true)에서 "DELETED 마스킹이 없으면 가중치를 아무리 올려도 오염된
  그래디언트를 더 세게 역전파할 뿐"이라는 논리로 이것이 **가중치 조정보다 선행돼야 하는 버그**임을
  확정했다.

### [WARNING] 4. review_v1_7.md §2(정적 이웃 맵의 지역성)는 v1.8에서 전혀 다루지 않음

- `build_static_neighbor_map()`은 v1.7 도입 이후 v1.8까지 코드 변경이 없다. epoch 0의 초기 좌표
  기준으로 "당시 이웃"만 영구 고정하므로, 학습 중 원래 이웃이 아니었던 두 지점이 서로 가까워지거나
  교차하는 전역적 자기교차는 여전히 구조적으로 탐지 불가능하다. v1.8이 손댄 `compute_mesh_order_loss`
  Top-K 풀링은 "이미 연결된 물리 엣지" 안에서만 작동하므로 이 구조적 한계를 해소하지 못한다.
- 리포트 상 여러 섹션(0, 1, 2, 7, 8, 11, 16)의 "target miss"가 이런 비전역적 왜곡과 함께 나타날
  가능성이 있으나, epoch별 `l_order`/`l_smooth_lse` 실측 로그가 저장되어 있지 않아 이번 리뷰만으로는
  직접 확증할 수 없다(§한계 참고).

### [INFO] 5. Critic 라운드에서 OpenAI 응답에 확인 불가능한 근거 인용(신뢰도 하향 반영)

- OpenAI CLI의 Critic 라운드 응답이 `cfg.yaml`, "PR 코멘트(2023-10-17)", "학습 로그(2024-01-14)",
  "spike 재현 로그(v1.8_20240210)", "DELETED 노드가 spike-cluster의 73%" 등 **이 저장소에 실재하지
  않는 파일·로그·수치를 구체적으로 인용**했다. 이 프로젝트에는 `cfg.yaml`이 없고(가중치는
  `AI_design_v1_8.py` 내 Python 상수), 실행 콘솔 로그도 저장돼 있지 않다(§한계 참고). 이는 명백한
  환각(hallucination)이다.
- 다만 해당 응답의 **결론 자체**(DELETED 마스킹이 가중치 조정보다 선행해야 한다)는 Gemini가 실제
  코드만 근거로 독립적으로 도달한 결론과 일치하고, Gemini의 Critic 응답은 이런 조작된 근거를 인용하지
  않았다. Trust Score 산정 시 OpenAI Critic 응답의 신뢰성(C, Credibility)을 낮게 책정했고, 최종
  결론은 이 환각을 제외한 코드 근거만으로 재확인된 부분만 채택했다.

---

## v1.8이 review_v1_7.md 대비 고친 것 / 안 고친 것

| review_v1_7.md 원인 | v1.8 조치 | 평가 |
|---|---|---|
| §1 `SMOOTH_LSE_TEMPERATURE=0.01`(hard-max, 스케일 폭증) | 1.5mm로 재보정 | ✔ 스케일 정상화는 해결. 단, `W_SMOOTH_LSE=0.25`를 동반 재조정하지 않아 **억제력 자체는 v1.7보다 약해졌을 수 있음**(위 [ERROR] 1) |
| §2 정적 이웃 맵의 지역성(전역 자기교차 미탐지) | 미변경 (`build_static_neighbor_map` 그대로) | ✘ 전혀 다루지 않음. Order 손실 Top-K 풀링(§1, v1.8 핵심 변경)은 별개 문제(Order 손실의 mean-dilution)를 고친 것이지 이 구조적 한계와 무관 |

---

## 권장 사항 (우선순위순, Critic 라운드 수렴 결과)

두 모델 모두 Solver 라운드에서 서로 다른 1순위(Gemini: DELETED 마스킹, OpenAI: 가중치 상향)를
제시했으나, Critic 라운드 교차검증에서 **"오염된 입력(DELETED 노드) 위에 가중치만 올리면 오염된
그래디언트를 더 세게 역전파할 뿐"**이라는 논리로 아래 순서에 수렴했다(Gemini conf95, can_exit=true).

1. **[최우선/버그 수정] DELETED 노드 마스킹** — `compute_smoothness_lse()`(및 필요시
   `compute_smoothness_loss_angle()`)에서 `t_final < ε`(예: 0.1mm)인 노드를 계산 대상에서 제외한다.
   좌표 자체를 freeze(grad 차단)하는 방법도 대안. 이것이 P0인 이유: 가중치 조정이나 섹션별 분리는
   "오염된 그래디언트 신호"를 전제로 하는 튜닝이라, 오염원을 먼저 제거하지 않으면 효과가 검증되지
   않는다.
2. **[동시 적용] 손실 가중치 재조정 + 섹션별 LSE 분리** — DELETED 마스킹 직후 재조정해야 유효한
   측정이 된다:
   - `W_SMOOTH_LSE`를 0.25 → 1.0~2.0 수준으로 상향(T=1.5 재보정으로 줄어든 절댓값을 보상).
   - `compute_smoothness_lse`를 섹션별로 분리 집계(예: `scatter`로 섹션별 `logsumexp` 후 평균/Top-K)
     해 "한 섹션의 최악값이 다른 섹션의 그래디언트를 죽이는" whack-a-mole을 완화.
3. **[후순위, 구조적 개선] 동적/확장된 이웃 관계 탐지** — review_v1_7.md §2의 구조적 한계(전역
   자기교차) 자체를 해소하려면 정적 이웃 맵을 k-NN 기반 동적 갱신이나 half-plane 테스트로 대체해야
   한다. 구현 난이도와 검증 비용이 높으므로 P0/P1로 그래디언트 흐름을 먼저 정상화한 뒤 착수를
   권장(Critic 라운드 공통 판단).
4. **[검증 계획] 마스킹 전/후 ablation + 로깅 강화** — `l_smooth_lse`, `l_order`, top-5 노드 편차를
   epoch별로 로깅해 저장한다. 지금까지 v1.5~v1.8 리뷰가 모두 "콘솔 로그가 저장되지 않아 실측 곡선을
   확인할 수 없었다"는 한계를 반복하고 있다 — 다음 실행부터는 반드시 로그 파일을 남길 것.

---

## 추가 검토 (2차 `/synod review`) — 위 4가지 외의 새로운 붕괴 원인

사용자 요청으로 `AI_design_v1_8.py` 전체(2011줄)를 처음부터 끝까지 재검토하며, 위 §1~4(가중치
불균형/global LSE/DELETED 마스킹/정적 이웃 맵)와 무관한 **새로운** 스파이크 원인을 추가로 조사했다.
Gemini(conf92, can_exit=true)와 OpenAI(conf88, can_exit=false)가 강하게 수렴한 항목을 코드로
직접 재확인한 결과는 다음과 같다. 두 모델 응답 모두에서 실제 코드와 다른 주장(§아래 "기각된 주장"
참고)이 섞여 있어, **코드 라인으로 직접 재확인된 것만** 채택했다.

### [ERROR] 6. Stage 4의 게이트 강제 이진화(hard flip)가 경계 섹션에 불연속 충격을 가하고, 이를 흡수할 예산이 없음

- **근거(코드, `run_training_multi` Stage 4 진입부)**: 메인 학습 루프 종료 시 "애매한" 게이트
  (`|log_alpha| <= 2.4`)가 하나라도 있으면(`ambiguous > 0`) Stage 4가 자동 발동한다. 이때
  `model.log_alpha_candidates[k, c] = 5.0 if hard_exists[k, c] else -5.0`로 **모든 애매 게이트를
  단일 스텝에 ±5.0으로 강제 이진화**하고 `requires_grad_(False)`로 완전히 동결한다. 이는 정확히
  살아남은 run과 삭제된 run의 경계 섹션(리포트 기준 후보 파트가 7/17 섹션에서만 생존 — 경계는
  섹션 9/10 부근)에서 `t_final = t_raw * z_gate`가 한 스텝에 불연속적으로 켜지거나 꺼진다는 뜻이다.
- **흡수 예산 부족**: 이 충격을 재수렴시켜야 할 Stage 4는 (a) 학습률이 원래의 10%(`lr * STAGE4_LR_SCALE
  = 1e-4`)로 급감, (b) 딱 `STAGE4_EPOCHS=200` epoch만 배정, (c) **optimizer를 새로 생성**해 AdamW의
  1·2차 모멘텀 버퍼가 초기화된다(기존 관성 소실). 반면 smoothness 관련 억제력(§1~3의 약점)은 전혀
  개선되지 않은 채로 이 구간에 그대로 진입한다 — "형상이 가장 크게, 가장 갑자기 바뀌어야 하는 순간"과
  "smoothness 억제력이 가장 약한 상태"가 정확히 겹친다.
- Gemini와 OpenAI가 Solver 라운드에서 독립적으로 이 메커니즘(및 optimizer 재생성으로 인한 모멘텀
  손실)에 도달했고, 둘 다 이 원인의 심각도를 최상위로 평가했다(Gemini: Critical/98%, OpenAI: Very
  High/확신 다수).

### [ERROR] 7. `compute_shape_continuity_loss`의 최근접-이웃(Chamfer-style) 매칭이 스파이크를 놓칠 수 있음

- **근거(코드)**: 인접 섹션 간 연속성 검사가 `dist = torch.cdist(ca, cb, p=2); min_dist, _ =
  torch.min(dist, dim=1)`로 이루어진다. 이는 섹션 A의 노드 i가 섹션 B의 **대응되는 같은 위치 노드**와
  얼마나 떨어졌는지가 아니라, **B의 아무 노드든 가장 가까운 것**과의 거리만 본다. 노드 인덱스
  대응(순서 보존)이나 전단사(bijective) 매칭이 전혀 강제되지 않는다.
- **메커니즘**: 스파이크로 인해 노드 i가 원래 위치에서 크게 벗어나더라도, 그 새 위치가 우연히 섹션
  B의 다른 노드 근처(threshold=2mm 이내)라면 이 손실은 위반을 전혀 감지하지 못한다 — 두 인접
  단면의 전체 윤곽이 비슷한 범위 안에 있는 한, "같은 위치의 대응 노드가 서로 멀어졌다"는 스파이크의
  전형적 시그니처를 구조적으로 놓칠 수 있다. candidate 파트(3/4)는 노드 수가 적어 이런 우연한 매칭이
  더 쉽게 발생할 수 있다는 지적도 두 모델 모두에서 나왔다(OpenAI: "경계 스파이크가 주로 candidate
  파트에서 남는 현상과 일치").

### [WARNING] 8. `l_phys_total`의 섹션 간 단순 평균이 국소 붕괴를 희석

- **근거(코드)**: `l_phys_total = torch.stack(l_phys_terms).mean()` — 17개 섹션 각각의 Mp 물리
  손실(Huber)을 **단순 평균**해 하나의 스칼라로 합친다. 이는 review_v1_8.md §2(smoothness의 mean
  희석)와 동일한 패턴이 물리 손실에도 그대로 존재한다는 뜻이다 — 17개 중 1~2개 섹션에서 Mp가 크게
  어긋나더라도(예: 후보 파트의 존재 경계 섹션), 전체 그래디언트에서 1/17 수준으로 희석된다.
- 두 모델 모두 이 패턴을 "이미 알려진 smoothness 희석 문제와 동일한 구조적 결함이 물리 손실에도
  반복된다"는 형태로 지적했고, review_v1_8.md의 기존 진단(§1~3)과 정합적이다.

### [INFO] 9. Collision 손실의 두 가지 추가 희석 지점 (기존 지적의 새 위치에서의 재발)

- **파트-레벨 z_gate 평균화**: `l_collision = usv18.compute_collision_loss_v5(..., z_gate=z_gate_part_avg)`
  에서 `z_gate_part_avg = z_gate.mean(dim=0)`(17섹션 평균)를 사용한다(코드 주석에 "candidate 파트만
  근사가 생김을 인지"라고 명시된 기지 사실). `compute_collision_loss_v5` 내부에서 이 값이
  `pair_gate = z_gate[seg_part] * z_gate[pt_part]`로 collision loss에 곱해지는데, candidate 파트가
  17섹션 중 7섹션에서만 생존하면 평균 z_gate가 ~0.41 수준까지 낮아져, **살아남은 7개 섹션에서도
  collision 억제력이 전체 대비 약 60% 깎인 채로 적용된다.** 이는 §1(가중치 불균형)·§9-a와는 다른,
  "생존 섹션의 collision 억제력이 죽은 섹션들 때문에 부당하게 희석되는" 별개의 경로다.
- **노드 단위 평균화**: `compute_collision_penalty_unclamped`/`compute_collision_loss_v5` 모두
  방향별 위반량을 `violation.sum() / valid.float().sum()`로 **평균**한다 — 한두 노드의 깊은 국소
  관통(송곳형)이 다수의 정상 노드에 의해 평균에서 희석되는, smoothness/물리 손실과 동일한 패턴이
  collision에도 존재한다.

### 기각된 주장 (코드 직접 대조로 반증, 최종 문서에서 제외)

- Gemini가 제기한 "GNN 디코더의 무제한 좌표 변위 출력(Unbounded Delta)"은 **사실이 아니다** —
  실제 코드(`forward()`)에는 `delta_coords = torch.clamp(delta_coords, -self.max_displacement,
  self.max_displacement)`가 존재해 변위가 이미 상한(`max_displacement=50.0`)으로 제한되어 있다.
- OpenAI가 제기한 "gradient clipping이 `clip_grad_value_(model.parameters(), 0.5)`로 값 기준 클리핑을
  한다"는 주장도 **사실이 아니다** — 실제 코드는 `torch.nn.utils.clip_grad_norm_(model.parameters(),
  10.0)`로 전역 노름(norm) 기준 클리핑이며 임계값도 다르다.
- OpenAI 응답에 등장한 "실제 로그 저장을 보면 평균 2~6개 섹션이 gate≈0.3~0.7", "실측 로그에서
  gradient ≈0.06×만 전달되는 것이 확인", "실험 재현(후보 A 발동 후 경계 톱니 잔존)" 등은 review_v1_8.md
  §5에서 이미 확인한 것과 동일한 유형의 환각(hallucination)이다 — 이 프로젝트에는 저장된 콘솔 로그가
  없다(§한계 참고). 결론(Stage4 hard flip이 원인이라는 것) 자체는 코드로 독립 검증되어 채택했지만,
  인용된 수치는 모두 배제했다.

### 우선순위 업데이트

기존 §5의 P0~P3(DELETED 마스킹 → 가중치/섹션별 LSE → 동적 이웃맵 → 검증 로깅)에 아래를 추가한다.
Stage 4 hard-flip은 "형상이 가장 급변하는 구간에서 억제력이 가장 약하다"는 점에서 기존 P0(DELETED
마스킹)만큼이나 근본적이므로, 실무적으로는 병행 적용을 권장한다.

- **[P0과 동급] Stage 4 이진화 완화**: `±5.0` 단일 스텝 강제 대신 수 십 epoch에 걸친 점진적 어닐링
  (예: `log_alpha`를 목표값까지 선형/코사인 보간)으로 대체하거나, hard-flip 직후 몇 epoch은 기존
  optimizer 모멘텀을 유지(재생성하지 않음)하고 LR도 즉시 10%로 낮추지 않고 별도 워밍다운을 준다.
- **[P1과 병행] `compute_shape_continuity_loss`를 인덱스 대응 기반으로 변경**: `torch.cdist` +
  `min(dim=1)` 대신, 두 섹션이 노드 순서가 동일하다는 전제(같은 `build_bpillar_section()` 토폴로지를
  스케일링만 다르게 사용하므로 성립)를 활용해 같은 로컬 인덱스끼리 직접 거리 비교하는 방식으로
  교체한다.
- **[P1과 병행] `l_phys_total`/collision loss의 섹션·노드 평균을 Top-K 또는 LogSumExp로 교체**: 이미
  §1의 `compute_mesh_order_loss`에 적용된 Top-K 풀링 패턴을 물리 손실과 collision 손실에도 동일하게
  적용해, 소수 섹션/노드의 국소 붕괴가 평균에 희석되지 않도록 한다.
- **[검토]** `z_gate_part_avg`를 섹션 전체 평균이 아니라 "생존 섹션만의 평균"으로 바꾸는 안을
  검토한다(코드 주석이 이미 이 근사를 인지하고 있으므로, 정밀화 시 collision 억제력이 정상화된다).

---

## 한계

- 이전 리뷰들과 동일하게, `report.md`/`3d.html`/PNG만 근거로 했다. epoch별 `l_smooth_lse`, `l_order`,
  DELETED 노드 좌표 발산 여부를 보여주는 콘솔 로그가 이번에도 저장되어 있지 않아, 제시된 메커니즘은
  코드 로직과 최종 리포트 수치로부터의 논리적 추론이며 실측 곡선으로 직접 검증되지는 않았다.
- OpenAI Critic 응답의 일부 인용 근거(§5)는 이 저장소에 존재하지 않는 파일/로그였다 — 최종 결론에서
  해당 부분은 제외하고, Gemini의 독립적인 코드 기반 결론과 겹치는 부분만 채택했다.
- DELETED 마스킹 적용 후 실제로 스파이크가 해소되는지는 재실행 전까지 확정할 수 없다 — 다음 버전의
  검증 계획으로 남긴다.

---

<details>
<summary>Synod 세션 상세 (review 모드, Solver 1라운드 + Critic 1라운드로 수렴)</summary>

**Round 1 (Solver)**: Gemini(Architect, conf92, can_exit=true)가 (a) angle loss의 mean 희석,
(b) `compute_smoothness_lse`의 전역 단일 스칼라 집계로 인한 whack-a-mole, (c) DELETED 노드의
Ghost Gradient Hijacking을 코드 근거로 제시. OpenAI(Explorer, conf83, can_exit=false)는 동일 결론에
수렴하되 손실 가중치 표를 정량적으로 비교해 "T=1.5 재보정 후 `l_smooth_lse` 절댓값이 줄었는데
weight는 그대로라 v1.7보다 억제력이 약해졌을 수 있다"는 세부 주장을 추가. epoch 로그 부재를
이유로 can_exit=false.

**Round 2 (Critic)**: 두 모델 모두 상대 주장을 코드 근거로 교차검증하며 conf 75~95로 수렴.
Gemini Critic(conf95, can_exit=true)이 "DELETED 마스킹(P0) → 가중치/섹션분리(P1) → 동적 이웃맵(P2)"
순서를 논리적으로 확정(오염원 제거가 튜닝보다 선행해야 함). OpenAI Critic(conf75, can_exit=true)도
동일 순서에 동의했으나, 근거로 이 저장소에 존재하지 않는 파일/로그를 인용하는 환각이 발견됨 —
결론의 논리 자체는 Gemini의 독립적 코드 분석과 일치하므로 채택하되, 인용된 구체적 수치/파일은
최종 문서에서 배제.

**Judge(Claude) 최종 종합**: 사용자 가설(Mp 압력이 PNA 이격을 유도, smoothness가 억제 실패)은
코드 근거로 확증. 1차 원인은 손실 가중치 불균형(20~40배)과 LSE의 전역 집계 방식, 2차 신규 발견은
DELETED 노드의 마스킹 누락(Ghost Gradient), 3차는 review_v1_7.md §2(정적 이웃 맵 지역성)가 v1.8에서
전혀 다뤄지지 않았다는 사실. 권장 조치 우선순위는 DELETED 마스킹(버그 수정) > 가중치/섹션별 LSE
(튜닝) > 동적 이웃맵(구조 개선)으로 Critic 라운드 합의를 그대로 채택.

**신뢰 점수**: Gemini 92→95(Critic), OpenAI 83→75(Critic, 환각으로 하향 조정하되 핵심 결론은 유지).
**최종 신뢰도 88%** (핵심 원인 진단은 높은 확신, DELETED 마스킹 적용 후 실측 검증은 미완료).

---

**2차 세션 (Solver 1라운드, "§1~4 외 추가 원인" 재검토)**: Claude가 먼저 코드 전체를 재독해
Stage 4 강제 이진화(hard flip)와 `compute_shape_continuity_loss`의 최근접-이웃 매칭 두 후보를
찾아 프롬프트에 포함시켰다. Gemini(Architect, conf92, can_exit=true)와 OpenAI(Explorer, conf88,
can_exit=false) 둘 다 두 후보를 강하게 확증했고, 독립적으로 `l_phys_total`의 섹션 간 mean 희석
(review_v1_8.md §2 패턴이 물리 손실에도 반복됨)을 새로 지적해 수렴했다. 이 라운드에서도 두 모델
모두 코드와 다른 구체적 주장을 냈다 — Gemini는 좌표 디코더 출력이 무제한이라고 주장했으나 실제로는
`torch.clamp(..., -max_displacement, max_displacement)`로 제한되어 있었고, OpenAI는 gradient
clipping이 값 기준(`clip_grad_value_`, 0.5)이라고 주장했으나 실제로는 노름 기준
(`clip_grad_norm_`, 10.0)이었다 — Claude가 두 코드 라인을 직접 대조해 반증하고 최종 문서에서
제외했다. 또한 OpenAI가 인용한 로그 통계·재현 실험은 이 프로젝트에 저장된 로그가 없으므로 1차
세션과 동일한 유형의 환각으로 판정해 배제했다. 채택된 결론(§6~9)은 모두 Claude가 코드 라인 번호
수준으로 재확인한 것만 남겼다. Critic 라운드 없이(두 모델의 핵심 결론이 Claude의 코드 직접
검증으로 이미 교차확인됐으므로) Judge 종합으로 마무리.

</details>
