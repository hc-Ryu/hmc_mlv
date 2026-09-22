# review_v1_6.md — AI_design_v1_6.py 실행 결과 리뷰 (구조적 붕괴 재발 원인 분석)

작성일: 2026-08-09 (`/synod review` 세션, Gemini flash conf95 + OpenAI o3 conf82)
대상: `reports/v1_6/AI_design_v1_6.md`(실행 로그, 620 epoch 완주 + Stage4 620→820, 크래시 없음),
`reports/v1_6/AI_design_v1_6_report.md`, `reports/v1_6/AI_design_v1_6_3d.html`,
`reports/v1_6/AI_design_v1_6_result.png`

---

## 요약

command_v1_6.md(§1 연속성 폐루프 하한, §2 게이트 사전 첨예화)는 크래시 없이 완주했고 Mp 오차도
5% 내외로 양호하다. 그러나 사용자가 보고한 대로 **구조적 붕괴(파트 날카로운 돌출)가 v1.5와
동일하게 재발**했으며, 전 섹션에 걸쳐 나타나고 상위(높은 인덱스) 섹션으로 갈수록 심해진다.

**근본 원인은 §1(연속성 폐루프)의 구현 자체에 있는 회귀 버그다.** 콘솔 로그를 확인한 결과
`w_continuity` 값이 **epoch 0부터 619(Stage4 진입 직전)까지 단 한 번의 예외도 없이 정확히
1.000으로 고정**되어 있다 — 원래 의도(고정 기저 스케줄이 epoch 400 이후 0.05까지 감쇠하고,
그 위에 폐루프가 필요할 때만 개입)가 전혀 작동하지 않고, **연속성 제약이 학습 내내 최대
강도로 상시 적용**되고 있었다. 두 모델(Gemini/OpenAI) 모두 독립적으로 동일한 코드 결함
(`max(w_base, w_adaptive)` 형태의 "한쪽 방향으로만 열리는 래칫")을 지목했고, 이것이 오히려
"섹션 간 강제 유사성 vs 섹션별로 크게 다른 목표 Mp"라는 새로운 충돌을 만들어 날카로운 스파이크로
이어졌다는 물리적 메커니즘까지 일치했다.

---

## 발견된 문제

### [ERROR] 1. `get_adaptive_continuity_weight()`의 `max()` 결합이 편도(one-way) 래칫을 만든다

- **근거(코드, `AI_design_v1_6.py` L868-884)**:
  ```python
  def get_adaptive_continuity_weight(epoch, stage2_end, continuity_state,
                                      w_max=1.0, w_min=0.05, beta=0.1,
                                      w_floor=CONTINUITY_W_FLOOR, tau=CONTINUITY_TAU,
                                      eta=CONTINUITY_ETA):
      w_base = continuity_weight_schedule(epoch, stage2_end, w_max, w_min, beta)
      if epoch > 0:
          error = continuity_state["last_l_continuity"] - tau
          continuity_state["w_adaptive"] = max(
              w_floor, min(w_max, continuity_state["w_adaptive"] + eta * error))
      else:
          continuity_state["w_adaptive"] = w_max
      return max(w_base, continuity_state["w_adaptive"])
  ```
  `tau=0.1`, `eta=2.0`일 때, `l_continuity`가 0.1을 조금이라도 넘는 epoch이 단 한 번만 있어도
  `w_adaptive`는 `min(w_max, ...)`에 의해 즉시(1~2 epoch 내) 상한 1.0으로 포화된다. 이후
  `l_continuity`가 tau 밑으로 **지속적으로** 떨어지는 구간이 나타나야만 `w_adaptive`가 다시
  내려가는데, 실측 로그상 `l_continuity`는 epoch 200 이후 대부분 0.03~0.45 범위에 머물러
  tau=0.1을 자주 초과한다 — "내려갈 기회"가 사실상 열리지 않는다. 게다가 최종 반환값이
  `max(w_base, w_adaptive)`이므로, `w_adaptive`가 한 번이라도 천장에 도달하면 **원래 기저
  스케줄(`w_base`)의 감쇠가 무엇이든 상관없이 결과는 계속 1.0**이 된다 — "하한(floor)"으로
  설계했던 메커니즘이 실제로는 "상한 override"로 뒤집혀 작동한 것이다.
- **로그 실측 검증**: epoch 420(예시)에서 `w_base`를 직접 계산하면(`sigmoid`, beta=0.1,
  center=400) 약 0.16까지 이미 감쇠했어야 하는데, 로그의 `w_continuity`는 여전히 1.000이다.
  이는 `w_base`의 감쇠와 무관하게 `w_adaptive`가 천장에 고정돼 있다는 확실한 증거다(단순히
  `w_base` 자체가 안 내려간 것이 아님).
- **부차 요인(OpenAI 지적, 코드로 확인됨)**: `last_l_continuity`는 EMA(β=0.9)로 평활되는데,
  epoch 360~620 구간에서 19차례 발생하는 동적 분할(Dynamic Split) 이벤트마다 pooling 그룹이
  바뀌며 `l_continuity`가 일시적으로 튄다. β=0.9의 EMA는 이 스파이크의 영향을 약 10~20
  epoch에 걸쳐 서서히만 지워, tau 밑으로 내려가는 "회복 구간"이 열릴 틈을 추가로 줄인다.
- **기각된 가설(OpenAI 제안, 코드 확인 결과 사실 아님)**: "`continuity_state` 딕셔너리가
  분할 이벤트마다 재생성돼 `w_adaptive`가 매번 `w_max`로 리셋되는 것 아니냐"는 가설은 실제
  코드(`AI_design_v1_6.py` L1551, `continuity_state = {...}`가 epoch 루프 **진입 전 단 한
  번만** 실행됨을 확인)로 반증됐다 — 재생성은 일어나지 않는다.

### [ERROR] 2. 상시 최대 연속성 강도가 "강제 유사성 vs 발산하는 섹션별 목표"를 충돌시켜 스파이크를 만든다

- **메커니즘(두 모델 공통 도출)**: 17개 섹션의 목표 Mp는 섹션마다 최대 약 1.8배
  (26.4M~47.5M N·mm) 차이가 난다. `w_continuity=1.0`이 학습 후반까지 상시 적용되면, 인접
  섹션 간 좌표가 서로 크게 달라지는 것 자체에 큰 페널티가 붙는다. 그런데 국소 Mp 목표를
  맞추려면 단면 2차모멘트(관성모멘트, `I ∝ ∫y²dA`)를 늘려야 하고, 이는 보통 좌표를 신경축에서
  멀리 이동시켜야 달성된다 — **"넓게 퍼지는 변화"는 연속성 페널티에 걸리지만, 아주 가늘고
  뾰족한 스파이크는 면적(연속성이 보는 "벌크" 좌표 분포)에는 거의 영향을 주지 않으면서 관성
  모멘트만 크게 늘릴 수 있는 "루프홀(loophole)"**이 된다는 것이 두 모델이 독립적으로 도달한
  설명이다. 즉 v1.5(연속성 억제력 부족 → 무방비 붕괴)와 v1.6(연속성 억제력 과잉 고정 → 우회
  경로로서의 스파이크)은 **서로 반대 극단의 원인이 같은 증상(날카로운 돌출)으로 나타난
  것**이다.
- **"위쪽(상위 인덱스) 섹션일수록 심각" 패턴에 대한 해석**: `l_continuity`는 인접 섹션 쌍만
  연결하는 항이라 계산 자체에 방향성 누적 버그는 없음을 코드로 확인했다(OpenAI 지적,
  Claude 검증 동의 — 전역 pair-count로 정규화되는 구조). 대신 두 모델은 이 패턴을 (a) 목표
  Mp 테이블 자체의 섹션 간 변화 폭(11~13번 섹션 부근에서 최대)이나 (b) §2 게이트 첨예화와
  §4 엔트로피 스케줄(둘 다 epoch 400 부근에서 활성)이 섹션 묶음 전반에 걸쳐 상호작용하며
  만드는 2차 증폭 효과와 연관 지었다 — 다만 이 부분은 **1차 원인(연속성 래칫)만큼 확실하게
  검증되지 않았다**(OpenAI conf82, "secondary amplifier" 수준으로 평가, 아래 한계 참고).

### [WARNING] 3. §2(게이트 사전 첨예화)는 v1.5와 다른 패턴을 보이나 근본 원인은 아님

- candidate 게이트 평균(g3/g4)이 v1.5에서는 epoch 80 이후 0.49~0.52에 정체했던 반면, v1.6
  에서는 "epoch 90 무렵 0.36까지 하강 → epoch 300 무렵 0.264까지 추가 하강 → epoch 610까지
  0.72로 재상승"하는 하강-반등 패턴을 보인다. §2의 `beta_steep`(epoch 200~350에 1.0→3.0
  증폭)이 이 패턴에 관여하는 것은 명확하나, OpenAI가 지적한 대로 **v1.5(연속성이 0.03까지
  약했던 상태)에서도 유사한 형태의 섹션별 붕괴가 이미 있었다는 점이 "게이트가 주원인"이라는
  가설을 약화시킨다** — 연속성 래칫(문제 1) 쪽이 훨씬 강한 설명력을 가진다.

---

## 권장 사항 (우선순위순)

1. **[최우선] `get_adaptive_continuity_weight()`의 `max()` 결합 방식 교체** — 현재의
   "override형" 결합 대신, `w_adaptive`가 `w_base`의 의도된 감쇠를 완전히 무력화하지 못하도록
   재설계한다. 예: 폐루프를 절대값이 아닌 **기저 스케줄에 대한 배율/보정치**로 바꾸거나
   (`return w_base * min(w_adaptive_ratio, ...)`형), 혹은 `w_adaptive`의 상한 자체를
   `w_base`가 아니라 `CONTINUITY_W_FLOOR`(0.3) 근방으로 제한해 "필요할 때만 하한을 보장"하는
   원래 취지로 되돌린다.
2. **[최우선] eta(폐루프 이득) 하향 및 스파이크 필터링** — 현재 eta=2.0은 tau를 살짝만 넘어도
   즉시 천장까지 밀어붙인다. eta를 0.1~0.3 수준으로 대폭 낮추거나, 분할 이벤트 발생 직후
   N epoch(예: 20) 동안은 `l_continuity` 판독을 폐루프 입력에서 제외(또는 별도 완만한 필터
   적용)해 위상 변화로 인한 일시적 스파이크가 제어 루프를 오염시키지 않게 한다.
3. **[권장] `w_continuity`뿐 아니라 `w_adaptive`도 로그에 별도 출력** — 현재 로그는 최종
   결합값만 보여줘 `max()` 문제를 진단하는 데 시간이 걸렸다. `w_base`와 `w_adaptive`를 각각
   출력하면 향후 유사 회귀를 즉시 식별할 수 있다(OpenAI 제안).
4. **[차순위] §2/§4 상호작용 및 섹션별 심각도 패턴 재조사** — 문제 1·2를 먼저 고치고
   재실행한 뒤에도 "상위 섹션일수록 심각" 패턴이 남는지 확인 필요. 남아있다면 별도 세션에서
   게이트 첨예화-엔트로피 상호작용을 §2 범위로 좁혀 재검토.

---

## 한계

- 이번 리뷰는 콘솔 로그와 코드 정적 분석에 강하게 의존했다 — `w_base`와 `w_adaptive`를
  분리 로깅하지 않은 상태라 "언제 정확히 천장에 도달했는지"는 간접 추론(epoch 420 시점
  `w_base` 이론값 vs 로그값 비교)에 의존한다. 권장사항 3(분리 로깅) 적용 후 재실행하면
  이 부분을 직접 확인 가능하다.
- 섹션 인덱스에 따른 심각도 증가 패턴의 정량적 원인(목표 Mp 프로파일 vs 게이트-엔트로피
  상호작용)은 이번 세션에서 결정적으로 규명되지 않았다 — 문제 1·2 수정 후 재실행 결과를
  보고 재조사가 필요하다.

---

<details>
<summary>Synod 세션 상세 (review 모드, Solver 1라운드, 코드 교차검증으로 조기 수렴)</summary>

**Round 1 (Solver)**: Gemini(Architect, conf95)가 `max(w_base, w_adaptive)` 래칫 메커니즘을
수식으로 정확히 재현하고("epoch 1~2 만에 천장 포화"), "가늘고 뾰족한 스파이크가 관성모멘트
루프홀"이라는 물리적 메커니즘을 제시했다. OpenAI(Explorer, conf82)는 동일한 래칫 메커니즘을
독립적으로 도출했을 뿐 아니라, EMA(β=0.9)가 19회의 동적 분할 스파이크를 오래 끌고 가는
부차 요인, 그리고 "`continuity_state` 재생성 버그"라는 대안 가설을 제시했다.

**Claude 코드 교차검증**: `AI_design_v1_6.py`를 직접 읽어 (a) epoch 420 시점 `w_base` 이론값
(~0.16)과 로그값(1.000)의 불일치로 "천장 포화"를 확정, (b) `continuity_state` 초기화가
루프 진입 전 1회뿐임을 확인해 OpenAI의 재생성 가설을 반증. 두 모델의 핵심 진단(래칫 버그)은
코드로 직접 재현·검증됐으므로 별도 Critic/Defense 라운드 없이 Judge 종합으로 진행했다.

**신뢰 점수**: Gemini 95(T2.0, 수식 재현 완전 일치), OpenAI 82(T1.6, 핵심 진단은 유효하나
재생성 가설 기각으로 일부 감점). **최종 신뢰도 90%**.

</details>
