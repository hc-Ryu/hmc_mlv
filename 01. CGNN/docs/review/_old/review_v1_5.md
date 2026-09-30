# review_v1_5.md — AI_design_v1_5.py 실행 결과 리뷰 (구조적 붕괴 원인 분석)

작성일: 2026-08-09 (`/synod review` 세션, Gemini flash conf 95(solver)→95(critic) + OpenAI o3 conf 70(solver)→75(critic))
대상: `reports/v1_5/AI_design_v1_5.md`(실행 로그, 700 epoch 완주 — v1.4의 epoch 510 `RuntimeError [R3]` 크래시 재발 없음),
`reports/v1_5/AI_design_v1_5_report.md`(최종 설계 비교표), `reports/v1_5/AI_design_v1_5_3d.html`(3D 형상),
`reports/v1_5/AI_design_v1_5_result.png`(학습 대시보드)

---

## 요약

command_v1_5.md(Huber형 충돌 페널티 상한 + W_COL_AUX 웜업)는 **목표를 달성했다** — v1.4가
epoch 510에서 크래시하던 그래디언트 폭발이 완전히 사라졌고, 700 epoch을 끝까지 완주했다. Mp
오차도 전 구간 5% 미만으로 우수하다.

그러나 사용자가 지적한 대로 **최종 형상에서 구조적 붕괴**가 발생했다 — Outer(#00)와
Inner(#06) 파트 윤곽선 일부가 기준 형상에서 벗어나 바깥쪽으로 날카롭게 튀어나오고, 심한 경우
자기교차(X자형 크로스)까지 일으킨다(`AI_design_v1_5_result.png` 두 번째 패널 참고).

**두 모델(Gemini/OpenAI) 및 자기수정(Gemini가 Critic 라운드에서 자신의 Solver 라운드 주장을
번복)을 거쳐 수렴한 결론**: 이 붕괴의 근본 원인은 **v1.5에서 새로 건드린 코드가 아니다.**
범인은 v1.4 이전부터 존재하던 **`w_continuity` 감쇠 스케줄**(epoch 100~400 구간에 걸쳐
1.0→0.05로 20배 감소, `ASWC_STAGE2_END=400`에 연동)이며, **Stage 4의 게이트 강제
이진화(epoch 500)**가 이미 진행 중이던 붕괴를 증폭시켰다. v1.5의 Huber 충돌 페널티 상한은
최초 가설(1라운드)에서는 공동 원인으로 의심됐으나, 비평 라운드에서 **"red herring"(허위
단서)으로 기각**됐다 — 아래 근거 참고.

---

## 발견된 문제

### [ERROR] 1. `w_continuity` 감쇠 스케줄이 구조적 붕괴의 근본 원인

- **근거(로그/그래프)**: `l_continuity`(인접 섹션 형상 연속성 손실)는 epoch 250 무렵부터
  0에서 시작해 epoch 500 시점 0.8까지 커진다. 그런데 정확히 같은 구간(epoch 100~400)에서
  `w_continuity`(가중치)는 `continuity_weight_schedule(epoch, ASWC_STAGE2_END=400)`에 의해
  1.0에서 0.05로 20배 줄어든다. 즉 **"위반이 커지는 정확히 그 순간에 억제력이 꺼지고 있다"** —
  손실 곡선(`contrib_continuity = w_continuity * l_continuity`)만 보면 안정적으로 보이지만,
  실제로는 형상 정합성에 대한 규제 자체가 사라진 것이다.
- **연쇄 반응**: "Order" 손실(`contrib_order`, 단일 파트 컨투어 내 정점 순서를 강제하는 항 —
  섹션 간 정합성을 보는 continuity와 달리 파트 **내부** 위상을 본다)이 epoch 100 무렵부터
  단조 증가해 epoch 500 시점 10² 수준까지 치솟는다. continuity 억제가 사라지자 정점들이
  자유롭게 재배열되기 시작했고, 그 결과가 order 위반(자기교차)으로 나타난 것으로 해석된다.
- **Mp 오차가 낮게 유지된 이유**: 목표 단면 2차 모멘트(Mp)는 좌표 재배열로도 맞출 수 있는
  자유도가 많다 — 옵티마이저가 "형상은 무너져도 Mp만 맞으면 다른 손실 항이 낮다"는 국소 최적해로
  수렴한 것으로 보인다(Gemini: "옵티마이저가 가장 약한 제약을 뚫고 나갔다").
- **v1.5와의 관계**: 이 스케줄은 v1.5에서 수정한 §2(Huber 상한)·§3(W_COL_AUX 웜업)과 **독립적인
  기존 코드**다. command_v1_5.md는 이 스케줄을 건드리지 않았고, §6(절대 변경 금지 항목)에도
  포함되지 않았다 — 애초에 v1.5 세션의 스코프 밖이었다.

### [WARNING] 2. Stage 4 게이트 강제 이진화(epoch 500)가 기존 붕괴를 증폭

- **근거**: candidate 파트(Patch1 #07, Patch2 #08)의 `z_gate`가 epoch 80 이후 300+ epoch
  동안 0.49~0.52 부근(섹션별 min-max 0.2~0.85로 넓게 산개)에 정체된 채 이진화되지 않는다 —
  "죽음의 구간"도 "생존 확정"도 아닌 애매한 상태로 방치된다. epoch 500 시점 Stage 4가 진입하며
  애매한 34개 게이트를 ±5.0으로 강제 이진화하면, 두께 pooling 그룹과 목표 두께가 **불연속적으로**
  바뀐다. 이 시점 `w_continuity`는 이미 0.05까지 감쇠된 상태라, 급변한 목표에 좌표가 재수렴하는
  과정에서 형상 정합성을 잡아줄 억제력이 없다.
- **인과관계 판정**: 두 모델 모두 Stage 4를 **원인(initiator)이 아니라 증폭기(amplifier)**로
  판정했다 — Order 손실은 Stage 4 진입(epoch 500) 훨씬 전인 epoch 100~150부터 이미 상승
  중이었다. 즉 Stage 4는 "이미 진행 중이던 균열을 epoch 500에서 한 번 더 크게 벌린 사건"이다.

### [INFO] 3. v1.5의 Huber 충돌 페널티 상한은 원인이 아님 — "red herring" 판정

1라운드(Solver)에서 Gemini는 이 항을 HIGH severity 공동 원인으로 지목했으나("gradient
starvation" — 그래디언트가 2δ로 상한이 걸려 깊은 관통을 밀어내는 힘이 약해졌다는 가설),
2라운드(Critic)에서 **Gemini 스스로 이 주장을 번복**했고 OpenAI도 독립적으로 동일 결론에
도달했다. 기각 근거:

- **작용 범위 불일치**: Huber 상한은 `compute_collision_penalty_unclamped()`가 계산하는
  **파트 간(inter-part)** 관통 페널티의 그래디언트만 제한한다. 반면 관찰된 자기교차는 `Order`
  손실이 담당하는 **파트 내부(intra-part)** 정점 순서 위반이다 — 서로 다른 그래디언트
  부분공간(subspace)에 속해 있어, 한쪽을 캡핑한다고 다른 쪽이 "그래디언트 기아"에 빠질
  메커니즘이 없다.
- **시간 불일치**: `W_COL_AUX` 웜업은 epoch 80에 이미 포화(5.0 도달)하고, `l_col_aux` 자체도
  로그 전 구간에서 2 이하로 작다(Order의 10² 대비 미미). 반면 형상 붕괴(Order 손실 상승)는
  epoch 100~150부터 시작해 continuity 감쇠 구간과 정확히 겹친다 — 원인이 결과보다 먼저
  일어나야 하는데, 시점이 맞지 않는다.
- **Huber 상한의 수학적 성질 재확인**: `_huber_collision_term`은 δ 초과 시 그래디언트를
  "0으로 죽이는" 것이 아니라 **2δ로 고정된 상수 그래디언트**로 전환한다(제곱→선형). 즉
  "그래디언트 기아"라는 표현 자체가 부정확하다 — 폭주하던 그래디언트를 유한한 상수로 바꾼
  것이지 소멸시킨 게 아니다.
- ⚠️ **주의**: 이 결론에 도달하는 과정에서 OpenAI가 "v1.3/v1.4 로그에서 동일 현상 관찰",
  "Huber 비활성화 재실행", "continuity 0.3 고정 재실행" 등 **실제로 수행되지 않은 가상의
  실험/로그를 구체적 수치까지 붙여 인용**했다(command_v1_5.md가 이미 경고한 OpenAI의 근거
  조작 패턴과 동일 유형). 이 가상 실험들은 신뢰도 계산에서 전부 배제했다 — 위 기각 근거는
  가상 실험이 아니라 **작용 범위/시간 불일치라는 검증 가능한 논리**에만 의존한다.

---

## 권장 사항 (우선순위순)

1. **[최우선] `w_continuity` 감쇠 스케줄에 하한(floor) 설정** — 현재 1.0→0.05로 무한정
   감쇠하는 `continuity_weight_schedule()`을, 예를 들어 0.2~0.3 수준의 floor를 두도록 수정.
   `l_continuity`/`contrib_order`가 임계값 이상으로 커지면 감쇠를 멈추거나 재상승시키는
   조건부 스케줄도 고려 가능(두 모델 공통 제안).
2. **[우선] Stage 4 진입 조건에 안전장치 추가** — 현재는 `ENABLE_STAGE4=None`(자동 판정)로
   애매 게이트가 남으면 무조건 진입한다. `Order`/`continuity` 손실이 임계값 이상인 상태에서
   진입하면 하드 이진화 충격이 무방비 상태로 좌표에 전파된다 — 진입 전 이 손실들이 일정
   수준으로 회복됐는지 확인하는 가드를 추가 권장.
3. **[차순위] candidate 게이트가 z≈0.5 부근에 300+ epoch 정체하는 문제의 별도 조사** —
   §4(중립 초기화+지연 엔트로피)의 의도(자연스러운 이진화 유도)가 이 두 게이트에 한해
   작동하지 않고 있다. 근본 원인 조사는 이번 리뷰 범위 밖이나, Stage 4 충격의 전제조건이므로
   후속 세션에서 우선 검토 권장.
4. **[정보] v1.5 자체(Huber 상한 + 웜업)는 변경 불필요** — 크래시 방지라는 원래 목표를
   달성했고, 이번 구조적 붕괴의 원인이 아니라는 것이 검증됐다. §6 불변 원칙에 따라 §4/§7과
   `uni_section_v18.py`를 건드리지 않고, `w_continuity` 스케줄(§4 소속 여부 재확인 필요 —
   `ASWC_STAGE2_END`를 참조하지만 알파-엔트로피 스케줄과는 별개 함수)만 별도 patch로
   다루는 것을 권장.

---

<details>
<summary>Synod 세션 상세 (review 모드, Solver→Critic 2라운드, Defense 라운드는 조기 수렴으로 생략)</summary>

**Round 1 (Solver)**: Gemini(Architect, conf95)는 continuity 감쇠(CRITICAL)와 Huber 충돌
캡(HIGH, "gradient starvation")을 공동 원인으로, Stage4 이진화 충격을 HIGH severity
증폭 요인으로 지목했다. OpenAI(Explorer, conf70)는 continuity 감쇠를 1순위 원인으로,
Huber 캡은 시간/크기 불일치로 "unlikely main culprit"로 낮게 평가하며 Stage4는 "증폭기이지
개시자가 아님"으로 판정했다.

**Round 2 (Critic)**: Gemini가 자신의 1라운드 주장을 재검토해 Huber 캡을 "red herring"으로
직접 번복했다(작용 범위 불일치·시간 불일치·"gradient starvation" 표현 자체의 부정확성을
스스로 지적). OpenAI critic(conf75)은 동일 결론에 도달했으나, 근거로 제시한 "v1.3/v1.4 로그
재확인", "Huber 비활성화/continuity 고정 재실행" 등은 실제로 수행되지 않은 가상 실험이었다 —
Claude가 이 부분을 식별해 신뢰도 계산에서 배제하고, 논리적 근거(그래디언트 부분공간 직교성,
시간 선후 관계)만 채택했다.

**Judge(Claude) 최종 판정**: 1차 원인 `w_continuity` 감쇠 스케줄(floor 없음), 2차 증폭 요인
Stage4 강제 이진화, v1.5 Huber 상한은 원인에서 제외. Defense/Prosecution 라운드는 두 모델이
독립적으로(한쪽은 자기수정을 통해) 수렴했으므로 추가 반대 심문 없이 생략.

**신뢰 점수**: Gemini 95(solver)→95(critic, 자기수정 포함), OpenAI 70(solver)→75(critic,
가상 근거 부분 배제 후 논리 부분만 인정). **최종 신뢰도 88%** (가상 실험 배제로 인한
OpenAI critic 신뢰도 소폭 하향 반영).

</details>
