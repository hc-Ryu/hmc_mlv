# review_v1_19_1.md — 파트 존재 게이트(Existence Gate) 학습 불안정성 진단 및 개선안

(Synod v1.0 다중 모델 숙의: Claude Validator + Gemini 3 flash/thinking-high(Architect) +
OpenAI o3/reasoning-high(Explorer), 2026-09-21. Round 1에서 두 외부 모델 모두 confidence>=90,
can_exit=true로 조기 합의에 도달해 Critic/Defense 라운드 없이 종료.)

## §0 요약 (TL;DR)

`review_v1_19.md` §5.2가 미해결로 남겨둔 "Patch1/Patch2의 생존 패턴이 실행마다 크게 달라진다
(같은 코드에서 Patch2가 0/17, 1/17, 17/17로 갈렸다)" 현상의 원인을 확정하고 개선안을 도출했다.

**결론**: 하이퍼파라미터 자체의 버그라기보다, ① 결정 경계(z=0.5)에서의 중립 초기화, ②
epoch 400까지 지연되는 엔트로피(이진화 압력) 부재, ③ 그 공백 구간(epoch 200-350)에서 이뤄지는
게이트 가파름 증폭(steepening), ④ Stage4의 되돌릴 수 없는 강제 이진화라는 **설계 조합**이
경계 민감성을 구조적으로 허용하고, ⑤ `torch.manual_seed()`가 전혀 호출되지 않는다는 실행상
허점이 그 민감성을 실행마다 다른 결과로 노출시키는 방아쇠 역할을 한다.

**우선순위 조치**:
1. (필수, 즉시) `torch.manual_seed()` 등 전역 결정성 확보 — 문제를 "측정 가능"하게 만드는 전제조건
2. (핵심) 초기화를 z=0.5(경계)에서 z≈0.55-0.65 정도로 완만하게 "생존" 쪽으로 편향
3. (핵심) steepening(가파름 증폭)과 entropy 스케줄을 동조화 — 노이즈가 증폭되고 나서야 규제가
   걸리는 현재의 시점 불일치 제거
4. (안전장치) 게이트 파라미터 조기 freeze, log_alpha 업데이트 gradient clipping, Stage4 이후
   구제(rescue) 패스
5. (검증 도구, 기본값 아님) 다중 시드 앙상블 다수결 — 재훈련 비용이 크므로 1-4 적용 후에도
   불확실하면 검증 용도로만 사용

---

## §1 문제 재확인

- 대상: `AI_design_v1_19_0_ref.py`의 section-aware HardConcrete 존재 게이트
  (`log_alpha_candidates`, shape `(17, len(CANDIDATE_PARTS))`), Patch1/Patch2 파트의 생존 여부를
  결정.
- 증상: **동일 코드·동일 설정**으로 여러 번 실행해도 Patch2가 0/17(전부 삭제), 1/17, 17/17(전부
  생존)로 실행마다 크게 갈린다(`review_v1_19.md` §5.2, 미해결로 기록됨).
- 구체 사례(`reports/v1_19_0_ref/`): 해당 실행에서 Patch2가 0/17로 전부 삭제됐고, 동시에 가장
  하중이 큰 sec0/sec1의 Mp 달성률이 각각 37.9%/64.9%로 크게 미달했다. 전체 질량 감량은 이미
  목표(95%) 대비 139.6% 초과 달성 상태라 질량 예산이 삭제를 강제한 것은 아니었다.

## §2 근본 원인 (코드 대조로 확정, 4개 메커니즘의 조합)

| # | 메커니즘 | 코드 근거 |
|---|---|---|
| 1 | 결정 경계에서의 중립 초기화 | `INIT_LOG_ALPHA_V14 = 0.0` → `z_init≈0.5` (line 819). v1.4 §4 주석: "중립 초기화... z_init≈0.5로 시작" |
| 2 | 엔트로피(이진화 압력) 지연 | `alpha_ent_schedule(ep)`가 `ep < ASWC_STAGE2_END(400)`이면 0 반환 (line 3805-3811). 그 전까지는 순수 task gradient만 게이트를 움직임 |
| 3 | 공백 구간 중 가파름 증폭 | epoch 200-350 사이 `log_alpha_candidates`에 1.0→`GATE_STEEP_MAX=3.0` 배율 적용(line 862-864, 1980-1987) — entropy가 켜지기(400) *전에* 이미 노이즈로 정해진 부호를 증폭 |
| 4 | Stage4 비가역적 강제 이진화 | 학습 종료 후 미확정 게이트를 `±5.0`으로 강제 이진화(line 4113-4118), "DELETED는 부활시키지 않는다" |
| 5 (트리거) | torch RNG 미고정 | `--seed`는 `generate_gp_target_mp(seed=args.seed)`(목표 Mp 생성)에만 사용(line 456, 2815, 4247). `torch.manual_seed()` 호출이 스크립트 전체에 없음 |

Patch2는 3노드/floor, 두께 1.6mm, fy=440MPa로 다른 파트(Outer/Inner 1470MPa, Plate 980MPa) 대비
Mp 기여가 작다 — 즉 task gradient 신호가 0.5 근방에서 상대적으로 평탄(flat)하다. 이 상태에서
①~④가 겹치면, "어느 쪽으로 수렴하는가"가 구조적 타당성이 아니라 ⑤ 시드 미고정으로 인한
**실행별 무작위 노이즈**에 의해 사실상 결정된다. review_v1_19.md가 관찰한 0/17 vs 1/17 vs 17/17
편차와 정확히 부합한다.

## §3 숙의 과정 및 모델 기여

<details>
<summary>Synod 세션 상세 (Round 1 — 조기 합의로 종료)</summary>

### 모델 기여
- **Gemini (Architect, confidence 95, can_exit=true)**: z_init≈0.8-0.9로의 낙관적 초기화를
  1순위로 제안. steepening(200-350)과 entropy 스케줄을 250-400 구간으로 동조화하는 재설계안 제시.
- **OpenAI (Explorer, confidence 92, can_exit=true)**: 시드 고정만으로는 평탄 그래디언트 문제를
  해결하지 못한다는 점을 명시적으로 경고. z_init≈0.62 정도의 완만한 편향 + 조기 미세 entropy
  (0.005 → 0.05 램프) + 게이트 파라미터 조기 freeze + log_alpha gradient clipping + Stage4 이후
  구제(rescue) 패스를 제안. 다중 시드 앙상블은 "기본 파이프라인이 아니라 검증 도구"로 한정할 것을
  권고.
- **Claude (Validator)**: 두 모델의 공통 합의(시드 고정 필요·불충분, steepening/entropy 동조화)를
  확인하고, 초기화 편향 강도에 대한 이견(z≈0.8-0.9 vs z≈0.55-0.65)에서는 OpenAI의 완만한 접근을
  채택 권고 — 기존 코드가 "질량 역설"(v1.3 §2, W_SPARSE_V13=0.0 무력화 이력)을 이미 한 번 겪은
  바 있어, 과도하게 낙관적인 초기화로 여러 약한 파트가 한꺼번에 "일단 생존" 상태로 시작해 mass
  loss가 뒤늦게 쳐내야 하는 비대칭 구조를 재현할 위험이 있다고 판단.

### 신뢰 점수
- Claude: 정성적 검증 역할(Trust Score 계산 대상 아님, aggregator)
- Gemini: confidence 95 (evidence/logic 직접 인용, can_exit=true)
- OpenAI: confidence 92 (evidence/logic 직접 인용, can_exit=true, 반대 시나리오까지 명시)

### 해결된 주요 쟁점
1. "시드 고정으로 충분한가?" → 두 모델 모두 **불충분, 필요조건일 뿐**이라는 데 일치. 평탄
   그래디언트/경계 초기화 문제는 별도로 고쳐야 함.
2. "초기화를 얼마나 낙관적으로 할 것인가?" → Gemini(z≈0.8-0.9, 공격적) vs OpenAI(z≈0.62, 보수적
   + 안전장치 병행). Claude 중재로 **OpenAI 안 채택**(§4.2) — 기존 mass-loss 역설 이력을 고려한
   보수적 선택.
3. "다중 시드 앙상블을 기본으로 할 것인가?" → 두 모델 모두 **아니오** — 재훈련 비용 대비 1~4번
   조치가 우선이고, 앙상블은 최후 검증 수단으로만 사용.

</details>

## §4 개선 설계 (우선순위별)

### 4.1 [필수, 즉시] 전역 결정성 확보

학습 진입점에 아래를 추가한다. 이 자체가 근본 원인을 고치지는 않지만, 이후 조치들의 효과를
"같은 노이즈에서 비교"할 수 있게 만드는 전제조건이다.

```python
def set_global_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

`main()`에서 `args.seed`로 호출. 단, `torch.backends.cudnn.deterministic=True`는 GPU 처리량을
~10% 낮출 수 있으므로 `--deterministic` CLI 플래그로 선택 가능하게 둔다.

### 4.2 [핵심] 완만한 낙관적 초기화

`INIT_LOG_ALPHA_V14 = 0.0` (z≈0.5, 정확히 결정 경계)을 **z≈0.55-0.65 구간**(`log_alpha≈0.2-0.6`)
으로 소폭 상향한다. Gemini가 제안한 z≈0.8-0.9는 기각 — 근거는 §3의 쟁점 2 참고(약한 backbone
gradient로는 여러 후보 파트가 한꺼번에 false-positive로 고착될 위험, 그리고 기존 §3
"질량 역설" 이력과의 상호작용 우려).

선택 사항(비용 대비 효과가 낮아 후순위): 파트별 `(area × yield)` 비례 스케일로 초기 편향을
차등화하는 휴리스틱. Patch2처럼 원래 기여가 작은 파트는 스케일이 낮아져 과도하게 낙관적으로
시작하지 않는다.

### 4.3 [핵심] steepening ↔ entropy 스케줄 동조화

현재: entropy는 epoch 400부터, steepening(가파름 증폭)은 epoch 200-350 — **entropy가 켜지기 전에
이미 노이즈로 정해진 부호가 증폭되는 시점 불일치**가 핵심 결함이다.

개선안:
- epoch 0-250: 가파름 1.0 유지(태스크 그래디언트만으로 기본 학습), entropy는 아주 작은 값(예
  0.005)으로 조기 도입해 정확히 0.5에 머무는 것만 약하게 억제(과도하지 않게 — mass loss가 여전히
  음의 방향으로 파트를 쳐낼 수 있도록 상한 0.05 유지).
- epoch 250-400: `alpha_ent`를 0.005→0.05로, 가파름을 1.0→3.0으로 **동시에** 선형 증가시켜 정렬.

### 4.4 [안전장치] 노이즈가 그대로 누적되지 않도록

- **게이트 조기 freeze**: epoch 0~25-50 구간은 `log_alpha_candidates`에 그래디언트를 적용하지
  않고 backbone(응력 예측기)만 먼저 학습시켜, 게이트가 아직 의미 없는 backbone 출력에 반응해
  표류하는 것을 막는다.
- **log_alpha gradient clipping**: 스텝당 `Δlog_alpha` 상한(예 0.1)을 둬 단일 미니배치 노이즈가
  게이트를 단번에 반대편으로 넘기지 못하게 한다.
- **Stage4 이후 구제(rescue) 패스**: Stage4 강제 이진화 후 섹션별 Mp 미달률을 계산해, 10%를
  초과하는 섹션이 있으면 해당 섹션의 관련 게이트만 일시적으로(예 20 epoch, entropy 0.1로 높여)
  재개방한다. sec0/sec1처럼 큰 미달이 남는 "전부 삭제" 파국을 사후에 완화하는 안전판이다.

### 4.5 [검증 도구, 기본값 아님] 다중 시드 앙상블

위 1~4를 적용한 뒤에도 특정 (section, part) 판정이 불확실하면, K=5 정도의 시드로 병렬 학습해
`z>0.5`가 과반(≥3/5)인 경우만 최종 생존으로 채택하는 다수결 방식을 **검증 전용**으로 사용한다.
학습 비용이 ×5이므로 기본 파이프라인에 넣지 않는다.

## §5 기대 효과 및 남은 리스크

- 4.1(결정성)만으로는 "재현 가능한 우연"이 될 뿐, 0/17 vs 17/17 중 어느 쪽이 나올지는 여전히
  임의적이다 — 반드시 4.2/4.3과 함께 적용해야 한다.
- 4.2의 편향이 너무 강하면 mass loss가 이를 상쇄하지 못해 다시 "질량 역설"이 재발할 수 있으므로,
  적용 후 §3(review_v1_19.md)의 면적 감량 달성률(현재 139.6%)이 과도하게 벗어나지 않는지 반드시
  재검증한다.
- 4.4의 freeze/clipping은 backbone 수렴 속도를 늦출 수 있어 전체 epoch 예산(`STAGE2_END=400` 등)
  재조정이 필요할 수 있다.
- cuDNN/CUDA 버전 차이로 `deterministic=True`여도 완전한 비트 단위 재현은 보장되지 않을 수 있음
  — 메이저 버전 업그레이드 시 게이트 격자 결과를 회귀 테스트할 것.

## §6 구현 체크리스트

- [ ] `set_global_seed()` 추가 및 `main()`에서 `args.seed`로 호출 (`--deterministic` 플래그로
      cudnn 결정성 on/off)
- [ ] `INIT_LOG_ALPHA_V14`를 0.0 → 0.2~0.6 구간 값으로 변경(버전 상수명 `INIT_LOG_ALPHA_V15` 등
      신설 권장, 기존 리포트와의 비교 가능성을 위해 이전 상수는 주석으로 보존)
- [ ] `alpha_ent_schedule()`을 "epoch 0-250: 0.005 고정 → epoch 250-400: 0.005→0.05 선형"으로
      재정의
- [ ] `GATE_STEEP_START/END`를 200-350 → 250-400으로 이동해 entropy 스케줄과 동조화
- [ ] 게이트 조기 freeze(epoch<25-50) 로직 추가
- [ ] log_alpha optimizer step에 gradient clipping 추가
- [ ] Stage4 이후 섹션별 Mp 미달률 기반 구제(rescue) 패스 구현
- [ ] 위 변경 적용 후 최소 3회 재실행해 Patch1/Patch2 생존 패턴의 실행 간 분산이 줄었는지 확인
      (기존: 0/17, 1/17, 17/17 관측)

## §7 미해결/별도 판단 필요 (review_v1_19.md §5.2 승계)

- Outer/Inner가 두께 상한(2.5mm)에 몰리고 Plate가 하한 쪽에 몰리는 두께 분기 경향은 본 문서의
  범위 밖이며, 본 개선안 적용 후에도 sec0/sec1의 Mp 미달이 이 두께/깊이 포화 때문에 완전히
  해소되지 않을 수 있다 — 별도 구조 검토 대상으로 유지.
