# uni_section_v6 학습 정체(freeze) 원인 및 수정 사항

`/synod review` 세션(`synod-20260710-092219-eee825`, Gemini flash + OpenAI o3 실제 병렬 교차검증)에서
도출된 `uni_section_v6.py` 학습 정체 원인과 수정 방안을 정리한다.

**증상**: epoch 39 이후 `l_phys/l_smooth/area/l_mass/l_collision`이 소수점까지 완전히 동일하게
140 epoch 동안 "동결"됨. `alpha_min`은 epoch 19만에 0.000 도달. 최종 Mp 오차 75.28% (전혀 미충족).

---

## 1. 근본 원인 (Gemini·OpenAI 공통 진단, 신뢰도 78%)

### 1.1 두께 하한 클램프에서 gradient 완전 소멸
```python
t_new = torch.clamp(t_new, min=0.1, max=3.0)
```
두께가 하한(0.1mm)에 닿으면 `clamp`의 미분이 정확히 0이 되어, 그 지점부터 `thickness_decoder`로의
역전파가 완전히 끊긴다. 로그의 "값 동결" 현상과 정확히 일치.

### 1.2 Ghost Gate 시그모이드 조기 포화
`GHOST_STEEPNESS=20.0`이 지나치게 가팔라 `t_new≈0.1`일 때 시그모이드 입력이 `20×(0.1-0.5)=-8`,
`σ(-8)≈0.0003`으로 즉시 포화된다. 그 지점의 미분도 `≈0.0003`이라 사실상 0. 로그의
`alpha_min=0.000`(epoch 19)과 정확히 일치.

### 1.3 `alpha`의 이중 역할이 구조적 충돌을 만듦 (OpenAI가 지적한 핵심 포인트)
`alpha`가 (a) `t_final = t_new*alpha + t_new*0.05`로 phys/mass loss에 쓰이는 동시에
(b) `.detach()`되어 collision loss masking에도 쓰인다. 이 때문에 모델은
"두께를 키워 Mp를 맞춰라"(phys)와 "두께를 얇게 유지해 collision을 회피하는 것이 더 쉬움" 사이에서,
가장 쉬운 국소 최적해인 "하한 클램프까지 밀어붙이기"로 수렴한다.

### 1.4 커리큘럼이 사실상 무력화됨
`w_phys(20)×l_phys(0.87)≈17.3` 대비 `w_collision(2)×l_collision(0.13)≈0.27`(게다가
`ghost_weight≈0`이라 사실상 0)이라, 커리큘럼 ramp가 끝나도 phys 기여도가 98~100%에서 벗어나지 못함.

**쟁점**: Gemini는 "collision loss의 alpha detach만 풀어서 완화"를 제안했고, OpenAI는
"alpha를 phys/mass 두께 계산에서 완전히 분리하고 collision masking 전용으로만 쓰자"는 더 근본적인
구조 변경을 제안. Judge 판정: **OpenAI 안 채택** — Gemini 안은 여전히 두께가 phys/mass에 그대로
쓰이는 구조라 재발 위험이 남지만, OpenAI 안은 "두께 증가 요구"와 "collision 회피 요구"의 충돌
자체를 근본적으로 제거함.

---

## 2. 핵심 수정 사항 (우선순위 순)

### 2.1 [최우선] `alpha`를 collision masking 전용으로 분리
```python
# 변경 전 (CGDN.forward 반환 직전):
alpha   = torch.sigmoid(self.GHOST_STEEPNESS * (t_new - self.GHOST_THRESHOLD))
t_final = t_new * alpha + t_new * 0.05      # ← phys/mass가 alpha에 종속됨

# 변경 후:
alpha   = torch.sigmoid(self.GHOST_STEEPNESS * (t_new - self.GHOST_THRESHOLD))
t_final = t_new                              # phys/mass는 alpha 영향 없이 t_new 그대로 사용
# alpha는 오직 compute_collision_loss_v3_with_alpha의 ghost_weight 계산에만 사용
```
이렇게 하면 phys loss가 "두께를 키워라"는 신호를 보낼 때, collision loss가 "두께를 줄여서 회피"하는
경쟁 신호를 만들지 않는다. 두 loss가 더 이상 같은 변수(`t_final`)를 놓고 싸우지 않는다.

### 2.2 Ghost Gate 시그모이드 완화
```python
class CGDN(nn.Module):
    GHOST_THRESHOLD = 0.5
    GHOST_STEEPNESS = 5.0   # 20.0 → 5.0: 포화 구간을 넓혀 gradient 생존 구간 확대
    DELTA_SCALE     = 1.5
```

### 2.3 하한 클램프를 soft clamp(시그모이드 매핑)로 교체
```python
# 변경 전:
# t_new = t_initial + delta_t_part
# t_new = torch.clamp(t_new, min=0.1, max=3.0)

# 변경 후 (Gemini 제안):
t_min, t_max = 0.1, 3.0
t_initial_logit = torch.logit((t_initial - t_min) / (t_max - t_min))
t_new = t_min + (t_max - t_min) * torch.sigmoid(t_initial_logit + delta_t_part)
```
하한/상한 근처에서도 미분이 완전히 0이 되지 않아, 한번 갇히면 못 나오는 상태를 방지.

### 2.4 손실 가중치 재조정
`w_phys`를 20→10 수준으로 낮추거나 `w_collision`을 높여, ramp가 끝났을 때 collision/mass가
total loss에 실제로 의미 있게(수 %가 아니라 두 자릿수 %) 기여하도록 조정.

```python
weights = {
    'w_phys':      10.0,   # 20.0 → 10.0
    'w_smooth':     0.5,
    'w_mass':       2.0,
    'w_collision':  5.0,   # 2.0 → 5.0: collision이 무력화되지 않도록 강화
}
```

### 2.5 (필요시) 회복 여지 확대
2.1~2.4 적용 후에도 특정 파트가 다시 클램프 경계에 붙는다면 `DELTA_SCALE`을 늘리거나(1.5→3.0)
학습률을 조정해 회복 여지를 넓힌다.

---

## 3. 적용 순서 요약 (uni_section_v7 방향)

1. §2.1 (alpha/phys 분리) — 가장 근본적인 구조 충돌 제거, 최우선 적용.
2. §2.2 (GHOST_STEEPNESS 완화) + §2.3 (soft clamp) — gradient 생존 경로 확보.
3. §2.4 (가중치 재조정) — 커리큘럼이 실제로 효과를 내도록.
4. 재학습 후 로그에서 `alpha_min`이 더 이상 epoch 20 이내에 0.000으로 떨어지지 않는지,
   `l_phys`가 epoch가 지나며 실제로 감소하는지(동결되지 않는지) 확인.
5. 그래도 동결이 재발하면 §2.5(DELTA_SCALE 확대)와 학습률 조정을 추가 적용.
