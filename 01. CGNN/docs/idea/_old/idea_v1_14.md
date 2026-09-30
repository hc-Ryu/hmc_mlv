# idea_v1_14.md — 파트 간 겹침 해소를 위한 v1.14 구현 아이디어

작성일: 2026-08-15 (`/synod idea` 세션, Gemini flash conf95(Solver)/95(Critic) + OpenAI gpt4o
conf85(Solver)/80(Critic) — Solver 라운드에서 핵심 아이디어(Top-K 방향 풀링)에 수렴, Critic
라운드에서 OpenAI가 제시한 리스크 아이디어(경계 노드 이동 허용)를 두 모델 모두 기각/제한적 변형으로
판정하며 can_exit=true 수렴, Defense 라운드는 생략)

근거 문서: `docs/review/review_v1_13.md`(파트 겹침 근본 원인 리뷰)
대상 파일: `AI_design_v1_13.py`, `uni-section/code/uni_section_v21.py`

---

## 요약

`review_v1_13.md`가 지목한 3개 근본 원인(① `compute_collision_loss_v5()`의 153-direction 단순
평균 희석, ② `fix_x_mask`/`fix_y_mask`로 인한 좌표 회피 경로 차단, ③ `build_collision_spec()`의
1회성 고정 clearance)에 대해, Gemini(Architect)와 OpenAI(Explorer)가 각각 독립적으로 아이디어를
제시했고 **핵심 1순위 아이디어(direction-레벨 Top-K 풀링)에서 즉시 수렴**했다. Critic 라운드에서는
OpenAI가 제시한 "경계 노드 이동 허용" 아이디어를 **두 모델 모두 조립성/제조성 리스크를 이유로
사실상 기각**하고, 위반 쌍에 한정된 극히 제한적인 변형만 인정하는 데 합의했다.

**v1.14 권장 로드맵**: [아이디어 1](Top-K 방향 풀링, 필수) + [아이디어 4](동적 clearance 재계산 또는
epoch 커리큘럼, 보조) 조합을 1차 패치로 채택하고, [아이디어 2](경계 접합부 clearance 마진 보상)는
1차 패치 결과를 본 뒤 필요 시 추가. [아이디어 3](경계 노드 이동 허용)은 **기각** — 필요하다면 후속
세션에서 "조인트 쌍 공동 이동 제약"이라는 훨씬 제한된 형태로만 재검토.

---

## 아이디어 목록 (우선순위순)

### [아이디어 1] (최우선, 두 모델 공통 제안) — Direction-레벨 Top-K 풀링 도입

**무엇을 바꾸나**: `uni-section/code/uni_section_v21.py` L654 `compute_collision_loss_v5()`의
마지막 집계 방식을

```python
if n_dirs > 0:
    total_loss = total_loss / n_dirs   # 현재: 153개 direction 전체 단순 평균
return total_loss
```

에서, 이미 각 direction 내부에 적용 중인 Top-K(30%) 패턴을 direction들 "사이"의 집계에도 동일하게
적용하는 계층적(hierarchical) Top-K로 교체한다:

```python
def compute_collision_loss_v5(..., top_k_dir_fraction=0.2, min_dirs_to_keep=5):
    """
    [v1.14] direction-레벨 집계를 단순 평균에서 Top-K 평균으로 교체 — 153개 direction 중 극소수만
    위반해도 나머지 정상 direction에 의해 그래디언트가 1/153로 희석되던 문제(review_v1_13.md #1)
    해소. compute_mesh_order_loss()(v1.8)/compute_smoothness_lse_v3()(v1.10)와 동일한 패턴.
    """
    dir_losses = []   # 기존 루프에서 direction별 loss_dir을 리스트로 수집(기존 total_loss+= 대신)
    for (sec_int, a, b), directions in collision_spec.items():
        ...
        for d in directions:
            ...
            dir_losses.append(loss_dir)

    if not dir_losses:
        return torch.tensor(0.0, device=new_coords.device)
    dir_losses_t = torch.stack(dir_losses)
    k = max(min_dirs_to_keep, int(dir_losses_t.numel() * top_k_dir_fraction))
    k = min(k, dir_losses_t.numel())
    top_k_losses, _ = torch.topk(dir_losses_t, k=k, largest=True)
    return top_k_losses.mean()
```

**왜 근본 원인을 해결하나**: 153개 중 실제 위반이 발생하는 direction은 극소수(Part0-1/1-2/1-4에
해당하는 일부 floor)뿐이다. 단순 평균 대신 Top-K(예: 상위 20%, 최소 5개 보장)로 바꾸면 위반
direction의 그래디언트가 나머지 정상 direction에 의해 희석되지 않고 물리(Mp) 손실과 대등하게
경쟁할 수 있다.

**리스크/트레이드오프**: Top-K 비율이 너무 작으면 학습 초반 다발적 위반이 무시될 수 있고, 너무
크면 기존과 다를 바 없어진다(Gemini). `min_dirs_to_keep`으로 학습 안정성 하한을 보장해야 한다.
OpenAI critic은 Top-K 비율 설정 오류 시 "중요한 충돌이 무시될 수 있다"는 리스크를 별도로 지적—
초기 K값은 review 권장치(30%, 기존 within-direction과 동일값)에서 시작해 재보정 권장.

**검증 방법**: 학습 로그에서 `collision_loss_v5` 값이 기존 대비 초기 epoch에 유의미하게 크게
측정되는지(희석이 풀렸다는 신호), epoch 진행에 따라 감소하며 최종 `final_section_v1_14.csv`에서
Part0-1/1-2/1-4 겹침이 review에서 쓴 것과 동일한 노드±t/2 밴드 겹침 검사로 사라졌는지 재확인.

---

### [아이디어 2] (보조) — 경계 접합부 clearance 마진 보상 (Gemini 제안)

**무엇을 바꾸나**: `build_collision_spec()`에서, 좌표가 고정된(`fix_x`/`fix_y`=1) 노드가 관여하는
파트쌍 방향에 한해 `clearance_default`(현재 0.5mm 전역 상수)를 국소적으로 상향(예: +0.2mm)한다.

**왜 근본 원인을 해결하나**: review #2 원인(좌표 이동 불가 → 두께 감소만이 유일한 회피 경로)에
대한 직접 대응. 좌표로 도망갈 수 없는 영역에는 애초에 더 넓은 안전 마진을 부여해, 두께가
T_MAX까지 saturate되더라도 물리적 침범에 도달하기 전에 collision loss가 먼저 강하게 반응하도록
한다.

**리스크/트레이드오프**: 해당 접합부 주변 파트 두께가 지나치게 얇아질 수 있음(T_MIN=0.7mm로
보호되지만, 목표 Mp 미달 가능성). 마진 폭(0.2mm)은 임의값이므로 스모크 테스트 후 재보정 필요.

**검증 방법**: 고정 경계 노드 인접부의 최종 두께 분포와 해당 영역 Mp 달성률을 [아이디어 1]만
적용했을 때와 비교.

---

### [아이디어 3] (기각 — 두 모델 Critic 라운드 공통 판정) — 경계 노드 이동 허용

OpenAI가 Solver 라운드에서 제안한 `fix_x_mask`/`fix_y_mask`를 완화해 경계 노드도 이동시켜 겹침을
회피하는 아이디어. **Critic 라운드에서 Gemini(conf95)와 OpenAI 자신(critic conf80) 모두 이 아이디어를
기각 또는 극히 제한적 변형으로만 인정**했다:

- `_FIX_MASK_TABLE`이 정의하는 고정 노드는 단순 기하 제약이 아니라 B-pillar 조립 기준면(용접
  플랜지, 인접 부품과의 조인트)을 나타낼 가능성이 높다. 이를 완화하면 GNN 상으로는 겹침이 없는
  형상이 나오더라도 실제로는 플랜지가 어긋나 조립/용접이 불가능한 "기하학적으로만 유효하고 제조
  불가능한" 결과물이 나올 위험이 크다.
- `CGDN17.forward()`의 `join_pairs` 중점 일치 제약과도 충돌 가능성이 있다.
- 두 모델 모두 "완전 기각"보다는, 굳이 필요하다면 **위반이 확인된 특정 노드 쌍에 한해, 접합되는
  두 부품의 플랜지 노드가 독립적으로가 아니라 동일한 변위 벡터로 함께 이동(rigid-body coupled
  displacement)하도록 제약을 결합한 극히 제한된 형태**로만 후속 검토할 것을 제안했다(Gemini
  Critic: "공동 이동 제약(Cooperative Translation)"). 이번 v1.14 범위에는 포함하지 않는다.

---

### [아이디어 4] (보조, 두 모델 공통 계열 제안) — Clearance 재계산/커리큘럼

Gemini는 **epoch 스케줄 기반 커리큘럼**(warmup 동안 느슨한 clearance → 목표치(0.5mm)로 점진 강화,
코사인 완화 권장), OpenAI는 **매 iteration마다 현재 `t_final`/좌표로 clearance 재계산**을 제안했다
— 둘 다 review #3 원인(1회성 고정 clearance)에 대한 대응이며, 구현 난이도(전자는 스케줄 함수 1개
추가, 후자는 `build_collision_spec()`을 학습 루프 안으로 옮기는 더 큰 리팩터링)에서 차이가 난다.

**권장**: 1차 패치는 구현 난이도가 낮은 Gemini의 epoch 커리큘럼 방식으로 시작하고, 그래도 잔존
겹침이 있으면 OpenAI의 매 iteration 재계산(더 정확하지만 `collision_spec`을 static dict에서 매
step 재계산 가능한 구조로 바꿔야 함 — `AI_design_v1_13.py` L2066 호출부와 `train_step()` 시그니처
변경 필요)으로 확장.

**리스크**: 학습 후반 clearance가 갑자기 강화되면 Mp 물리 손실과 충돌해 수렴이 흔들릴 수 있음 —
코사인/선형 완만한 전이로 완화 필요(Gemini critic 지적).

---

### [아이디어 5] (보조) — Unclamped 보조 충돌 페널티의 Huber화 (Gemini 제안)

`AI_design_v1_9.compute_collision_penalty_unclamped()`(현재 로그상 스파이크 후 0.0000으로
돌아오며 사실상 무력화된 상태)를 Huber 페널티로 재설계해, 침투가 깊을수록 그래디언트가 죽지
않고 오히려 강해지도록 한다. review #3(clamp(max=1.0) 포화 구간, 현재는 0.70mm < 1.0mm라
아직 미해당하지만 잠재 리스크로 기록됨)에 대한 선제 대응.

**권장**: 이번 v1.14의 필수 항목은 아니며, [아이디어 1] 적용 후에도 겹침이 1.0mm에 근접/초과하는
사례가 재발할 경우에만 추가.

---

## v1.14 패치 로드맵 (권장 적용 순서)

1. **[필수]** 아이디어 1 — `compute_collision_loss_v5()` direction-레벨 Top-K 풀링 적용.
2. **[필수, 저비용]** 아이디어 4의 epoch 커리큘럼 변형 — `build_collision_spec()`의 clearance를
   초기값에서 목표치(0.5mm)로 완만히 강화하는 스케줄 함수 추가.
3. **[선택]** 1·2 적용 후 재학습 결과에서 여전히 특정 접합부에 겹침이 남으면 아이디어 2(경계 마진
   보상) 추가.
4. **[후속 세션으로 이월]** 아이디어 5(Huber화)는 겹침이 1.0mm 근접 시에만, 아이디어 3(경계 노드
   이동)은 "조인트 쌍 공동 이동" 형태로 재정의 후 별도 `/synod design` 세션에서 재검토.
5. **[검증 필수]** 패치 후 두께가 여전히 전 파트 T_MAX로 균일 saturate되는지, 물리(Mp) 손실 대비
   collision loss 가중치 배분을 재조정해야 하는지 학습 로그로 재확인(review_v1_13.md 권장사항 4).

---

<details>
<summary>숙의 과정</summary>

### 모델 기여
- **Gemini (Architect, Solver conf95→Critic conf95, can_exit=true 양 라운드):** Top-K 방향 풀링을
  구체 코드 스케치와 함께 최초 제시, 경계 노드 clearance 마진 보상·epoch 커리큘럼·Huber화까지
  4개 보조 아이디어를 체계적으로 제시. Critic 라운드에서 OpenAI의 경계 노드 이동 아이디어를
  조립성/제조성 근거로 강하게 반박.
- **OpenAI (Explorer, Solver conf85→Critic conf80, can_exit=false→true):** Top-K 방향 풀링에
  동일하게 수렴하면서, Gemini가 다루지 않은 "매 iteration clearance 재계산"과 "경계 노드 이동
  허용"이라는 두 개의 독자적 아이디어를 추가로 제시. Critic 라운드에서는 자신의 경계 노드 이동
  아이디어를 스스로도 재검토해 "제한적 이동만 허용" 쪽으로 수정 제안.

### 해결된 주요 쟁점
1. "경계 노드 이동을 허용해도 되는가?" → 원칙적으로 기각(조립성/제조성 리스크), 필요 시 조인트
   쌍 공동 이동이라는 훨씬 제한된 형태로만 후속 검토(양 모델 Critic 라운드 수렴).
2. "clearance 재계산을 커리큘럼으로 할지 매 iteration으로 할지?" → 구현 비용을 고려해 커리큘럼
   방식을 1차로 채택, 부족하면 매 iteration 방식으로 확장.

### 신뢰 점수
- Gemini: Solver 95, Critic 95 (High trust)
- OpenAI: Solver 85, Critic 80 (High trust)
- Claude(조율자): review_v1_13.md 근본 원인과의 정합성 확인, 로드맵 우선순위 조율

</details>
