# idea_v1_8.md — Order 손실 희석 버그(불변 파일) 수정 + LSE 온도 재보정 설계 아이디어

작성일: 2026-08-09 (`/synod idea` 세션, Gemini flash conf95→90 + OpenAI gpt4o conf75→85)
근거: `docs/review/review_v1_7.md`(v1.7 리뷰 — smoothness LSE 온도 미보정 진단),
`reports/v1_7/`(실행 결과), 그리고 이번 세션에서 처음으로 `uni_section_v18.py`(구 불변 파일)
내부의 `compute_mesh_order_loss()` 정의를 직접 읽어 발견한 신규 사실.

**범위 변경**: 사용자가 이번 세션에서 처음으로 `uni_section_v18.py` 수정을 허용했다
(`v19`로 갱신 가능). v1.4~v1.7 네 버전 내내 "손댈 수 없는 파일"이라 아무도 열어보지
않았던 이 파일 안에서, 4개 버전 내내 원인 불명으로 계속 증가하던 "Order 손실"의 **실제
정의**를 이번에 처음 확인했다 — 그리고 결정적 결함을 발견했다.

---

## §0. 신규 발견 — Order 손실의 정체와 숨겨진 희석 버그

```python
# uni_section_v18.py (불변, v1.4~v1.7 내내 아무도 열어보지 않음)
def compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr, eps=0.5):
    """2D Mesh Order Loss."""
    src, dst = edge_index
    mask = (src < dst) & (edge_attr[:, 3] == 0.0)   # 물리 엣지만
    e0 = base_coords[dst[mask]] - base_coords[src[mask]]      # 원본(미변형) 엣지 벡터
    e0_hat = e0 / (e0.norm(dim=1, keepdim=True) + 1e-8)
    e_new = new_coords[dst[mask]] - new_coords[src[mask]]     # 현재 엣지 벡터
    proj = (e_new * e0_hat).sum(dim=1)     # 새 엣지가 원본 방향으로 얼마나 투영되는가
    violation = torch.relu(eps - proj)     # 방향이 뒤집히거나 접히면 위반
    return torch.mean(violation ** 2)      # ← 전체 ~2992개 물리 엣지에 걸친 평균
```
가중치는 `weights['w_order'] = 1.0`(고정 상수, 스케줄 없음).

이 손실은 **엣지가 원래 방향에서 얼마나 "뒤집혔는지"**를 측정한다 — 사실상 지금까지
리뷰 문서들이 계속 찾아 헤매던 "자기교차/메쉬 접힘"을 가장 직접적으로 감지할 수 있는
신호다. 그런데 `torch.mean()`으로 전체 물리 엣지(~2992개)에 걸쳐 평균을 낸다 — **v1.7에서
`compute_smoothness_loss_angle()`에 대해 이미 발견·수정했던 것과 정확히 동일한 종류의
희석 버그**가, 정작 "Order 손실"이라는 이름으로 4개 버전 내내 계속 관찰되고도 아무도
원인을 못 찾았던 이유는 이 코드가 "불변 파일" 안에 있었기 때문이다. 소수의 심하게 접힌
엣지가 있어도 2992로 나뉘어 손실값에는 거의 반영되지 않는다 — 그런데도 실측 로그에서
Order 손실이 100을 넘게 치솟는다는 것은, **평균으로도 가려지지 않을 만큼 광범위하게
엣지가 접히고 있다**는 뜻이다(review_v1_6.md의 "위쪽 섹션일수록 심각" 관찰과 정합적).

---

## 아이디어 평가

### 1. `compute_mesh_order_loss()` Top-K 평균 풀링 + `w_order` 동반 재조정 (최우선, P0)

**설계** (`uni_section_v19.py`, 신규 편집 허용 파일):

```python
def compute_mesh_order_loss(base_coords, new_coords, edge_index, edge_attr, eps=0.5,
                             top_k_fraction=0.02):
    """[v19] Top-K 평균 풀링 — 소수의 심하게 접힌 엣지가 2992개 평균에 희석되는 문제를
    해결한다. LogSumExp 대신 Top-K를 쓰는 이유: 지수 연산이 전혀 없어 v1.7이 겪은
    온도 미보정발 오버플로우 위험이 구조적으로 없다(design 세션에서 수학적으로 검증)."""
    src, dst = edge_index
    mask = (src < dst) & (edge_attr[:, 3] == 0.0)
    if not mask.any():
        return torch.tensor(0.0, device=new_coords.device)
    e0 = base_coords[dst[mask]] - base_coords[src[mask]]
    e0_hat = e0 / (e0.norm(dim=1, keepdim=True) + 1e-8)
    e_new = new_coords[dst[mask]] - new_coords[src[mask]]
    proj = (e_new * e0_hat).sum(dim=1)
    violation = torch.relu(eps - proj) ** 2
    k = max(1, int(violation.numel() * top_k_fraction))   # 약 60개(전체 2992개의 2%)
    top_k_violations, _ = torch.topk(violation, k=k, largest=True)
    return torch.mean(top_k_violations)
```

- **왜 LogSumExp가 아니라 Top-K인가(design 세션 핵심 결론)**: v1.7의 실패 원인을 재검토한
  결과, `T=0.01`에서 10mm급 편차가 들어오면 `exp(10/0.01)=exp(1000)`으로 **float32
  오버플로우가 사실상 확정적**이다(review_v1_7.md의 "near-hard-max" 진단보다 더 정밀한
  재확인). Top-K 평균은 지수 연산이 전혀 없어 이런 오버플로우 경로 자체가 존재하지
  않는다 — 두 모델 모두 "수치적으로 더 안전하다"는 데 동의했다.
- **★ 반드시 함께 적용해야 하는 것 — `w_order` 재조정(Critic 라운드에서 Gemini가 자기
  수정)**: Top-K는 분모를 2992에서 60으로 줄이므로(약 50배), 동일한 국소 위반이라도
  손실·그래디언트 크기가 **약 50배 커진다.** `w_order=1.0`을 그대로 두면 이 손실이 Mp
  목표 등 다른 손실 항을 압도해 v1.7이 겪었던 것과 동일한 "다른 목표를 희생하는" 부작용이
  재현된다. **`w_order`를 대략 `top_k_fraction`(0.02) 근방으로 하향 조정**해 원래
  균형을 유지해야 한다 — Gemini는 최초 제안(1라운드)에서 이를 "후순위 안전장치"로
  분류했다가, 2라운드(Critic)에서 스스로 "필수 선행 조건"으로 정정했다. **이 두 변경은
  반드시 같은 커밋에서 함께 적용한다** — 풀링만 바꾸고 가중치를 그대로 두면 v1.7과 동일한
  실패가 예상된다.
- **미해결 관찰 항목(Gemini/OpenAI 공통 지적)**: Top-K는 매 스텝 "어떤 60개 엣지가
  최악인가"가 바뀔 수 있어 그래디언트가 불연속적으로 튈 위험(chatter)이 있다 — 심각한
  문제가 되면 soft-top-k(온도가 있는 부드러운 버전)나 momentum 기반 완충을 고려.
- **미해결 관찰 항목**: `eps=0.5`(임계값)이 실제 메쉬의 엣지 길이 스케일에 맞는지
  검증되지 않았다(OpenAI 지적) — 절대값(0.5mm)인지 상대값인지 코드만으로는 명확하지 않다.
  §6 검증 계획에서 실제 엣지 길이 분포와 비교 확인 필요.

### 2. v1.7 smoothness LSE 온도 물리 단위 재보정 (최우선, P0)

**설계**: `AI_design_v1_7.py`(→v1.8)의 `SMOOTH_LSE_TEMPERATURE`를 `0.01`(임의값)에서
**물리적 의미가 있는 스케일**로 교체한다.

```python
SMOOTH_LSE_TEMPERATURE = 1.5   # mm. 판금 제조 공차 스케일에 맞춤(design 세션 근거) —
                                # 0.01의 exp(deviation²/T) 오버플로우 경로를 원천 차단
```

- **근거**: 판금(sheet metal) 구조물의 국소 허용 편차가 대략 1~2mm 수준이라는 물리적
  직관에 맞춰 T를 설정하면, `exp(Δ/T)`가 정상적인 수치 범위(오버플로우 없음)에서
  "부드러운 최댓값"으로 동작한다. T=0.01 대비 150배 커지므로 손실 스케일 자체도
  현실적인 범위로 줄어들 것으로 예상된다.
- **§1과의 관계**: 두 항 모두 "지역적 매끄러움 위반"을 다루지만 서로 다른 메커니즘
  (정적 이웃 Laplacian형 vs 엣지 방향 반전형)이므로, 이번 버전에서는 **통합하지 않고
  독립적으로 재보정만** 한다 — 아래 §3 참고.

### 3. (검토했으나 이번 버전에서는 보류) 손실 항 통합 및 하드 제약 대안

1라운드에서 OpenAI가 두 가지 대안을 제시했다:
- **손실 통합**: `l_order`, `l_smooth`(원본), `l_smooth_lse`(v1.7 신규) 세 개의 기하학적
  "정상성" 손실이 중복/상충할 수 있으니 통합하자는 제안.
- **하드 기하 제약**: 소프트 페널티 대신, 자기교차를 만들 좌표 업데이트 자체를 클램핑/
  투영하는 방식이 4연속 소프트 페널티 튜닝 실패 이력을 고려할 때 더 근본적일 수 있다는
  제안.

**Judge 판정(Critic 라운드에서 Gemini가 명시적으로 반대, Claude 동의)**: 두 아이디어
모두 방향성은 타당하나, **이번 버전에서 §1·§2와 동시에 적용하면 안 된다** —
review_v1_7.md가 이미 "개선이 §1 덕분인지 §2 덕분인지 분리할 수 없었다"고 명시했는데,
여기에 손실 통합이나 하드 제약까지 얹으면 다음 리뷰에서 원인 분리가 아예 불가능해진다.
**변수 분리(isolation of variables) 원칙에 따라 이번 버전은 §1·§2(둘 다 기존 항의
보정이지 신규 항 추가나 구조 변경이 아님)로 범위를 한정**하고, 손실 통합·하드 제약은
§1·§2 적용 후 재실행 결과를 보고 **v2.0 이후 별도 세션**에서 검토한다.

---

## 권장 구현 순서

1. **[P0] §1(Order Top-K + w_order 재조정)과 §2(LSE 온도 재보정)를 동시에 적용** —
   두 항 모두 기존 함수의 "보정"이지 새 메커니즘 추가가 아니므로 이번 버전 범위로 적절.
   §1의 두 변경(풀링 방식 + 가중치)은 반드시 같은 커밋에서 함께 적용한다(따로 적용 시
   v1.7과 동일한 실패 재현 위험).
2. **[재검증 필요] `eps=0.5` 실제 스케일 확인** — §1 적용과 함께, 실행 로그에서
   `proj` 값의 분포를 한 번 점검해 임계값이 합리적인지 확인.
3. **[보류, 후속 버전] 손실 통합 / 하드 제약** — §6 검증 결과를 보고 별도 세션에서 재검토.

**주의**: `uni_section_v18.py → v19.py` 전환 시, `AI_design_v1_7.py`(→v1.8)의 모든
`import uni_section_v18 as usv18` 참조를 `uni_section_v19`로 갱신해야 한다 — 파일명
변경에 따른 임포트 경로 수정은 command 문서에서 명시적으로 다룰 것.

---

## §6 검증 계획 (권고)

1. Top-K(k=60) 적용 후 재실행 로그에서 `l_order`가 기존처럼 무한정 증가하지 않고
   수렴하는지 확인.
2. `w_order` 재조정값(≈0.02) 적용 후 Mp 오차가 v1.7 대비 악화되지 않는지 확인(§1
   미조정 시 v1.7 재발 여부의 대조군으로 삼을 수 있음).
3. LSE 온도(T=1.5mm) 적용 후 전체 손실 스케일이 v1.6 수준(2~10)으로 복귀하는지 확인.
4. 최종 3D 형상에서 잔존 스파이크가 추가로 개선됐는지 육안 확인.
5. `eps=0.5`와 실제 엣지 길이 분포 비교(히스토그램 또는 최소/평균/최대 통계).

---

<details>
<summary>Synod 세션 상세 (idea 모드, Solver→Critic 2라운드, Defense 라운드는 생략)</summary>

**Round 1 (Solver)**: Gemini(Architect, conf95)가 `compute_mesh_order_loss()`의 mean 희석과
v1.7 LSE 온도 문제를 "동일 계열의 스케일 오보정"으로 통합 진단하고, Top-K 풀링(수치적으로
지수 연산이 없어 안전) + 물리 단위 LSE 온도(1.5mm)를 제안했다. `exp(10/0.01)=exp(1000)`
계산으로 v1.7의 실패를 float32 오버플로우로 더 정밀하게 재진단했다. OpenAI(Explorer,
conf75)는 희석 제거가 숨겨져 있던 더 큰 위반 규모를 드러낼 위험, 손실 통합 필요성, 하드
기하 제약이라는 더 근본적인 대안, `eps=0.5` 자체의 검증 필요성을 제기했다.

**Round 2 (Critic)**: Gemini가 자신의 1라운드 제안을 스스로 재검토해 "`w_order`는 후순위
안전장치가 아니라 Top-K와 반드시 동시 적용해야 하는 필수 선행 조건"으로 정정했다(N/K≈50배
스케일 변화를 수식으로 도출). OpenAI의 "손실 통합을 지금 함께 하자"는 제안은 변수 분리
원칙 위반으로 명시적으로 반대했고(review_v1_7.md의 원인 분리 실패 사례를 근거로 제시),
Claude도 동의했다. Top-K의 그래디언트 불연속(chatter) 위험과 `eps` 검증 필요성이 새로운
관찰 항목으로 추가됐다. OpenAI critic(conf85)은 동일 결론에 도달했으나 Gemini만큼 구체적인
수치 근거를 제시하지는 못했다.

**Judge(Claude) 최종 판정**: §1(Top-K + w_order 동시 재조정)과 §2(LSE 온도 물리 단위
재보정)를 P0로 채택, 손실 통합·하드 기하 제약은 변수 분리 원칙에 따라 후속 버전으로 보류.

**신뢰 점수**: Gemini 95(solver)→90(critic, 자기 수정 반영), OpenAI 75(solver)→85(critic,
Gemini의 자기수정과 독립적으로 수렴). **최종 신뢰도 90%**.

</details>
