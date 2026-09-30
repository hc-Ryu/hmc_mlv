# review_v1_13.md — AI_design_v1_13.py 결과 파트 간 겹침(overlap) 근본 원인 리뷰

작성일: 2026-08-15 (`/synod design` 세션, Gemini flash conf95(Solver)/92(Critic) + OpenAI o3
conf68(Solver)/67(Critic) — 1위 원인에서 조기 수렴, 2위 원인 비중에 대해서만 이견이 남아 Critic
라운드까지 진행하고 Defense 라운드는 생략(추가 debate로 인한 수렴 이득이 낮다고 Judge 판단))

대상: `AI_design_v1_13.py`, `uni-section/code/uni_section_v21.py`(v1_13이 실제로 `import`하는
파일. `build_collision_spec`/`compute_collision_loss_v5`는 v20을 참조하지 않고 v21 파일 내부에
L612/L654로 그대로 복사되어 있음 — 상세는 아래 "코드 근거 상세" 참고), `reports/v1_13/AI_design_v1_13.md`,
`initial_section/initial_section_v2.csv`, `reports/v1_13/final_section_v1_13.csv`

---

## 요약

v1.13은 initial section 좌표 로더를 CSV 기반(`build_bpillar_from_csv`)으로 교체하면서, **입력
형상(`initial_section_v2.csv`) 자체는 파트 간 겹침이 없도록(부동소수점 오차 수준으로만 접합)
사전 검증된 상태**였다. 그런데 700-epoch 학습을 마친 결과(`reports/v1_13/final_section_v1_13.csv`)에는
Part0↔Part1(floor 12~16, 최대 0.35mm), Part1↔Part2(floor 5~16, 최대 0.70mm), Part1↔Part4(floor
0~12, 최대 0.65mm) 세 쌍에서 실질적인 파트 간 겹침이 발생했다. 동시에 모든 파트의 최종 두께가
초기값(2.3/1.6/1.6/1.4/1.6mm)과 무관하게 전부 `T_MAX_V13=2.3mm`로 균일하게 saturate됐다.

**근본 원인(1위, Gemini/OpenAI 공통 conf 상위 수렴)**: `uni_section_v21.compute_collision_loss_v5()`
(v21 파일 자체 내부에 정의된 함수 — v20의 동명 함수와 코드가 동일하지만 import 관계가 아니라
파일 복사본, 상세는 아래 참고)가 섹션×파트쌍 전체 방향(direction) 153개에
대해 개별 위반량을 `top_k_fraction=0.3` pooling으로 뽑아낸 뒤, 최종적으로

```python
total_loss = total_loss / n_dirs   # n_dirs ≈ 153
```

로 **전 방향 단순 평균**을 취한다. 실제로 겹침이 발생하는 방향은 이 중 극소수(3개 파트쌍 × 일부
floor)뿐이므로, 그 국소 위반의 그래디언트가 1/153 수준으로 희석되어 두께를 키우려는 물리(Mp) 손실
압력에 압도당했다. 이는 이 프로젝트가 과거에도 반복적으로 겪은 "mean 희석" 결함(`compute_mesh_order_loss()`
v1.8, `compute_smoothness_lse_v3()` v1.10)과 동일한 패턴이며, 유독 `compute_collision_loss_v5()`에는
아직 이 수정이 적용되지 않았다.

**근본 원인(2위, 1위를 구조적으로 고착시키는 보조 요인 — Critic 라운드에서 비중에 이견은 있었으나
존재 자체는 두 모델 모두 인정)**: `CGDN17.forward()`의 `fix_x_mask`/`fix_y_mask`로 겹침이 발생한
접합부 노드들의 좌표가 고정되어 있어, `gap = sigma*proj - t_sum_half - clearance` 수식에서 `proj`
(좌표 항)는 그래디언트를 받지 못하고 `t_sum_half`(두께 항)만 유일한 회피 경로로 남는다. 즉 좌표
이동으로 간섭을 피하는 선택지가 원천 차단된 상태에서, 그 유일한 회피 경로(두께 감소)의 신호가
1위 원인 때문에 극도로 약해진 것이 겹침 고착의 메커니즘이다.

**보조/부차 요인(3위, 설명력 부족하지만 존재 자체는 확인)**: `build_collision_spec()`이 학습 시작
시 **단 한 번**, 초기(미변형) 좌표·두께로 `clearance`를 고정한다. `initial_section_v2.csv`가
이미 거의 0 여유(맞닿음)로 검증된 형상이므로 `clearance = min(0.5, slack-0.05) ≈ -0.05mm`로
계산되어 약 0.05mm의 침범을 애초에 "정상"으로 허용한다. 다만 관측된 겹침(0.35~0.70mm)이 이보다
훨씬 크므로, 이 요인만으로는 결과를 설명하지 못하며 1위 원인의 보조 요인으로 봐야 한다.

---

## 코드 근거 상세

### 1) 전 방향 평균 희석 — `uni-section/code/uni_section_v21.py` L654 `compute_collision_loss_v5()`

(v1_13이 실제 `import`하는 파일은 v21이며, 이 함수는 v21 L654에 직접 정의되어 있다 — v20의
동명 함수를 참조/상속하는 구조가 아니라, v21이 v20의 사본이라 코드가 그대로 복제되어 있을 뿐이다.
v21 파일 헤더 주석: "uni_section_v21.py는 uni_section_v20.py의 사본이며, 초기 형상 생성 방식만
다르다... 그 외 모든 함수는 uni_section_v20.py와 동일". `build_bpillar_from_csv()`만 v21의
신규 함수이고, `compute_collision_loss_v5`/`build_collision_spec`은 로직 변경 없이 그대로
복사됐다.)

```python
def compute_collision_loss_v5(new_coords, t_final, part_ids, section_ids, collision_spec, z_gate=None,
                               top_k_fraction=0.3):
    ...
    for (sec_int, a, b), directions in collision_spec.items():
        ...
        for d in directions:
            ...
            gap = d['sigma'] * proj - t_sum_half - d['clearance']
            violation = torch.relu(-gap).clamp(max=1.0) * valid.float()
            sq_v = (violation ** 2)[valid.bool()]
            if sq_v.numel() > 0:
                k_col = max(1, int(sq_v.numel() * top_k_fraction))
                top_sq_v, _ = torch.topk(sq_v, k=k_col, largest=True)
                loss_dir = top_sq_v.mean()
            ...
            total_loss = total_loss + loss_dir
            n_dirs += 1

    if n_dirs > 0:
        total_loss = total_loss / n_dirs   # ★ 153개 direction 전체 평균 — 여기가 희석 지점
    return total_loss
```

`build_collision_spec()`은 `AI_design_v1_13.py` L2066에서 `collision_spec = usv18.build_collision_spec(...)`로
학습 시작 시 딱 1회 호출되며, 실제 학습 로그(`reports/v1_13/AI_design_v1_13.md` 인접 버전 로그
기준, `[collision v5] 153쌍(섹션x파트쌍) 부호 앵커 산정 완료`)에서 direction 수가 153개(v1.13
자체 로그에서도 유사 규모로 재현될 것으로 판단됨)임을 확인했다. 즉 각 direction 내부에서는
Top-K(30%)로 국소 위반에 집중하지만, direction들 "사이"의 합산은 Top-K가 아니라 단순 평균이라
방향(파트쌍) 층위에서 동일한 희석이 재발한다 — 이 프로젝트가 `compute_mesh_order_loss()`(v1.8)와
`compute_smoothness_lse_v3()`(v1.10)에서 이미 발견·수정했던 "mean 희석" 패턴이 `compute_collision_loss_v5()`
자체에는 아직 적용되지 않았다.

### 2) 좌표 고정으로 인한 단일 회피 경로 — `AI_design_v1_13.py` `CGDN17.forward()`

```python
delta_x = delta_coords[:, 0:1] * (~fix_x_mask).float()
delta_y = delta_coords[:, 1:2] * (~fix_y_mask).float()
```

`fix_x_mask`/`fix_y_mask`가 1인 노드는 `delta_coords`가 강제로 0이 되어 `new_coords`가 고정된다.
`compute_collision_loss_v5()`의 `gap = sigma*proj - t_sum_half - clearance`에서 `proj`는 좌표의
함수이므로, 좌표가 고정된 접합부에서는 `∂gap/∂proj` 경로의 그래디언트가 애초에 존재하지 않고
`t_sum_half`(두께)만 유일하게 조정 가능한 변수로 남는다. Critic 라운드에서 두 모델 모두 이 메커니즘
자체는 사실로 확인했으며, 다만 "이것이 1위 원인의 희석 정도를 수식적으로 더 키우는가"에는 이견이
있었다(Gemini: 강화 관계로 봄 / OpenAI critic: t-gradient 크기 자체(`-0.5` 상수)는 변하지 않으므로
'강화'라는 표현은 과장이나, 대안 경로가 없어진다는 점에서 무시할 수 없는 보조 원인이라는 데는 동의).

### 3) 초기 1회 고정 clearance — `uni_section_v21.py` L612 `build_collision_spec()`

```python
slack = (sigma * vals).min().item() - (t_a0 + t_b0) / 2.0
clearance = min(clearance_default, slack - init_buffer)   # clearance_default=0.5, init_buffer=0.05
```

`initial_section_v2.csv` 기준 접합부 slack이 거의 0이므로 `clearance ≈ -0.05mm`로 계산되어,
약 0.05mm의 침범은 애초에 위반으로 잡히지 않는 여유가 학습 시작부터 고정된다. 이 값 자체는 학습
700epoch 내내 재계산되지 않는다(호출은 학습 루프 진입 전 단 1회, `AI_design_v1_13.py` L2066).
다만 이 정적 0.05mm 버퍼만으로는 관측된 0.35~0.70mm 겹침을 설명할 수 없어, 1위 원인의 부차적
기여로 분류한다.

### 4) 두께 saturate 원인(참고) — 물리(Mp) 손실 압력

`reports/v1_13/AI_design_v1_13.md` 계열 로그에서 `MpErr`가 학습 후반까지 13~25% 수준에서
정체되어 물리 손실이 완전히 수렴하지 못한 채 지속적인 상방 압력을 가하는 것으로 관찰된다. 이
압력과 (1)의 희석된 collision loss가 경쟁하면서, 두께가 collision 제약을 사실상 무시하고
`T_MAX_V13=2.3mm`까지 밀려 올라간 것이 최종 saturate의 직접 동인이다. `l_col_aux`(v1.4/v1.9의
별도 무클램프 보조 충돌 페널티, `W_COL_AUX=5.0`)도 로그상 간헐적으로 스파이크했다가 0.0000으로
돌아오는 패턴을 보여, 지속적인 억제력으로 작동하지 못했음을 시사한다.

### 기각/근거 부족으로 분류한 후보

- **z_gate pair_gate 무력화**: `compute_collision_loss_v5()`의 `pair_gate = z_gate[seg]*z_gate[pt]`는
  후보 파트(candidate, part 3/4)에만 유효한 감쇠이며, 실제 겹침이 발생한 Part0↔Part1(둘 다
  `S_PROTECT`, `z_gate=1.0` 고정)에는 적용되지 않는다. 따라서 z_gate가 이번 겹침의 원인이라는
  근거는 부족하다(OpenAI 지적, 코드 확인으로 재확인).
- **`.clamp(max=1.0)`으로 인한 그래디언트 완전 소실**: Critic 라운드에서 OpenAI가 새로 지적한
  후보로, `gap < -1.0mm`(1mm 이상 깊은 침투)에서는 violation이 포화되어 추가 그래디언트가 사라진다.
  다만 이번 결과의 최대 겹침이 0.70mm로 이 포화 임계(1.0mm)에 도달하지 않았으므로, 현재
  데이터로는 결정적 원인이 아니라 **잠재적 리스크(향후 겹침이 더 커질 경우 악화 요인)**로만 기록한다.

---

## 권장 수정 방향

우선순위는 위 근본 원인 순위와 대응한다. 실행 여부/구체 파라미터는 후속 `/synod design` 세션에서
재검증 권장.

1. **[핵심] `compute_collision_loss_v5()`의 direction-레벨 평균을 Top-K 또는 활성 위반(violation>0)
   기준 가중 평균으로 교체.** 현재 `total_loss / n_dirs`(153) 단순 평균을 이 프로젝트가 이미
   `compute_mesh_order_loss()`/`compute_smoothness_lse_v3()`에 적용한 것과 동일한 패턴(활성/위반
   방향에만 집중)으로 바꾸면, 국소 겹침의 그래디언트가 희석되지 않고 물리 손실과 대등하게 경쟁할
   수 있다.
2. **[보조] `build_collision_spec()`의 clearance 재계산 주기 검토.** 학습 시작 시 1회 고정 대신,
   두께가 크게 변하는 시점(예: Stage4 진입 시점, 또는 N epoch마다)에 현재 `t_final` 기준으로
   `clearance`를 재추정하거나, 최소한 `init_buffer`를 늘려 초기 near-zero slack 상황에서 음수
   clearance가 나오지 않도록 가드레일을 추가.
3. **[보조] `.clamp(max=1.0)` 포화 구간 완화.** Huber형 또는 무클램프 이차 페널티(이 프로젝트가
   `AI_design_v1_9.compute_collision_penalty_unclamped()`에서 이미 쓴 패턴)를 collision loss에도
   적용해, 깊은 침투일수록 그래디언트가 죽지 않고 오히려 강해지도록 조정.
4. **[검증 필요, 이번 리뷰 범위 밖]** 위 수정 적용 후 `T_MAX_V13`까지 두께가 균일하게 saturate되는
   현상 자체가 완화되는지, 그리고 완화되지 않는다면 물리(Mp) 손실 가중치 대비 collision loss
   가중치 배분을 별도로 재조정해야 하는지 후속 세션에서 재확인.

---

<details>
<summary>숙의 과정</summary>

### 모델 기여
- **Gemini (Architect, Solver conf95→Critic conf92, can_exit=true 양 라운드):** 1위 원인(n_dirs
  희석)을 최초 제시하고, Critic 라운드에서 좌표 고정(2위 원인)이 1위 원인의 유일한 대안 경로를
  차단해 결합적으로 겹침을 고착시킨다는 인과관계를 정식화.
- **OpenAI (Explorer, Solver conf68→Critic conf67, can_exit=false 양 라운드):** 1위 원인에 동일하게
  수렴하면서도, z_gate/좌표고정 등 다른 후보들을 코드 근거로 하나씩 반증·격하해 원인 순위의
  엄밀성을 높였고, Critic 라운드에서 `.clamp(max=1.0)` 포화 구간이라는 두 모델 모두 놓쳤던
  잠재 리스크를 새로 지적.

### 해결된 주요 쟁점
1. "좌표 고정(c)이 겹침의 결정적 원인인가, 무관한가?" → 결정적 단독 원인은 아니되, 1위 원인(n_dirs
   희석)의 유일한 대안 경로를 차단하는 결합적 보조 원인으로 판정(양 모델 Critic 라운드에서 수렴).
2. "정적 clearance(a)가 근본 원인인가?" → 아니오, 관측된 겹침 크기(0.35~0.70mm)를 설명하기에
   -0.05mm 버퍼는 너무 작아 부차 요인으로만 인정.

### 신뢰 점수
- Gemini: Solver 95, Critic 92 (High trust)
- OpenAI: Solver 68, Critic 67 (Good trust)
- Claude(조율자): 코드 직접 대조로 z_gate 기각 근거 및 clamp 포화 임계(0.70mm < 1.0mm) 재확인

</details>
