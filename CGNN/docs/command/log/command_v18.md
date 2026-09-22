# command_v18.md — uni_section_v17_01.py 초기 단면 좌표 현실화 (cosmetic-only)

설계 근거: `/synod idea` 세션 (Gemini Architect conf 95→93.5 avg / OpenAI Explorer conf 80→77.5 avg,
Critic 라운드에서 두 모델 모두 non-penetration/trimming-range 정합성을 핵심 리스크로 지적).
최종 합의 신뢰도: **88%**.

## 0. 목적 (범위를 벗어나지 않기 위해 반드시 먼저 읽을 것)

`uni-section/code/uni_section_v17_01.py`의 `build_bpillar_section()` (~line 803)가 만드는 초기
단면 형상은 현재 대부분 **평평한 계단형(step)** y값이다. 목적은 이 초기 형상을 `AI_design_v0.py`의
`lower_section`/`upper_section` (Outer/Reinf/Inner 3-part hat 프로파일: flange 평탄부 → shoulder
경사 → crown 평탄부)에서 **비율적 영감만** 얻어, uni_section의 5-part 구조에 맞게 더 "모자
(hat) 형상"답게 보이도록 y값만 바꾸는 것이다.

**이 작업은 순수 cosmetic geometry 변경이다.** 다음은 절대 변경하지 않는다:
- part 개수 (5), part별 노드 개수 (30, `x = i * dx`, `dx = 160/29` 고정)
- 각 branch의 `x_ratio` 임계값(조건식) 자체와 그에 따른 `fix_x`/`fix_y` 배정 로직
- part 2/3/4의 트리밍(존재) 범위 필터 (`x_ratio < ... or > ...: continue`)
- `t_val`, `fy_val` (part_configs의 두께/항복강도)
- `add_edge()` 기반 엣지 연결 순서, `join_pairs` (현재 빈 텐서)

바꾸는 것은 오직 각 `if/elif` branch 안에서 `y_coord_node`에 대입되는 **숫자/수식**뿐이다.

## 1. Critic 라운드에서 확정된 두 가지 안전장치 (반드시 준수)

1. **비관통(non-penetration) 순서 유지**: 현재 y값 기준으로 Outer Hat(#0) 크라운(60) > Inner
   Hat(#2) 크라운(45) > Inner Plate(#1) 크라운(15) > Patch1(#3, 16.5) > Patch4(#4, 7.45) 순서가
   성립한다. 새 y값도 이 상대적 순서(같은 x 부근에서 outer가 항상 inner보다 위)를 반드시 유지해야
   한다. 이 순서가 깨지면 `compute_collision_loss_v3`가 학습 시작부터 비정상적으로 큰 penalty를
   발생시킨다.
2. **트리밍 범위 정합성**: part 2는 `x_ratio ∈ [0.2, 0.8]`, part 3는 `[0.3, 0.7]`, part 4는
   `[0.7, 0.8]`에서만 노드가 존재한다. 새 y-수식은 이 활성 구간 밖에서 평가되지 않으므로 별도 처리
   불필요하지만, 구간 경계에서 값이 튀지 않도록 각 part의 수식은 자기 자신의 활성 구간(및 그 안의
   기존 fix/no-fix 서브 분기)만 사용해야 한다 — 다른 part의 x_ratio 구간을 참조하지 않는다.

## 2. 범위: Part 0 (Outer Hat)과 Part 2 (Inner Hat)만 수정

Critic 라운드 결과, Part 1/3/4는 이미 3단(tier) 이상의 값으로 나뉘어 있어 collision 리스크 대비
추가 개선 이득이 작다고 판단되었다 (Judge 판단: Gemini의 3안 중 Idea 3 "branch 분리" 및 OpenAI가
요구한 전-part 수정은 리스크 대비 이득이 낮아 기각, Idea 1을 Outer/Inner Hat 두 part로만 스코프
축소). AI_design_v0.py에서 실제로 "hat" 실루엣을 갖는 것도 Outer/Inner 두 part뿐이므로 대응 관계가
가장 명확하다.

**Part 1, 3, 4의 y_coord_node 대입식은 이번 커맨드에서 변경하지 않는다.**

## 3. 구현 지시

### 3.1 헬퍼 함수 추가

`build_bpillar_section()` 함수 정의 직전(또는 함수 최상단, 로컬 함수로)에 순수 함수를 추가한다:

```python
def _hat_profile(x_ratio, flange_y, crown_y, ramp_lo, ramp_hi, plateau_lo, plateau_hi):
    """flange(평탄) -> shoulder(선형 램프) -> crown(평탄) -> shoulder -> flange 형태의
    대칭 사다리꼴(trapezoid) 프로파일. x_ratio는 이미 해당 if/elif 분기 조건을 만족한
    상태에서만 호출되므로 분기 조건 자체는 건드리지 않는다."""
    if x_ratio < plateau_lo:
        # ramp_lo -> plateau_lo 구간 선형 보간 (flange -> crown)
        frac = (x_ratio - ramp_lo) / (plateau_lo - ramp_lo)
        frac = min(max(frac, 0.0), 1.0)
        return flange_y + frac * (crown_y - flange_y)
    elif x_ratio > plateau_hi:
        # plateau_hi -> ramp_hi 구간 선형 보간 (crown -> flange)
        frac = (ramp_hi - x_ratio) / (ramp_hi - plateau_hi)
        frac = min(max(frac, 0.0), 1.0)
        return flange_y + frac * (crown_y - flange_y)
    else:
        return crown_y
```

이 함수는 노드/엣지/조건식과 무관한 순수 수학 함수이므로 토폴로지에 영향을 주지 않는다.

### 3.2 Part 0 (Outer Hat) — else 분기만 수정

현재 (유지):
```python
if part_id == 0:
    if x_ratio <= 0.1667 + eps or x_ratio >= 0.8333 - eps:
        fix = 1.0
        y_coord_node = 30.0
    else:
        y_coord_node = 60.0
```

변경 후 (조건식은 그대로, else의 대입값만 교체):
```python
if part_id == 0:
    if x_ratio <= 0.1667 + eps or x_ratio >= 0.8333 - eps:
        fix = 1.0
        y_coord_node = 30.0
    else:
        y_coord_node = _hat_profile(
            x_ratio, flange_y=30.0, crown_y=65.0,
            ramp_lo=0.1667, ramp_hi=0.8333,
            plateau_lo=0.3334, plateau_hi=0.6666,
        )
```
근거: AI_design_v0.py Outer part는 flange(y=0) 대비 crown이 약 2배(≈54~60) 높다. 여기서는
flange=30 대비 crown을 65로 잡아 유사한 비율(≈2.2배)의 상승감을 재현하되, 기존 crown 값(60)보다
살짝만 높여 collision 여유(part2 crown 45와의 gap)를 과도하게 줄이지 않도록 한다.

### 3.3 Part 2 (Inner Hat) — else 분기만 수정

현재 (유지):
```python
elif part_id == 2:
    if (x_ratio <= 0.3 - eps and x_ratio > 0.2 + eps) or (x_ratio >= 0.7 + eps and x_ratio < 0.8 - eps):
        fix = 1.0
        y_coord_node = 9.65
    else:
        y_coord_node = 45.0
```

변경 후:
```python
elif part_id == 2:
    if (x_ratio <= 0.3 - eps and x_ratio > 0.2 + eps) or (x_ratio >= 0.7 + eps and x_ratio < 0.8 - eps):
        fix = 1.0
        y_coord_node = 9.65
    else:
        y_coord_node = _hat_profile(
            x_ratio, flange_y=9.65, crown_y=45.0,
            ramp_lo=0.2, ramp_hi=0.8,
            plateau_lo=0.4, plateau_hi=0.6,
        )
```
근거: crown 값(45.0)은 Part 0 crown(65.0)보다 항상 낮게 유지되어 §1의 비관통 순서를 그대로
보존한다. flange 값(9.65)도 변경하지 않으므로 Part 1과의 관계도 기존과 동일하다.

### 3.4 Part 1, 3, 4

수정하지 않는다 (§2 참조).

## 4. 검증 체크리스트 (구현 후 반드시 실행)

수정 전/후 각각 `build_bpillar_section()`을 호출해 다음을 비교하는 간단한 스크립트를 작성/실행한다:

```python
data_old, reg_old = build_bpillar_section()   # 수정 전 커밋에서 실행한 결과를 별도 저장해둔 것과 비교, 또는 git stash로 비교
data_new, reg_new = build_bpillar_section()   # 수정 후

assert data_old.x.shape[0] == data_new.x.shape[0], "노드 개수가 바뀜"
assert data_old.edge_index.shape == data_new.edge_index.shape, "엣지 개수가 바뀜"
assert torch.equal(data_old.x[:, 2], data_new.x[:, 2]), "fix_x 마스크가 바뀜"
assert torch.equal(data_old.x[:, 3], data_new.x[:, 3]), "fix_y 마스크가 바뀜"
assert torch.equal(data_old.x[:, 4], data_new.x[:, 4]), "part_id 배열이 바뀜"
assert torch.equal(data_old.x[:, 6], data_new.x[:, 6]), "t_val이 바뀜"
assert torch.equal(data_old.x[:, 7], data_new.x[:, 7]), "fy_val이 바뀜"
assert data_old.join_pairs.numel() == data_new.join_pairs.numel() == 0, "join_pairs가 바뀜"

# 비관통 순서 확인 (§1-1): 동일 x 근방에서 outer(#0) y >= inner hat(#2) y >= inner plate(#1) y
import numpy as np
x_new = data_new.x[:, 0].numpy(); y_new = data_new.x[:, 1].numpy(); pid = data_new.x[:, 4].numpy()
for xr in np.linspace(0.2, 0.7, 11):
    xq = xr * 160.0
    def y_at(part):
        m = (pid == part)
        idx = np.argmin(np.abs(x_new[m] - xq))
        return y_new[m][idx]
    y0, y1, y2 = y_at(0.0), y_at(1.0), y_at(2.0)
    assert y0 >= y2 >= y1, f"x_ratio={xr}: non-penetration 순서 위반 (outer={y0}, inner_hat={y2}, plate={y1})"

print("[OK] uni_section_v17_01.py 토폴로지/BC/비관통 순서 모두 보존됨 — cosmetic y-shape만 변경됨")
```

모든 assert가 통과해야 이번 변경을 완료로 간주한다.

## 5. 숙의 과정 요약

<details>
<summary>Synod 세션 기록</summary>

- **Gemini (Architect, Solver conf 95, Critic conf 92)**: 기존 if/elif 조건은 그대로 두고 대입값만
  piecewise-linear/trapezoid 함수로 교체하는 안(Idea 1)을 제안·추천. Critic 라운드에서 "독립적으로
  각 part를 보간하면 outer/inner hat이 서로 뚫고 지나갈 수 있다"는 non-penetration 리스크를 스스로
  지적.
- **OpenAI (Explorer, Solver conf 80, Critic conf 75)**: 방향에는 동의하되 트리밍 범위(part
  2/3/4) 정합성과 구체적 수치 스킴 부재, 검증 스크립트 필요성을 지적.
- **Judge(Claude) 판단**: 두 모델이 공통으로 요구한 "비관통 순서 유지 + 구체적 수치 + 회귀 검증"을
  모두 반영하되, 리스크/이득을 고려해 수정 범위를 Part 0/2(실제 hat 형상을 가진 두 part)로
  축소하고 Part 1/3/4는 원안 그대로 유지하도록 스코프를 좁힘.
- **최종 신뢰도**: 88% (Gemini 평균 93.5, OpenAI 평균 77.5, Trust-weighted).

</details>
