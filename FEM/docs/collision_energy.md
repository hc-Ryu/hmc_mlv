# 측면 충돌 시 B-pillar 흡수 에너지 추정 (FEM `--energy-target` 근거)

작성: 2026-10-06 · 대상: `bpillar_fem_v2.py --energy-target` 목표값 선정

## 1. 왜 에너지 기준인가

- 충돌 시험은 대차 질량·속도가 정해져 있어 **입력 운동에너지(½mv²)가 같다.** 설계를 같은 충돌 조건에서 비교하려면 같은 에너지를 넣어야 한다.
- `bpillar_fem_v2.py`는 시간·질량이 없는 준정적 변위제어 해석이라 충격량(∫F dt)은 정의되지 않는다. 대응하는 양은 **외력 일 ∫F dδ (흡수 에너지)**이다.
- 질량이 같으면 충격량을 같게 하는 것은 속도를 같게 하는 것과 같고, 결국 에너지를 같게 하는 것과 같은 조건이다.
- 외력 일은 하중 절점마다 (힘 × y 변위 증분)을 사다리꼴로 누적한다. 분포 하중이라 "총 하중 × 제어점 변위"로 계산하면 과대평가된다.

## 2. 시험 조건별 입력 운동에너지

| 조건 | 대차 질량 | 속도 | ½mv² |
|:--|:--|:--|:--|
| 예시: 2 t 차량, 30 km/h | 2000 kg | 8.33 m/s | 69.4 kJ |
| FMVSS 214 (MDB) | 1368 kg | 54 km/h | 약 154 kJ |
| **IIHS 측면 2.0** | 1905 kg (4200 lb) | 약 60 km/h (37 mph) | **약 260–265 kJ** |

IIHS 2.0은 기존 IIHS 측면 시험보다 에너지가 82% 많다 [S1, S2].

## 3. IIHS 측면 2.0에서 B-pillar 몫 추정

### 3.1 변형에 쓰이는 에너지 (계산, 완전 비탄성 충돌)

맞은 차(질량 m₂)가 정지 상태이고 충돌 후 두 물체가 함께 움직인다고 보면, 변형에 쓰이는 에너지는

E_def = KE × m₂ / (m₁ + m₂)  (m₁ = 배리어 1905 kg)

| 맞은 차량 m₂ | E_def |
|:--|:--|
| 1.5 t | 약 117 kJ |
| 1.8 t | 약 129 kJ |
| 2.0 t | 약 136 kJ |

완전 비탄성은 변형 에너지의 상한 쪽 가정이다. 실제로는 반발, 차량 회전, 타이어 마찰로 일부가 빠진다.

### 3.2 배리어와 차량의 분담 (추정 — 문헌 수치 미확보)

- IIHS 배리어 앞면은 알루미늄 허니콤 변형체라 스스로 상당량을 흡수한다. 구형 유럽 배리어(EEVC)는 인증 조건이 35 km/h에서 총 45 kJ 흡수였다 [S5]. IIHS 2.0 배리어의 값은 아니고, 배리어 앞면이 수십 kJ를 흡수한다는 정성적 근거로만 쓴다.
- 배리어 몫을 E_def의 30–50%로 가정하면 **차량 측면 몫은 60–90 kJ**이다.
- **이 30–50%는 출처를 확인한 수치가 아니다.** 검색 중 본 "배리어 42% / 차량 38%" 같은 수치는 원 논문을 확인하지 못해 쓰지 않았다.

### 3.3 차량 측면 안에서 B-pillar의 몫

- IIHS 배리어는 주로 B-pillar와 도어를 때리고 A·C-pillar와는 거의 닿지 않는다. 대부분 세단에서 배리어가 sill보다 높은 곳을 쳐서 "모든 에너지를 B-pillar와 도어 구조가 흡수해야 한다" [S3].
- front-to-side 충돌 연구에서 운전석 도어의 변형 일은 B-pillar보다 15% 크다 [S4]. 즉 도어 ≈ 1.15 × B-pillar.
- IIHS 배리어는 B-pillar 중심이라 앞·뒤 도어 2개가 함께 하중을 받는다. 그러면 도어 2개 ≈ 2.3 × B-pillar이고, **B-pillar 비율 ≈ 1 / (1 + 2.3) ≈ 30%**이다. sill·바닥 몫을 빼면 실제로는 이보다 약간 작다.

### 3.4 결론

| 단계 | 값 | 근거 |
|:--|:--|:--|
| 입력 운동에너지 | 260–265 kJ | 계산 [S1] |
| 변형 에너지 (맞은 차 1.5–2.0 t) | 120–135 kJ | 계산 |
| 차량 측면 몫 | 60–90 kJ | 추정 (문헌 수치 없음) |
| B-pillar 비율 | 약 30% | [S3], [S4] |
| **B-pillar 흡수 에너지** | **약 15–30 kJ (중앙값 약 20 kJ)** | |

## 4. FEM 해석 조건

```bash
python bpillar_fem_v2.py --csv <단면.csv> --out-dir <결과 폴더> --energy-target 30 --energies 10 15 20 25 30
```

- 기준값은 20 kJ(중앙값)이다. 분담 비율의 불확실성이 크므로 10–30 kJ 범위 전체의 Plate 침입량을 본다.
- 모델 한계: 양단 완전 고정이고 도어의 하중 분담이 없다. 같은 에너지를 넣으면 실차보다 B-pillar 변형이 클 가능성이 높다(보수적).

### 4.1 결과: v23 후처리 단면 (`results/v23_energy/`, 2026-10-06)

| 흡수 에너지 (kJ) | 10 | 15 | 20 | 25 | 30 |
|:--|:--|:--|:--|:--|:--|
| 그때 하중 (kN) | 496 | 555 | 610 | 681 | 753 |
| 최대 Plate 침입 (mm) | 10.8 | 15.5 | 21.6 | 28.9 | 36.8 |
| 최대 섹션 | 8 | 9 | 9 | 9 | 8 |
| 상태 | stable | stable | snap_through | snap_through | snap_through |

1차 한계하중 574 kN은 약 16 kJ에서 나온다. 기준값 20 kJ는 한계하중을 넘은 뒤(snap_through)라서, 그 값은 관성을 뺀 정적 평형값이다.

## 5. 출처

원문 확인 여부를 함께 적는다. "요약만"은 검색 결과 요약으로만 확인했고 본문은 열지 못했다(403·유료)는 뜻이다.

| 번호 | 자료 | 확인 | 쓴 내용 |
|:--|:--|:--|:--|
| S1 | IIHS, *Small SUVs struggle in new, tougher side test* — https://www.iihs.org/news/detail/small-suvs-struggle-in-new-tougher-side-test | 요약만 | 4200 lb, 37 mph |
| S2 | IIHS Status Report 54(7), *New side test will have more impact* — https://www.iihs.org/api/datastoredocument/status-report/pdf/54/7 | 요약만 | 기존 대비 에너지 82% 증가 |
| S3 | Reichert, Kan, Arnold-Keifer, *IIHS Side Impact Parametric Study using LS-DYNA*, 15th Int. LS-DYNA Users Conf., 2018 — https://lsdyna.ansys.com/wp-content/uploads/2022/11/iihs-side-impact-parametric-study-using-ls-dyna-r.pdf | **원문** | B-pillar·도어 집중 하중, 속도·질량 민감도 (50→60 km/h: 침입 +6 cm, 1500→2000 kg: +2 cm) |
| S4 | *Impact energy and the risk of injury to motorcar occupants in the front-to-side vehicle collision*, Nonlinear Dynamics, 2022 — https://link.springer.com/article/10.1007/s11071-022-07779-8 | 요약만 | 도어 변형 일 = B-pillar × 1.15 |
| S5 | EEVC WG6, *Structures — Improved side impact protection in Europe*, 9th ESV, 1982 — https://www.eevc.net/EEVC/EN/Past/WG06/Side-Impact.pdf | **원문** | 구형 배리어 인증: 35 km/h, 총 45 kJ 흡수 |
| (참고) | *Experimental study on side pole collision and deformable barrier collision of car* — https://www.researchgate.net/publication/342555384 | 요약만 | MDB 측면 차량 흡수 39.2 kJ, 폴 50.3 kJ (시험 조건 미확인 — 추정에 쓰지 않음) |

**확인하지 못한 것:** IIHS 2.0에서 B-pillar 흡수 에너지를 직접 밝힌 공개 자료, 배리어/차량 분담 비율. 정확한 값이 필요하면 OEM 해석 자료나 IIHS 2.0 배리어 사양서(force–deflection)로 배리어 흡수량을 계산해야 한다.
