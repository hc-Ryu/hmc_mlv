# B-pillar 격자 FEM (데모)

CGNN 설계 결과(단면 CSV)를 받아 B-pillar를 판 strip 빔 격자로 단순화하고, 섹션 5–11에 −y 방향 하중을 걸어 **섹션별 변형량 CSV**를 출력합니다. 정밀 충돌 해석이 아니라 "설계 입력 → 변형량 데이터" 데모입니다. 설계 배경은 `docs/idea/idea_v1.md`를 참고하세요.

## 실행: `bpillar_fem_demo.py` (단일 파일)

```bash
conda activate fem          # Python 3.12 + openseespy 3.8 (3.11에서는 openseespy import 실패)
cd hmc_mlv/FEM

python bpillar_fem_demo.py                                   # 기본 설계 CSV, 약 20분
python bpillar_fem_demo.py --quick                           # 거친 격자, Plate 30 mm까지 (약 4분)
python bpillar_fem_demo.py --csv 설계.csv --plate-target 80 --forces 100 200 300
```

| 옵션 | 기본값 | 내용 |
|:--|:--|:--|
| `--csv` | CGNN v1_21_0_x1.3 최종 단면 | 입력 단면 CSV (`floor_idx, point_idx, x_mm, y_mm, r_mm, t_mm`, 5개 파트 × 17섹션) |
| `--out-dir` | `FEM/results` | 결과 폴더 |
| `--name` | CSV 파일 이름 (`--quick`이면 `_quick` 추가) | 결과 파일 이름 앞부분 |
| `--plate-target` | 50 | **Plate 최대 침입량이 이 값(mm)에 닿을 때까지 하중을 올린다** |
| `--force-target` | – | **총 하중이 이 값(kN)에 닿을 때까지** 하중을 올린다. 주면 `--plate-target` 대신 사용 (예: `--force-target 1050`) |
| `--max-disp` | 300 | 제어 절점 변위 상한 (mm) — 목표 전에 닿으면 멈춤 |
| `--forces` | 도달한 최대 하중까지 6단계 | 결과를 뽑을 하중 수준 (kN) |
| `--n-sub` | 4 | 섹션 사이 분할 수 (클수록 정밀, 느림) |
| `--du` | 1 | 변위 증분 (mm) |
| `--quick` | – | `n_sub=1`, 2 mm 증분, Plate 목표 30 mm |

## 출력 (`FEM/results/<name>_*`)

| 파일 | 내용 |
|:--|:--|
| `<name>_deformation.csv` | 하중 수준 × 섹션별 변형량 (아래 표) |
| `<name>_shapes.csv` | 하중 수준별 **변형된 섹션 단면 좌표**: `F0_kN, status, sec, part, point_idx, x0_mm, y0_mm`(변형 전 = 입력 CSV), `x_mm, y_mm`(변형 후) |
| `<name>_shapes_F<하중>kN.png` | 하중 수준마다 17개 섹션 단면 그림 (점선 = 변형 전, 실선 = 변형 후, 파트별 색) |
| `<name>_curve.csv` | 제어 변위 – 총 하중 곡선 |
| `<name>_opensees.log` | OpenSees 경고 |

**`<name>_deformation.csv`** — 하중 수준 × 섹션(0 = sill … 16 = roof rail) 한 행씩

| 열 | 의미 |
|:--|:--|
| `F0_kN` | 하중 수준 (sec 5–11에 가한 총 하중) |
| `status` | `stable`(1차 한계하중 이하) / `snap_through`(한계하중을 넘어 다음 평형으로 뛰어넘음, 정적 평형값) / `not_reached`(해석 범위 안에 없음 → `--plate-target` 늘리기) |
| `plate_intrusion` | **Plate(가장 안쪽 = 탑승자 쪽 판) 절점의 −y 최대 이동량 (mm) — 최대 변형량 판단 기준** |
| `plate_distortion` | Plate 자체 찌그러짐 (강체운동 제외, mm) |
| `outer_intrusion` | Outer(충돌면) 절점의 −y 최대 이동량 (mm, 참고) |
| `d_crush` | Outer 윗면 – Plate 사이 거리 감소 = 단면 높이 압궤 (mm, 참고) |
| `control_disp`, `z_mm` | 제어 절점 변위, 섹션 높이 위치 |

## 모델 요약

- **절점:** CSV의 5개 파트(Outer / Plate / Inner / Patch1 / Patch2) × 17섹션 point. 섹션 사이는 보간 절점으로 채웁니다.
- **요소:** 섹션 안(원주) 선분과 섹션 사이(같은 point끼리) 선분을 판 strip 파이버 보로 둡니다. 두께 방향 6층이고, 면내 I = b·t·L²/(12(1+ν))로 두어 격자 전단 강성을 판의 G·t와 맞춥니다.
- **파트 접합:** 플랜지 point 쌍을 한 절점으로 병합합니다.
- **경계:** sec 0, sec 16 전 절점을 고정합니다.
- **하중:** sec 5–11 Outer 윗면에 균일 압력을 줍니다. 정적 변위제어로 누른 뒤 곡선에서 하중 수준별 상태를 읽습니다.
- **접촉:** 위 파트와 아래 파트 사이에 y 방향 절점-절점 gap을 둡니다.
- **하중 증가:** Plate 최대 침입량이 `--plate-target`에 닿을 때까지 누릅니다. 탑승자 안전은 가장 안쪽 판(Plate)의 침입으로 판단합니다. `_curve.csv`에 하중과 함께 Plate 최대 침입량이 기록됩니다.

한계: 국부좌굴은 strip 좌굴로만 근사하고, 접촉은 절점-절점(판 두께 범위 약 1 mm 겹침 가능)이며, 한계하중을 넘은 뒤의 값은 관성을 뺀 정적 평형값입니다. 양단 고정 외의 차체 지지는 없습니다.

## `legacy/` (상세 분석용 패키지)

단일 파일로 합치기 전의 패키지 버전입니다. 기본 설정 결과는 단일 파일과 같고(섹션 변형량 차이 < 0.04 mm), 추가로 explicit 동적 해석, crown6/impactor 하중, x 방향 벽–벽 접촉, 파트별 항복 추적, 관통 검사, 그림, 재처리를 지원합니다. `legacy/` 폴더에서 `python run_v1.py -h`, `python -m pytest tests -q`로 실행합니다. 결과는 `FEM/results/`에 저장됩니다.
