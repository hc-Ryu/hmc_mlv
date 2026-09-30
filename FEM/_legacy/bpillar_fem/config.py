"""해석 설정 (idea_v1.md §1.2, §3–§5, §7).

단위계: mm, N, tonne, s, MPa.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

FEM_ROOT = Path(__file__).resolve().parents[2]      # hmc_mlv/FEM (이 패키지는 FEM/legacy/ 아래)
DEFAULT_CSV = (FEM_ROOT.parent / "01. CGNN" / "reports" / "v1_21_0_x1.3"
               / "final_section_v1_21_0_x1.3.csv")

N_SECTIONS = 17                     # CGNN 설계 CSV는 항상 17섹션 (io_csv에서 검사)


@dataclass
class Config:
    # ── 입력 / 기하 (§1.2) ──
    csv_path: str = str(DEFAULT_CSV)
    height: float = 1051.7          # sec 0(sill) ~ sec 16(roof rail) 거리
    n_sub: int = 4                  # C4: 섹션 사이 종방향 분할 수

    # ── 재료 ──
    E: float = 210000.0
    nu: float = 0.3
    rho: float = 7.85e-9
    hardening: float = 0.01         # Steel02 b
    n_thick: int = 6                # 두께 방향 파이버 층수 (>=6 권장, §7.1)
    n_width: int = 2                # 폭 방향 파이버 분할 (>=2 필수, §7.1)
    inplane_mode: str = "calibrated"  # 'calibrated': 면내 I = b*t*L^2/(12(1+nu)) → 격자 전단강성 ≈ G*t
                                      # 'geometric' : 면내 I = t*b^3/12 (전단 25–100% 수준으로 과소)
    inplane_factor: float = 1.0       # 위 면내 I에 곱하는 추가 배율

    # ── 연결 (§3) ──
    # 플랜지 접합은 절점 병합으로 처리한다 (explicit에서 rigidLink가 불안정, §7.2)
    tie_levels: str = "all"         # 'all': 모든 레벨 병합(연속 플랜지), 'sections': 원 섹션 레벨만
    use_diagonals: bool = False     # C2
    diag_area_ratio: float = 0.0    # A_d = ratio * t * 0.5*(seg + dz_sub). 패치 테스트로 보정 필요

    # ── 접촉 (§7.4, 정적 해석 전용) ──
    contact: bool = True
    contact_k: float = 2.0e5        # 쌍당 penalty 강성 (N/mm)
    contact_act: float = 3.0        # 현재 간격 − 접촉거리 < act 이면 쌍 생성 (mm)
    contact_window: float = 6.0     # 짝 찾기 직교 방향 창 반폭 (mm) — 섹션 내 point 간격 5–18 mm
    contact_x: bool = True          # x 방향 벽–벽 접촉(자기 접촉 포함, §7.6)

    # ── 경계조건 (§4) ──
    bc: str = "fixed"               # 'fixed': 양단 전 절점 6DOF 고정, 'pinned': 병진만 고정

    # ── 하중 (§5) ──
    load_secs: tuple = (5, 11)
    crown_pts: tuple = (13, 14, 15, 16, 17, 18)   # Outer crown
    plate_crush_pts: tuple = (13, 14, 15, 16, 17, 18)  # d_crush 기준 Plate 구간
    control_pt: int = 15            # 정적 변위제어 기준 Outer point (하중 구간 가운데 섹션)
    load_mode: str = "top"          # 'top'(윗면 균일 압력, 기본) | 'crown6'(crown 6점) | 'impactor'(강체 평판, 정적 전용) — §7.7
    impactor_k: float = 2.0e5       # 임팩터–Outer 절점 접촉 penalty (N/mm)
    weights: str = "uniform"        # 'uniform' (W1) | 'hann' (W2)
    load_dist: str = "band"         # 'band': sec5–11 사이 모든 레벨 | 'sections': 원 섹션 레벨만
    alpha: float = 1.0              # F0 = alpha * P_c
    Pc: float | None = None         # None이면 CSV 단면에서 계산 (section_props.collapse_load)
    profile: str = "P1"             # P1 ramp-hold | P2 ramp-hold-unload | P3 half-sine
    t_ramp: float = 0.010
    t_hold: float = 0.010
    t_free: float = 0.010           # P2 제하 후 자유 진동 시간
    t_pulse: float = 0.020          # P3 전체 길이

    # ── 풀이 (§5.3) ──
    mode: str = "static"            # 'static'(정적 변위제어, 기본) | 'explicit'(동적, 느림)
    static_target: float = 150.0    # 정적: 제어 절점 목표 변위 (mm)
    static_du: float = 1.0          # 정적: 변위 증분 (mm)
    dt: float | None = None         # explicit: None이면 절점 강성/질량으로 자동 산정
    cfl: float = 0.5
    mass_scale: float = 1.0
    damping_ratio: float = 0.10     # explicit: 질량비례 Rayleigh, f1 기준
    f1_hz: float = 600.0            # explicit: 1차 고유진동수 추정(고정-고정 보)
    n_frames: int = 400             # explicit: 출력 프레임 수

    # ── 출력 ──
    out_dir: str = str(FEM_ROOT / "results")
    run_name: str = ""

    @property
    def dz(self) -> float:
        return self.height / (N_SECTIONS - 1)

    @property
    def control_sec(self) -> int:
        """하중 구간 가운데 섹션 — 정적 변위제어 기준, 붕괴하중 힌지 위치."""
        return (self.load_secs[0] + self.load_secs[1]) // 2

    @property
    def P_c(self) -> float:
        """이상화 3힌지 붕괴하중 상한 (N), CSV 단면에서 계산 (idea_v1.md §5.4)."""
        if self.Pc is not None:
            return self.Pc
        from .section_props import collapse_load_from_csv
        return collapse_load_from_csv(self.csv_path, self.height, tuple(self.load_secs),
                                      self.control_sec)

    @property
    def F0(self) -> float:
        return self.alpha * self.P_c

    def name(self) -> str:
        if self.run_name:
            return self.run_name
        return f"{self.mode}_{self.profile}_a{self.alpha:.2f}_n{self.n_sub}_{self.bc}"

    def save(self, path: Path) -> None:
        d = asdict(self)
        d.update(dz=self.dz, P_c=self.P_c, F0=self.F0)
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
