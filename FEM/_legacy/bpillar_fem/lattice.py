"""OpenSeesPy 격자 모델 구성 (idea_v1.md §3, §4, §7.1)."""
from __future__ import annotations

import numpy as np
import openseespy.opensees as ops

from .config import N_SECTIONS
from .geometry import Mesh


def inplane_inertia(b: float, t: float, L: float, cfg) -> float:
    """strip 면내(강축) 굽힘 I.

    'calibrated': 강결 격자(Vierendeel)의 셀 전단 유연도
        γ/q = (1/Et) (a²/(f_c c²) + c²/(f_l a²))
    를 판의 1/(Gt)와 같게 두고 두 방향에 반씩 나누면, 요소마다 I = b t L² / (12(1+ν)) 가 된다
    (b = strip 폭, L = 요소 길이). patch_test.py로 검증.
    """
    if cfg.inplane_mode == "calibrated":
        i = b * t * L * L / (12.0 * (1.0 + cfg.nu))
    elif cfg.inplane_mode == "geometric":
        i = t * b ** 3 / 12.0
    else:
        raise ValueError(cfg.inplane_mode)
    return cfg.inplane_factor * i


def strip_fibers(b: float, t: float, nw: int, nt: int, i_inplane: float):
    """폭 b × 두께 t strip의 파이버 (y: 폭 방향 = 면내, z: 두께 방향 = 판 법선).

    두께 방향은 균등 분할(판 굽힘 소성 힌지), 폭 방향 위치는 면내 I가 i_inplane이 되도록 스케일한다.
    """
    ys = -b / 2 + (np.arange(nw) + 0.5) * b / nw
    ys *= np.sqrt((i_inplane / (b * t)) / np.mean(ys ** 2))
    zs = -t / 2 + (np.arange(nt) + 0.5) * t / nt
    area = b * t / (nw * nt)
    return [(float(y), float(z), area) for y in ys for z in zs]


def build_model(mesh: Mesh, with_mass: bool = True) -> None:
    cfg = mesh.cfg
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    for tag, (x, y, z) in mesh.coords.items():
        ops.node(tag, float(x), float(y), float(z))

    for p in mesh.parts:   # 재료 tag = pid+1
        ops.uniaxialMaterial("Steel02", p.pid + 1, p.fy, cfg.E, cfg.hardening, 20.0, 0.925, 0.15)

    G = cfg.E / (2.0 * (1.0 + cfg.nu))
    for i, e in enumerate(mesh.elements, start=1):
        mat = e.pid + 1
        if e.kind == "diag":
            ops.element("corotTruss", i, e.n1, e.n2, e.area, mat)
            continue
        ops.geomTransf("Corotational", i, *e.vecxz)
        ops.section("Fiber", i, "-GJ", G * e.b * e.t ** 3 / 3.0)
        i_in = inplane_inertia(e.b, e.t, e.length, cfg)
        for y, z, a in strip_fibers(e.b, e.t, cfg.n_width, cfg.n_thick, i_in):
            ops.fiber(y, z, a, mat)
        ops.beamIntegration("Lobatto", i, i, 3)
        ops.element("dispBeamColumn", i, e.n1, e.n2, i, i)

    # 경계조건 (§4): sec 0(sill), sec 16(roof rail) 레벨의 모든 절점
    fix = [1, 1, 1, 1, 1, 1] if cfg.bc == "fixed" else [1, 1, 1, 0, 0, 0]
    if cfg.bc not in ("fixed", "pinned"):
        raise ValueError(cfg.bc)
    for sec in (0, N_SECTIONS - 1):
        for nd in mesh.section_nodes(sec):
            ops.fix(nd, *fix)

    if with_mass:
        for nd, m in mesh.mass.items():
            j = mesh.rot_inertia[nd]
            ops.mass(nd, m, m, m, j, j, j)
