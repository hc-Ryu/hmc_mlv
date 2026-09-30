"""격자 셀 패치 테스트 (idea_v1.md §3.3, §8.1-3): 평판 격자의 막 강성을 연속체 판과 비교.

평판(x–z 평면, 법선 y)을 격자로 만들고 경계 절점에 변위장을 부여해 변형에너지로 등가 강성을 구한다.
  - 단순 전단  u_x = γ z            → G_eff t  vs  G t
  - 단축 인장  u_z = ε z (측면 자유) → E_eff t  vs  E t
out-of-plane DOF(u_y, r_x, r_z)는 구속해 막 거동만 본다.
"""
from __future__ import annotations

import numpy as np
import openseespy.opensees as ops

from types import SimpleNamespace

from .lattice import inplane_inertia, strip_fibers


def _grid(a, c, nx, nz, t, E, nu, mode, nw=2, nt=6):
    icfg = SimpleNamespace(inplane_mode=mode, inplane_factor=1.0, nu=nu)
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    tag = lambda i, k: 1000 * k + i + 1
    for k in range(nz + 1):
        for i in range(nx + 1):
            ops.node(tag(i, k), i * a, 0.0, k * c)
            ops.fix(tag(i, k), 0, 1, 0, 1, 0, 1)
    ops.uniaxialMaterial("Elastic", 1, E)
    G = E / (2 * (1 + nu))
    e = 0

    def add(n1, n2, b, L):
        nonlocal e
        e += 1
        ops.geomTransf("Linear", e, 0.0, 1.0, 0.0)
        ops.section("Fiber", e, "-GJ", G * b * t ** 3 / 3)
        for y, z, A in strip_fibers(b, t, nw, nt, inplane_inertia(b, t, L, icfg)):
            ops.fiber(y, z, A, 1)
        ops.beamIntegration("Lobatto", e, e, 3)
        ops.element("dispBeamColumn", e, n1, n2, e, e)

    for k in range(nz + 1):                       # 원주 방향 (x): 폭 = c (가장자리 c/2)
        b = c * (0.5 if k in (0, nz) else 1.0)
        for i in range(nx):
            add(tag(i, k), tag(i + 1, k), b, a)
    for k in range(nz):                           # 종방향 (z): 폭 = a (가장자리 a/2)
        for i in range(nx + 1):
            b = a * (0.5 if i in (0, nx) else 1.0)
            add(tag(i, k), tag(i, k + 1), b, c)
    return tag


def _energy(tag, nx, nz, disp_fn, free_lateral=False):
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    bnd = []
    for k in range(nz + 1):
        for i in range(nx + 1):
            on_edge = i in (0, nx) or k in (0, nz)
            if free_lateral and k not in (0, nz):
                continue
            if on_edge:
                bnd.append((i, k))
    for i, k in bnd:
        ux, uz = disp_fn(i, k)
        ops.sp(tag(i, k), 3, uz)
        if not free_lateral:
            ops.sp(tag(i, k), 1, ux)
    if free_lateral:                              # 강체 이동만 막기
        ops.sp(tag(0, 0), 1, 0.0)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1e-10, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    ops.reactions()
    W = 0.0
    for k in range(nz + 1):
        for i in range(nx + 1):
            n = tag(i, k)
            W += ops.nodeReaction(n, 1) * ops.nodeDisp(n, 1) + ops.nodeReaction(n, 3) * ops.nodeDisp(n, 3)
    return 0.5 * W


def run(a=8.0, c=16.4, t=2.5, E=210000.0, nu=0.3, mode="calibrated", nx=12, nz=6):
    """반환: (G_eff/G, E_eff/E)."""
    G = E / (2 * (1 + nu))
    area = nx * a * nz * c
    gam, eps = 1e-4, 1e-4
    tag = _grid(a, c, nx, nz, t, E, nu, mode)
    W = _energy(tag, nx, nz, lambda i, k: (gam * k * c, 0.0))
    g_ratio = W / (0.5 * G * t * gam ** 2 * area)
    tag = _grid(a, c, nx, nz, t, E, nu, mode)
    W = _energy(tag, nx, nz, lambda i, k: (0.0, eps * k * c), free_lateral=True)
    e_ratio = W / (0.5 * E * t * eps ** 2 * area)
    return g_ratio, e_ratio


if __name__ == "__main__":
    for a, c in [(5.3, 16.4), (8.0, 16.4), (13.0, 16.4), (18.3, 16.4), (8.0, 65.7)]:
        for mode in ["geometric", "calibrated"]:
            for nx, nz in [(12, 6), (24, 12)]:
                g, e = run(a=a, c=c, mode=mode, nx=nx, nz=nz)
                print(f"a={a:5.1f} c={c:5.1f} {mode:10s} grid={nx}x{nz}  G_eff/G={g:6.3f}  E_eff/E={e:6.3f}")
