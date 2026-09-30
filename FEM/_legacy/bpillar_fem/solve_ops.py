"""해석 실행 (idea_v1.md §5.3): explicit 동적 해석, 정적 변위제어."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import openseespy.opensees as ops

from . import lattice, loads
from .config import N_SECTIONS, Config
from .geometry import Mesh, build_mesh
from .io_csv import OUTER, OUTER_BODY_PTS, read_parts


@dataclass
class History:
    """기록 절점의 변위 이력. U[frame, node, 0:3] = (ux, uy, uz)."""
    nodes: list
    times: list = field(default_factory=list)
    force: list = field(default_factory=list)       # 총 하중 (N)
    U: list = field(default_factory=list)
    kinetic: list = field(default_factory=list)
    ext_work: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def as_arrays(self):
        return (np.asarray(self.times), np.asarray(self.force), np.asarray(self.U))


def _record_nodes(mesh: Mesh, frac: dict) -> list[int]:
    nodes, seen = [], set()
    for sec in range(N_SECTIONS):
        for nd in mesh.section_nodes(sec):
            if nd not in seen:
                seen.add(nd)
                nodes.append(nd)
    for nd in frac:
        if nd not in seen:
            seen.add(nd)
            nodes.append(nd)
    return nodes


def _snapshot(nodes: list[int]) -> np.ndarray:
    return np.array([ops.nodeDisp(nd)[:3] for nd in nodes])


def prepare(cfg: Config) -> tuple[Mesh, dict]:
    mesh = build_mesh(read_parts(cfg.csv_path), cfg)
    frac = loads.nodal_fractions(mesh)
    return mesh, frac


def run_explicit(cfg: Config, log=print) -> tuple[Mesh, History]:
    mesh, frac = prepare(cfg)
    t0 = time.time()
    lattice.build_model(mesh, with_mass=True)
    log(f"[model] nodes={len(mesh.coords)} elements={len(mesh.elements)} "
        f"dt={mesh.dt:.3e}s build={time.time() - t0:.1f}s")

    times, vals, t_end = loads.time_history(cfg)
    ops.timeSeries("Path", 1, "-time", *times, "-values", *vals)
    ops.pattern("Plain", 1, 1)
    for nd, f in frac.items():
        ops.load(nd, 0.0, -cfg.F0 * f, 0.0, 0.0, 0.0, 0.0)

    alpha_m = 2.0 * cfg.damping_ratio * 2.0 * np.pi * cfg.f1_hz
    ops.rayleigh(alpha_m, 0.0, 0.0, 0.0)

    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("Diagonal")
    ops.algorithm("Linear")
    ops.integrator("ExplicitDifference")
    ops.analysis("Transient")

    dt = mesh.dt
    n_steps = int(np.ceil(t_end / dt))
    per_frame = max(1, n_steps // cfg.n_frames)
    nodes = _record_nodes(mesh, frac)
    hist = History(nodes=nodes)
    load_nodes = list(frac)
    load_vec = np.array([frac[nd] for nd in load_nodes])
    all_nodes = list(mesh.mass)
    masses = np.array([mesh.mass[nd] for nd in all_nodes])

    def ts_value(t):
        return float(np.interp(t, times, vals))

    def record(t, uy_prev):
        U = _snapshot(nodes)
        hist.times.append(t)
        hist.force.append(cfg.F0 * ts_value(t))
        hist.U.append(U)
        v = np.array([ops.nodeVel(nd)[:3] for nd in all_nodes])
        hist.kinetic.append(0.5 * float(np.sum(masses[:, None] * v ** 2)))
        uy = np.array([ops.nodeDisp(nd, 2) for nd in load_nodes])
        dw = 0.0 if uy_prev is None else float(np.sum(-load_vec * (uy - uy_prev)))
        hist.ext_work.append((hist.ext_work[-1] if hist.ext_work else 0.0)
                             + cfg.F0 * ts_value(t) * dw)
        return uy

    uy_prev = record(0.0, None)
    t, step, t_wall = 0.0, 0, time.time()
    while step < n_steps:
        k = min(per_frame, n_steps - step)
        ok = ops.analyze(k, dt)
        step += k
        t = step * dt
        if ok != 0:
            log(f"[explicit] analyze 실패 t={t:.5f}s (ret={ok})")
            hist.extra["failed_at"] = t
            break
        uy_prev = record(t, uy_prev)
        umax = float(np.max(-hist.U[-1][:, 1]))
        if not np.isfinite(umax) or umax > 1.0e3:
            log(f"[explicit] 발산 감지 t={t:.5f}s max(-uy)={umax:.3g}")
            hist.extra["diverged_at"] = t
            break
        if len(hist.times) % max(1, cfg.n_frames // 10) == 0:
            el = time.time() - t_wall
            log(f"  t={t * 1e3:7.2f} ms  step {step}/{n_steps}  max(-uy)={umax:8.3f} mm  "
                f"KE/W={hist.kinetic[-1] / max(hist.ext_work[-1], 1e-12):.3f}  "
                f"wall {el:.0f}s (eta {el / step * (n_steps - step):.0f}s)")
    hist.extra.update(dt=dt, n_steps=step, wall=time.time() - t_wall)
    return mesh, hist


def elem_yield_meta(mesh: Mesh) -> dict:
    """요소별 항복 판정 정보: 파트, 섹션(레벨 중심을 가장 가까운 섹션에 귀속), 파이버 최대 거리, ε_y."""
    cfg = mesh.cfg
    n = len(mesh.elements)
    pid = np.empty(n, dtype=np.int16)
    sec = np.empty(n, dtype=np.int16)
    ymax = np.zeros(n)
    zmax = np.zeros(n)
    eps_y = np.empty(n)
    beam = np.zeros(n, dtype=bool)
    fy = {p.pid: p.fy for p in mesh.parts}
    for i, e in enumerate(mesh.elements):
        pid[i] = e.pid
        lev_c = e.lev + (0.5 if e.kind != "circ" else 0.0)
        sec[i] = int(np.floor(lev_c / cfg.n_sub + 0.5))
        eps_y[i] = fy[e.pid] / cfg.E
        if e.kind != "diag":
            beam[i] = True
            fib = lattice.strip_fibers(e.b, e.t, cfg.n_width, cfg.n_thick,
                                       lattice.inplane_inertia(e.b, e.t, e.length, cfg))
            ymax[i] = max(abs(f[0]) for f in fib)
            zmax[i] = max(abs(f[1]) for f in fib)
    return dict(pid=pid, sec=sec, ymax=ymax, zmax=zmax, eps_y=eps_y, beam=beam)


def scan_yield(meta: dict) -> np.ndarray:
    """요소별 최대 파이버 변형률 / ε_y (3개 적분점 중 최대).

    3D 파이버 단면 변형 = [ε, κ_z, κ_y, θ], 파이버 변형률 = ε − y κ_z + z κ_y.
    파이버가 y × z 격자라 모서리 파이버에서 |ε| + |κ_z| y_max + |κ_y| z_max 가 정확한 최대다.
    """
    n = len(meta["pid"])
    r = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if not meta["beam"][i]:
            continue
        m = 0.0
        for ip in (1, 2, 3):
            d = ops.eleResponse(i + 1, "section", ip, "deformation")
            m = max(m, abs(d[0]) + abs(d[1]) * meta["ymax"][i] + abs(d[2]) * meta["zmax"][i])
        r[i] = m / meta["eps_y"][i]
    return r


IMP_NODE = 9_000_000
IMP_ELE = 8_000_000


def _build_impactor(mesh: Mesh) -> dict:
    """강체 평판 임팩터 (idea_v1.md §7.7).

    y 병진만 자유인 절점 1개가 판 전체를 대표한다. sec 5–11 사이 모든 레벨의 Outer 비플랜지 절점
    (pt 4–27)과 y 방향 압축 전용 gap으로 잇는다. 판 면은 각 레벨의 crown 최고점에 맞춘
    (z 방향으로 기둥 윤곽을 따르는) x 방향 무한 평면이다 — 초기 gap = crown 최고점 − 절점 y.
    x로 무한한 평면이라 절점이 옆으로 미끄러져도 y gap 쌍이 그대로 정확하다.
    참조 하중 1 kN(−y)을 임팩터 절점에 걸고 그 절점을 변위제어 → load factor = 총 하중(kN).
    """
    from .loads import load_levels
    cfg = mesh.cfg
    levs, _ = load_levels(mesh)
    ops.node(IMP_NODE, 0.0, 0.0, 0.0)
    ops.fix(IMP_NODE, 1, 0, 1, 1, 1, 1)
    ops.load(IMP_NODE, 0.0, -1000.0, 0.0, 0.0, 0.0, 0.0)
    # 시작 시 모든 gap이 열려 있어 임팩터 y 강성이 0 → 특이행렬. 1 N/mm 지면 스프링으로 방지
    # (150 mm에서 0.15 kN — 총 하중 대비 무시 가능)
    ops.node(IMP_NODE + 1, 0.0, 0.0, 0.0)
    ops.fix(IMP_NODE + 1, 1, 1, 1, 1, 1, 1)
    ops.uniaxialMaterial("Elastic", IMP_ELE, 1.0)
    ops.element("zeroLength", IMP_ELE, IMP_NODE + 1, IMP_NODE, "-mat", IMP_ELE, "-dir", 2)
    eles = []
    tag = IMP_ELE
    for lev in levs:
        lev = int(lev)
        nodes = [mesh.node(OUTER, lev, k) for k in OUTER_BODY_PTS]
        ys = np.array([mesh.coords[n][1] for n in nodes])
        top = ys.max()
        for n, y in zip(nodes, ys):
            tag += 1
            ops.uniaxialMaterial("ElasticPPGap", tag, cfg.impactor_k, -1.0e15, min(-(top - y), -1e-9))
            # i = Outer 절점, j = 임팩터: 변형 = u_imp − u_node, 판이 내려오면 음수 = 압축
            ops.element("zeroLength", tag, n, IMP_NODE, "-mat", tag, "-dir", 2)
            eles.append((tag, lev))
    return dict(node=IMP_NODE, eles=eles, levels=[int(v) for v in levs])


def impactor_section_force(mesh: Mesh, imp: dict) -> np.ndarray:
    """임팩터 접촉력의 섹션별 분배 (N, 레버 룰로 양쪽 섹션에 분배)."""
    n = mesh.cfg.n_sub
    out = np.zeros(N_SECTIONS)
    for tag, lev in imp["eles"]:
        f = abs(ops.eleForce(tag)[1])       # gap은 압축만 받으므로 절점 i의 y 힘 크기 = 접촉력
        if f <= 0:
            continue
        s, j = divmod(lev, n)
        w = j / n
        out[s] += f * (1 - w)
        if w > 0:
            out[s + 1] += f * w
    return out


def run_static(cfg: Config, log=print) -> tuple[Mesh, History]:
    """정적 변위제어: 제어 절점(하중 구간 가운데 섹션의 Outer control_pt, impactor면 판)의
    -y 변위를 static_du씩 증가.
    참조 하중 총합 = 1 kN → load factor = 총 하중(kN)."""
    mesh, frac = prepare(cfg)
    lattice.build_model(mesh, with_mass=False)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    imp = None
    if cfg.load_mode == "impactor":
        imp = _build_impactor(mesh)
        ctrl = imp["node"]
        log(f"[impactor] 강체 평판, 레벨 {len(imp['levels'])}개(sec {cfg.load_secs[0]}-{cfg.load_secs[1]}), "
            f"접촉 절점 {len(imp['eles'])}개, k={cfg.impactor_k:.0e} N/mm")
    else:
        for nd, f in frac.items():
            ops.load(nd, 0.0, -1000.0 * f, 0.0, 0.0, 0.0, 0.0)
        ctrl = mesh.node(OUTER, mesh.level_of_sec(cfg.control_sec), cfg.control_pt)

    def setup_analysis():
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1.0e-6, 50)
        ops.algorithm("Newton")

    cm = None
    if cfg.contact:
        from .contact import ContactManager
        cm = ContactManager(mesh)
        a, _ = cm.update()
        log(f"[contact] 초기 쌍 {a}개 (k={cfg.contact_k:.0e} N/mm, act={cfg.contact_act} mm, "
            f"window=±{cfg.contact_window} mm, x 방향={'on' if cfg.contact_x else 'off'})")
    setup_analysis()

    nodes = _record_nodes(mesh, frac)
    hist = History(nodes=nodes)
    hist.times.append(0.0)
    hist.force.append(0.0)
    hist.U.append(_snapshot(nodes))
    n0 = cm.active_count() if cm else {1: 0, 2: 0}
    hist.extra["n_contact"] = [n0[2]]           # y 방향 접촉 중인 쌍
    hist.extra["n_contact_x"] = [n0[1]]         # x 방향 접촉 중인 쌍
    if imp is not None:
        hist.extra["imp_sec_force"] = [np.zeros(N_SECTIONS)]
    ymeta = elem_yield_meta(mesh)
    hist.extra["yield_meta"] = ymeta
    hist.extra["yield_r"] = [np.zeros(len(ymeta["pid"]), dtype=np.float32)]

    target, du = cfg.static_target, cfg.static_du
    d, t_wall = 0.0, time.time()
    algos = ["Newton", "NewtonLineSearch", "KrylovNewton", "ModifiedNewton"]
    while d < target - 1e-9:
        step = min(du, target - d)
        ok, sub = -1, step
        while ok != 0 and sub >= du / 64:
            for alg in algos:
                ops.algorithm(alg)
                ops.integrator("DisplacementControl", ctrl, 2, -sub)
                ops.analysis("Static")
                ok = ops.analyze(1)
                if ok == 0:
                    break
            if ok != 0:
                sub /= 2
        if ok != 0:
            log(f"[static] 수렴 실패 d={d:.2f} mm")
            hist.extra["failed_at"] = d
            break
        d += sub
        F = ops.getLoadFactor(1) * 1000.0
        hist.times.append(d)          # 정적 해석에서 times = 제어 변위(mm)
        hist.force.append(F)
        hist.U.append(_snapshot(nodes))
        n_act = {1: 0, 2: 0}
        if cm is not None:
            n_new, n_del = cm.update()
            if n_new or n_del:          # 요소 추가/제거 후에는 해석 객체를 다시 만든다
                ops.wipeAnalysis()
                setup_analysis()
            n_act = cm.active_count()
        hist.extra["n_contact"].append(n_act[2])
        hist.extra["n_contact_x"].append(n_act[1])
        r = scan_yield(ymeta)
        hist.extra["yield_r"].append(r)
        if imp is not None:
            hist.extra["imp_sec_force"].append(impactor_section_force(mesh, imp))
        if len(hist.times) % 10 == 0:
            yl = " ".join(f"{mesh.parts[p].name}:{100 * np.mean(r[ymeta['pid'] == p] >= 1):.0f}%"
                          for p in range(len(mesh.parts)))
            log(f"  d={d:7.2f} mm  F={F / 1e3:8.2f} kN  contact y={n_act[2]:3d} x={n_act[1]:3d}"
                f"{f' (pairs {len(cm.pairs)})' if cm else ''}  yield[{yl}]  "
                f"wall {time.time() - t_wall:.0f}s")
    hist.extra.update(wall=time.time() - t_wall, control_node=ctrl)
    if cm is not None:
        hist.extra.update(contact_added=cm.n_added, contact_removed=cm.n_removed)
    return mesh, hist
