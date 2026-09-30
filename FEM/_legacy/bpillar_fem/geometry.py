"""격자 메쉬 생성 (idea_v1.md §3): 절점, 요소 연결성, 등가 strip 단면, 집중질량.

절점 번호: part*100000 + lev*100 + pt  (lev = sec*n_sub + j)
플랜지 쌍은 union-find로 한 절점에 병합한다(허브 우선순위 Plate > Outer > Inner > Patch1 > Patch2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import N_SECTIONS, Config
from .io_csv import FLANGE_PAIR_SPEC, Part

HUB_PRIORITY = {1: 0, 0: 1, 2: 2, 3: 3, 4: 4}


def raw_tag(pid: int, lev: int, pt: int) -> int:
    return pid * 100000 + lev * 100 + pt


@dataclass
class Element:
    kind: str               # 'circ' | 'long' | 'diag'
    pid: int
    n1: int
    n2: int
    b: float                # strip 폭 (diag는 면적으로 사용 안 함)
    t: float
    vecxz: tuple            # 판 법선 (geomTransf vecxz)
    length: float
    lev: int
    pt: int
    area: float = 0.0       # diag 전용


@dataclass
class Mesh:
    cfg: Config
    parts: list
    n_levels: int                               # 레벨 수 (lev = 0..n_levels-1)
    coords: dict = field(default_factory=dict)  # rep tag -> (x, y, z)
    rep: dict = field(default_factory=dict)     # raw tag -> rep tag
    elements: list = field(default_factory=list)
    mass: dict = field(default_factory=dict)    # rep tag -> 병진 질량
    rot_inertia: dict = field(default_factory=dict)
    dt: float = 0.0
    part_last_lev: dict = field(default_factory=dict)

    def node(self, pid: int, lev: int, pt: int) -> int:
        return self.rep[raw_tag(pid, lev, pt)]

    def has(self, pid: int, lev: int) -> bool:
        return lev <= self.part_last_lev[pid]

    def level_of_sec(self, sec: int) -> int:
        return sec * self.cfg.n_sub

    def section_nodes(self, sec: int) -> list[int]:
        """섹션 sec 레벨의 모든 파트 절점(병합 후 중복 제거, 순서 유지)."""
        lev = self.level_of_sec(sec)
        seen, out = set(), []
        for p in self.parts:
            if not self.has(p.pid, lev):
                continue
            for k in range(1, p.n_pt + 1):
                n = self.node(p.pid, lev, k)
                if n not in seen:
                    seen.add(n)
                    out.append(n)
        return out


def _level_xy(part: Part, lev: int, n_sub: int) -> np.ndarray:
    s, j = divmod(lev, n_sub)
    if j == 0:
        return part.xy[s]
    f = j / n_sub
    return (1.0 - f) * part.xy[s] + f * part.xy[s + 1]


def _seg_normals(xy: np.ndarray) -> np.ndarray:
    """세그먼트 k–k+1의 면내 단위 법선 (n_pt-1, 2)."""
    d = np.diff(xy, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    return nrm / np.linalg.norm(nrm, axis=1, keepdims=True)


def _point_normals(xy: np.ndarray) -> np.ndarray:
    """각 point의 판 법선 = 인접 세그먼트 법선 평균 (§3.2)."""
    sn = _seg_normals(xy)
    pn = np.zeros((xy.shape[0], 2))
    pn[:-1] += sn
    pn[1:] += sn
    bad = np.linalg.norm(pn, axis=1) < 1e-9       # 180° 꺾임 방어
    pn[bad] = np.vstack([sn, sn[-1:]])[bad]
    return pn / np.linalg.norm(pn, axis=1, keepdims=True)


def _tributary(xy: np.ndarray) -> np.ndarray:
    """원주 방향 기여 폭 b_L (양 끝 point는 반쪽)."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    b = np.zeros(xy.shape[0])
    b[:-1] += 0.5 * seg
    b[1:] += 0.5 * seg
    return b


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        key = lambda r: (HUB_PRIORITY[r // 100000], r)
        lo, hi = sorted([ra, rb], key=key)
        self.parent[hi] = lo


def build_mesh(parts: list[Part], cfg: Config) -> Mesh:
    n = cfg.n_sub
    dzl = cfg.dz / n
    n_levels = (N_SECTIONS - 1) * n + 1
    mesh = Mesh(cfg=cfg, parts=parts, n_levels=n_levels)
    mesh.part_last_lev = {p.pid: p.last_sec * n for p in parts}
    by_pid = {p.pid: p for p in parts}

    # ── 원시 절점 좌표 ──
    raw_xyz, lev_xy = {}, {}
    for p in parts:
        for lev in range(mesh.part_last_lev[p.pid] + 1):
            xy = _level_xy(p, lev, n)
            lev_xy[(p.pid, lev)] = xy
            for k in range(p.n_pt):
                raw_xyz[raw_tag(p.pid, lev, k + 1)] = (xy[k, 0], xy[k, 1], lev * dzl)

    # ── 플랜지 병합 (§3.4) ──
    uf = _UnionFind()
    for pa, ka, pb, kb in FLANGE_PAIR_SPEC:
        last = min(mesh.part_last_lev[pa], mesh.part_last_lev[pb])
        for lev in range(last + 1):
            if cfg.tie_levels == "sections" and lev % n:
                continue
            uf.union(raw_tag(pa, lev, ka), raw_tag(pb, lev, kb))
    for tag in raw_xyz:
        r = uf.find(tag)
        mesh.rep[tag] = r
        mesh.coords[r] = raw_xyz[r]

    def elem_len(a, b):
        return float(np.linalg.norm(np.subtract(mesh.coords[a], mesh.coords[b])))

    # ── 요소 ──
    for p in parts:
        last = mesh.part_last_lev[p.pid]
        for lev in range(last + 1):
            xy = lev_xy[(p.pid, lev)]
            # 원주 방향 strip: 폭 b_C = dz_sub (양 끝 레벨은 반쪽)
            b_c = dzl * (0.5 if lev in (0, last) else 1.0)
            sn = _seg_normals(xy)
            for k in range(1, p.n_pt):
                a, b = mesh.node(p.pid, lev, k), mesh.node(p.pid, lev, k + 1)
                if a == b:
                    continue
                mesh.elements.append(Element("circ", p.pid, a, b, b_c, p.t,
                                             (sn[k - 1, 0], sn[k - 1, 1], 0.0),
                                             elem_len(a, b), lev, k))
            if lev == last:
                continue
            # 종방향 strip: 폭 b_L = 두 레벨 기여 폭 평균
            xy2 = lev_xy[(p.pid, lev + 1)]
            bl = 0.5 * (_tributary(xy) + _tributary(xy2))
            pn = _point_normals(xy)
            for k in range(1, p.n_pt + 1):
                a, b = mesh.node(p.pid, lev, k), mesh.node(p.pid, lev + 1, k)
                mesh.elements.append(Element("long", p.pid, a, b, float(bl[k - 1]), p.t,
                                             (pn[k - 1, 0], pn[k - 1, 1], 0.0),
                                             elem_len(a, b), lev, k))
            if cfg.use_diagonals and cfg.diag_area_ratio > 0:
                seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                for k in range(1, p.n_pt):
                    area = cfg.diag_area_ratio * p.t * 0.5 * (seg[k - 1] + dzl)
                    for a, b in ((mesh.node(p.pid, lev, k), mesh.node(p.pid, lev + 1, k + 1)),
                                 (mesh.node(p.pid, lev, k + 1), mesh.node(p.pid, lev + 1, k))):
                        if a != b:
                            mesh.elements.append(Element("diag", p.pid, a, b, 0.0, p.t,
                                                         (0.0, 0.0, 0.0), elem_len(a, b),
                                                         lev, k, area=area))

    _lumped_mass_and_dt(mesh, by_pid, lev_xy)
    return mesh


def _lumped_mass_and_dt(mesh: Mesh, by_pid: dict, lev_xy: dict) -> None:
    from .lattice import inplane_inertia

    """절점 집중질량 m = rho*t*b_L*b_C (§3.3), 안정 시간간격, 회전관성.

    dt: 절점별 2*sqrt(m / Σk) (k = max(EA/L, 12EI_max/L^3), Gershgorin 상한)의 최소값 × cfl.
    회전관성 J: 회전 모드가 dt를 제한하지 않도록 J = 2 * Σk_rot * (dt/cfl)^2 / 4 로 둔다
    (회전 질량 스케일링 — 저주파 응답에는 영향이 작다).
    """
    cfg = mesh.cfg
    n = cfg.n_sub
    dzl = cfg.dz / n
    E = cfg.E
    mass = {}
    for p in mesh.parts:
        last = mesh.part_last_lev[p.pid]
        for lev in range(last + 1):
            b_c = dzl * (0.5 if lev in (0, last) else 1.0)
            bl = _tributary(lev_xy[(p.pid, lev)])
            for k in range(1, p.n_pt + 1):
                r = mesh.node(p.pid, lev, k)
                mass[r] = mass.get(r, 0.0) + cfg.rho * p.t * bl[k - 1] * b_c * cfg.mass_scale

    k_tr, k_rot = {}, {}
    for e in mesh.elements:
        L = e.length
        if e.kind == "diag":
            kt, kr = E * e.area / L, 0.0
        else:
            i_max = max(e.b * e.t ** 3 / 12.0, inplane_inertia(e.b, e.t, L, cfg))
            kt = max(E * e.b * e.t / L, 12.0 * E * i_max / L ** 3)
            kr = 4.0 * E * i_max / L
        for nd in (e.n1, e.n2):
            k_tr[nd] = k_tr.get(nd, 0.0) + kt
            k_rot[nd] = k_rot.get(nd, 0.0) + kr

    dt_nodes = [2.0 * np.sqrt(mass[nd] / k_tr[nd]) for nd in mass if k_tr.get(nd, 0) > 0]
    dt = cfg.dt if cfg.dt else cfg.cfl * float(min(dt_nodes))
    mesh.dt = dt
    mesh.mass = mass
    t_ref = dt / cfg.cfl
    mesh.rot_inertia = {nd: max(2.0 * k_rot.get(nd, 0.0) * t_ref ** 2 / 4.0, 1e-3 * mass[nd])
                        for nd in mass}


def summary(mesh: Mesh) -> dict:
    kinds = {}
    for e in mesh.elements:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    return dict(nodes=len(mesh.coords), elements=len(mesh.elements), **kinds,
                merged=len(mesh.rep) - len(mesh.coords),
                mass_kg=sum(mesh.mass.values()) * 1000.0, dt=mesh.dt,
                min_len=min(e.length for e in mesh.elements))
