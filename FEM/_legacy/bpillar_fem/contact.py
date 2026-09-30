"""적응형 절점-절점 gap 접촉 (idea_v1.md §7.4, §7.6).

OpenSees에는 이 격자에 쓸 범용 면 접촉이 없으므로, 같은 레벨의 절점 쌍에 전역 x 또는 y 방향
압축 전용 gap 요소(zeroLength + ElasticPPGap)를 둔다.

y 방향 (위–아래 적층, §7.4)
  층 순서(위→아래): Outer → Inner → Patch1 → Plate (Patch2는 전 절점이 Plate와 병합되어 제외)
  위 파트 절점마다, 아래쪽 파트 전체 중 x가 창 안에 있고 현재 아래에 있는 절점 가운데
  가장 높은 절점(=바로 아래 표면)을 짝으로 고른다.
x 방향 (벽–벽, §7.6)
  모든 파트 절점 u에 대해, |Δy| ≤ 창 이고 오른쪽(x_v > x_u)에 있는 가장 가까운 절점 v와 짝짓는다.
  대상: 다른 파트 절점, 또는 같은 파트에서 3칸 이상 떨어진 절점(접힌 벽의 자기 접촉).
  i = 왼쪽, j = 오른쪽: 변형 = u_j − u_i (가까워지면 음수 = 압축).

공통
- 접촉 거리 c = (t_a + t_b)/2 (판 중심선 간격). 해석 도중 추가한 zeroLength는 추가 시점을
  변형 0으로 삼으므로 gap = c − (현재 거리). 이미 c보다 가까우면 현재 거리의 90%에서 접촉.
- 쌍 갱신은 정적 스텝마다: 현재 간격 − c < act 인 새 쌍을 추가하고, 벌어져 있으면서(힘 0)
  직교 방향으로 창의 2배 이상 어긋난 쌍은 제거한다. 압축 중인 쌍은 제거하지 않는다
  (압축 상태에서 제거하면 OpenSees 정적 해석이 수렴하지 않음 — 시험으로 확인).
- 마찰 없음. 법선은 전역 x 또는 y로 고정(경사 벽은 두 방향 성분으로 근사).
"""
from __future__ import annotations

import numpy as np
import openseespy.opensees as ops

from .geometry import Mesh
from .io_csv import INNER, OUTER, PATCH1, PLATE

LAYER_ORDER = [OUTER, INNER, PATCH1, PLATE]     # 위 → 아래 (Patch2는 전 절점이 Plate와 병합)
SELF_MIN_GAP = 3                # 같은 파트 자기 접촉: point 번호 차이 최소값
ELE_BASE = 5_000_000
MAT_BASE = 5_000_000


class ContactManager:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        cfg = mesh.cfg
        self.k = cfg.contact_k
        self.act = cfg.contact_act
        self.win = cfg.contact_window
        self.use_x = cfg.contact_x
        self.t = {p.pid: p.t for p in mesh.parts}
        self.pairs = {}          # (a, b, dir) -> (ele_tag, c)   a = 아래/왼쪽, b = 위/오른쪽
        self.next_tag = 0
        self.n_added = self.n_removed = 0

        # 레벨별 절점: 병합으로 여러 파트가 공유하는 절점은 제외(플랜지 — 이미 붙어 있음)
        owner = {}
        for p in mesh.parts:
            for lev in range(mesh.part_last_lev[p.pid] + 1):
                for k in range(1, p.n_pt + 1):
                    owner.setdefault(mesh.node(p.pid, lev, k), set()).add(p.pid)
        self.levels = []
        for lev in range(mesh.n_levels):
            nodes, pids, pts, nb0, nb1 = [], [], [], [], []
            for pid in LAYER_ORDER:
                if not mesh.has(pid, lev):
                    continue
                n_pt = mesh.parts[pid].n_pt
                for k in range(1, n_pt + 1):
                    n = mesh.node(pid, lev, k)
                    if len(owner[n]) == 1:
                        nodes.append(n)
                        pids.append(pid)
                        pts.append(k)
                        nb0.append(mesh.node(pid, lev, max(k - 1, 1)))
                        nb1.append(mesh.node(pid, lev, min(k + 1, n_pt)))
            self.levels.append(dict(nodes=np.array(nodes, dtype=int), pids=np.array(pids),
                                    pts=np.array(pts), nb0=np.array(nb0, dtype=int),
                                    nb1=np.array(nb1, dtype=int)))
        self.all_nodes = np.array(sorted(owner), dtype=int)
        self.ref = {n: np.asarray(mesh.coords[n][:2]) for n in self.all_nodes}

    def _current_xy(self) -> dict:
        return {n: self.ref[n] + np.asarray(ops.nodeDisp(int(n))[:2]) for n in self.all_nodes}

    def update(self) -> tuple[int, int]:
        """쌍 추가/제거. (추가 수, 제거 수)."""
        xy = self._current_xy()
        added = removed = 0

        for key, (tag, c) in list(self.pairs.items()):
            a, b, d = key
            along = xy[b][d - 1] - xy[a][d - 1]              # 법선 방향 간격
            across = abs(xy[b][2 - d] - xy[a][2 - d])        # 직교 방향 어긋남
            if across > 2.0 * self.win and along - c > 0.05 * self.act:
                ops.remove("element", tag)
                del self.pairs[key]
                removed += 1

        for L in self.levels:
            nodes = L["nodes"]
            if not len(nodes):
                continue
            P = np.array([xy[n] for n in nodes])
            added += self._add_y(nodes, L["pids"], P)
            if self.use_x:
                tan = (np.array([xy[n] for n in L["nb1"]]) - np.array([xy[n] for n in L["nb0"]]))
                steep = np.abs(tan[:, 1]) > np.abs(tan[:, 0])       # 국부 접선이 수직에 가까움 = 벽
                added += self._add_x(nodes, L["pids"], L["pts"], P, steep)
        self.n_added += added
        self.n_removed += removed
        return added, removed

    def _add_y(self, nodes, pids, P) -> int:
        n_add = 0
        for i, pu in enumerate(LAYER_ORDER[:-1]):
            up = np.flatnonzero(pids == pu)
            lo = np.flatnonzero(np.isin(pids, LAYER_ORDER[i + 1:]))
            if not len(up) or not len(lo):
                continue
            for u in up:
                m = (np.abs(P[lo, 0] - P[u, 0]) <= self.win) & (P[lo, 1] < P[u, 1])
                if not m.any():
                    continue
                j = lo[np.flatnonzero(m)[np.argmax(P[lo[m], 1])]]
                n_add += self._try_add(int(nodes[j]), int(nodes[u]), 2,
                                       self.t[int(pids[j])], self.t[int(pids[u])],
                                       P[u, 1] - P[j, 1])
        return n_add

    def _add_x(self, nodes, pids, pts, P, steep) -> int:
        """벽–벽: 두 절점 모두 가파른 벽 위에 있을 때만 x 쌍을 만든다
        (위아래로 쌓인 절점이 옆으로 미끄러질 때 x 접촉이 잘못 걸리는 것 방지)."""
        n_add = 0
        dx = P[None, :, 0] - P[:, None, 0]                   # dx[u, v] = x_v − x_u
        dy = np.abs(P[None, :, 1] - P[:, None, 1])
        same = pids[:, None] == pids[None, :]
        far = np.abs(pts[:, None] - pts[None, :]) >= SELF_MIN_GAP
        ok = (dx > 0) & (dy <= self.win) & (~same | far) & steep[:, None] & steep[None, :]
        for u in range(len(nodes)):
            cand = np.flatnonzero(ok[u])
            if not len(cand):
                continue
            v = cand[np.argmin(dx[u, cand])]
            n_add += self._try_add(int(nodes[u]), int(nodes[v]), 1,
                                   self.t[int(pids[u])], self.t[int(pids[v])], dx[u, v])
        return n_add

    def _try_add(self, a: int, b: int, d: int, ta: float, tb: float, dist: float) -> int:
        """a = 아래/왼쪽, b = 위/오른쪽, d = 1(x) | 2(y), dist = 현재 법선 방향 거리(>0)."""
        if (a, b, d) in self.pairs:
            return 0
        c = 0.5 * (ta + tb)
        if dist - c > self.act:
            return 0
        c_eff = min(c, 0.9 * dist)
        self.next_tag += 1
        tag = ELE_BASE + self.next_tag
        mat = MAT_BASE + self.next_tag
        ops.uniaxialMaterial("ElasticPPGap", mat, self.k, -1.0e15, c_eff - dist)
        ops.element("zeroLength", tag, a, b, "-mat", mat, "-dir", d)
        self.pairs[(a, b, d)] = (tag, c_eff)
        return 1

    def active_count(self) -> dict:
        """현재 압축(접촉) 중인 쌍 수, 방향별."""
        n = {1: 0, 2: 0}
        for (a, b, d), (_, c) in self.pairs.items():
            gap = ((self.ref[b][d - 1] + ops.nodeDisp(int(b), d))
                   - (self.ref[a][d - 1] + ops.nodeDisp(int(a), d)))
            if gap <= c + 1e-6:
                n[d] += 1
        return n
