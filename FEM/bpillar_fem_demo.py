"""B-pillar 단면 CSV → 격자 FEM(OpenSeesPy) → 섹션별 변형량 CSV (데모, 단일 파일).

사용 (conda env fem — Python 3.12 + openseespy):
    python bpillar_fem_demo.py                                   # 기본 설계 CSV (Plate 침입 50 mm까지)
    python bpillar_fem_demo.py --quick                           # 거친 격자로 빠르게 (Plate 30 mm까지)
    python bpillar_fem_demo.py --csv 설계.csv --plate-target 80 --forces 100 200 300
    python bpillar_fem_demo.py --quick --force-target 1050       # 총 하중 1050 kN까지

하중 증가: Plate(가장 안쪽 = 탑승자 쪽 판)의 최대 침입량이 --plate-target에 닿을 때까지 누른다
          (제어 변위 상한 --max-disp). 하중 수준은 지정이 없으면 도달한 최대 하중까지 6단계.

출력 (FEM/results/, 파일 이름 앞부분 = CSV 파일 이름, --quick이면 _quick 추가)
    <name>_deformation.csv   하중 수준(F0) × 섹션(0–16)별 변형량 (mm)
                             plate_intrusion  : Plate 절점의 −y 최대 이동량 — ★ 최대 변형량 판단 기준
                             plate_distortion : Plate 자체 찌그러짐 (강체운동 제외)
                             outer_intrusion  : Outer(충돌면) 절점의 −y 최대 이동량 (참고)
                             d_crush          : Outer 윗면 – Plate 사이 거리 감소 = 단면 압궤 (참고)
                             status           : stable(1차 한계하중 이하) | snap_through(한계하중을 넘어
                                                다음 평형) | not_reached
    <name>_shapes.csv        하중 수준별 변형된 섹션 단면 좌표 (sec, part, point_idx, 변형 전/후 x, y)
    <name>_shapes_F*kN.png   하중 수준별 17개 섹션 단면 그림 (점선 = 변형 전)
    <name>_curve.csv         제어 변위 – 총 하중 – Plate 최대 침입량
    <name>_opensees.log      OpenSees 경고

모델 (단위: mm, N, MPa)
    - CSV의 5개 파트 × 17섹션 폴리라인 point = 절점. 섹션 사이는 N_SUB개로 나눠 보간 절점 추가.
    - 요소: 섹션 안(원주) 선분과 섹션 사이(같은 point 번호끼리) 선분 → 판 strip 파이버 보.
      strip 단면 = 폭 b × 두께 t, 두께 방향 6층(판 굽힘 소성), 면내 I = b·t·L²/(12(1+ν))
      (이 값이면 격자의 면내 전단 강성이 판의 G·t와 같아진다).
    - 파트 접합: 플랜지 point 쌍을 한 절점으로 병합.  경계: sec 0, sec 16 전 절점 고정.
    - 하중: sec 5–11의 Outer 윗면에 −y 균일 압력. 정적 변위제어로 누른 뒤
      곡선에서 하중 수준별 상태를 읽는다.
    - 접촉: 위 파트 절점과 바로 아래 파트 절점 사이 y 방향 압축 전용 gap (절점-절점, 간소화).
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import openseespy.opensees as ops
import pandas as pd

# ─────────────────────────────── 입력 가정 ───────────────────────────────
DEFAULT_CSV = (Path(__file__).resolve().parent.parent / "01. CGNN" / "reports" / "v1_21_0_x1.3"
               / "final_section_v1_21_0_x1.3.csv")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
HEIGHT = 1051.7                     # sec 0(sill) ~ sec 16(roof rail) 거리
N_SEC = 17
E, NU, HARDENING = 210000.0, 0.3, 0.01
N_THICK = 6                         # strip 두께 방향 파이버 층수

PART_NAMES = ["Outer", "Plate", "Inner", "Patch1", "Patch2"]
PART_FY = [1470.0, 980.0, 1470.0, 980.0, 440.0]
OUTER, PLATE, INNER, PATCH1, PATCH2 = range(5)
# 플랜지 접합 (part_a, point_a, part_b, point_b) — CGNN FLANGE_PAIR_SPEC
FLANGE_PAIRS = [
    (OUTER, 1, PLATE, 1), (OUTER, 2, PLATE, 2), (OUTER, 3, PLATE, 3),
    (OUTER, 28, PLATE, 28), (OUTER, 29, PLATE, 29), (OUTER, 30, PLATE, 30),
    (PLATE, 7, INNER, 1), (PLATE, 8, INNER, 2), (PLATE, 9, INNER, 3),
    (PLATE, 22, INNER, 16), (PLATE, 23, INNER, 17), (PLATE, 24, INNER, 18),
    (PLATE, 10, PATCH1, 1), (PLATE, 21, PATCH1, 12),
    (PLATE, 22, PATCH2, 1), (PLATE, 23, PATCH2, 2), (PLATE, 24, PATCH2, 3),
]
LOAD_SECS = (5, 11)                 # 하중 구간 섹션
CONTROL_SEC, CONTROL_PT = 8, 15     # 정적 변위제어 기준 절점 (Outer)
CROWN_PTS = range(13, 19)           # d_crush 기준: Outer 윗면 point
PLATE_PTS = range(13, 19)           #               그 아래 Plate point
CONTACT_K = 2.0e5                   # 접촉 penalty (N/mm)
CONTACT_WIN = 6.0                   # 접촉 짝 찾기 x 창 (mm)
CONTACT_ACT = 3.0                   # 간격 − 접촉거리 < 이 값이면 접촉 요소 생성 (mm)


# ─────────────────────────────── 1. CSV ───────────────────────────────
def read_parts(path) -> list[dict]:
    """CSV → 파트 목록. 파트 블록 끝의 구분 행(point_idx = 0)에 두께가 들어 있다."""
    df = pd.read_csv(path)
    df["block"] = (df["point_idx"] == 0).cumsum().shift(fill_value=0)
    parts = []
    for pid, blk in df.groupby("block"):
        data = blk[blk["point_idx"] > 0].sort_values(["floor_idx", "point_idx"])
        n_sec = data["floor_idx"].nunique()
        xy = data[["x_mm", "y_mm"]].to_numpy().reshape(n_sec, -1, 2)
        parts.append(dict(name=PART_NAMES[pid], t=float(blk.loc[blk["point_idx"] == 0, "t_mm"].iloc[0]),
                          fy=PART_FY[pid], xy=xy))
    return parts


# ─────────────────────────────── 2. 메쉬 ───────────────────────────────
def build_mesh(parts, n_sub, dz) -> dict:
    """절점(병합 후), 요소, 섹션별 절점을 만든다. 절점 번호 = pid*100000 + lev*100 + point."""
    dzl = dz / n_sub
    last = [(len(p["xy"]) - 1) * n_sub for p in parts]          # 파트별 마지막 레벨

    def lev_xy(pid, lev):                                        # 섹션 사이 레벨은 선형 보간
        s, j = divmod(lev, n_sub)
        xy = parts[pid]["xy"]
        return xy[s] if j == 0 else (1 - j / n_sub) * xy[s] + (j / n_sub) * xy[s + 1]

    raw = {}
    for pid, p in enumerate(parts):
        for lev in range(last[pid] + 1):
            for k, (x, y) in enumerate(lev_xy(pid, lev), start=1):
                raw[pid * 100000 + lev * 100 + k] = (x, y, lev * dzl)

    # 플랜지 병합: union-find, 대표는 Plate 쪽
    parent = {t: t for t in raw}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t
    hub = {PLATE: 0, OUTER: 1, INNER: 2, PATCH1: 3, PATCH2: 4}
    for pa, ka, pb, kb in FLANGE_PAIRS:
        for lev in range(min(last[pa], last[pb]) + 1):
            ra, rb = find(pa * 100000 + lev * 100 + ka), find(pb * 100000 + lev * 100 + kb)
            if ra != rb:
                lo, hi = sorted([ra, rb], key=lambda r: (hub[r // 100000], r))
                parent[hi] = lo
    node = {t: find(t) for t in raw}                              # 원래 번호 → 대표 절점
    coords = {r: raw[r] for r in set(node.values())}

    def n(pid, lev, k):
        return node[pid * 100000 + lev * 100 + k]

    def tributary(xy):                                           # 원주 방향 기여 폭
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        return np.r_[seg, 0] / 2 + np.r_[0, seg] / 2

    def normals(xy):                                             # point별 판 법선 (면내)
        d = np.diff(xy, axis=0)
        sn = np.stack([-d[:, 1], d[:, 0]], 1) / np.linalg.norm(d, axis=1)[:, None]
        pn = np.r_[sn, sn[-1:]] + np.r_[sn[:1], sn]
        return sn, pn / np.linalg.norm(pn, axis=1)[:, None]

    elems = []                                                   # (i, j, b, t, fy, normal)
    elem_info = []                                               # 요소별 (파트 id, 'circ' | 'long') — 시각화용
    for pid, p in enumerate(parts):
        for lev in range(last[pid] + 1):
            xy = lev_xy(pid, lev)
            sn, pn = normals(xy)
            b_c = dzl * (0.5 if lev in (0, last[pid]) else 1.0)
            for k in range(1, len(xy)):                          # 원주 방향
                i, j = n(pid, lev, k), n(pid, lev, k + 1)
                if i != j:
                    elems.append((i, j, b_c, p["t"], p["fy"], sn[k - 1]))
                    elem_info.append((pid, "circ"))
            if lev < last[pid]:                                  # 종방향 (같은 point끼리)
                b_l = 0.5 * (tributary(xy) + tributary(lev_xy(pid, lev + 1)))
                for k in range(1, len(xy) + 1):
                    elems.append((n(pid, lev, k), n(pid, lev + 1, k), b_l[k - 1], p["t"], p["fy"], pn[k - 1]))
                    elem_info.append((pid, "long"))

    # 하중: 하중 구간 모든 레벨의 Outer 윗면(|n_y| ≥ 0.5, 플랜지 제외)에 x 투영 폭 비율
    load = {}
    levs = np.arange(LOAD_SECS[0] * n_sub, LOAD_SECS[1] * n_sub + 1)
    w_lev = np.ones(len(levs))
    w_lev[[0, -1]] = 0.5
    for lev, wl in zip(levs, w_lev / w_lev.sum()):
        xy = lev_xy(OUTER, lev)
        _, pn = normals(xy)
        dx = np.abs(np.diff(xy[:, 0]))
        bx = np.r_[dx, 0] / 2 + np.r_[0, dx] / 2
        pts = [k for k in range(6, 26) if abs(pn[k - 1, 1]) >= 0.5]
        for k in pts:
            load[n(OUTER, lev, k)] = load.get(n(OUTER, lev, k), 0.0) + wl * bx[k - 1] / bx[np.array(pts) - 1].sum()

    # 섹션별 절점 (파트별 point 순서)
    sections = {s: {pid: [n(pid, s * n_sub, k) for k in range(1, len(p["xy"][0]) + 1)]
                    for pid, p in enumerate(parts) if s * n_sub <= last[pid]} for s in range(N_SEC)}
    # 접촉 후보: 다른 파트와 공유하지 않는(병합되지 않은) 절점, 레벨별
    shared = {r for r, cnt in Counter(node.values()).items() if cnt > 1}
    contact_levels = [{pid: [n(pid, lev, k) for k in range(1, len(p["xy"][0]) + 1)
                             if n(pid, lev, k) not in shared]
                       for pid, p in enumerate(parts) if lev <= last[pid]}
                      for lev in range((N_SEC - 1) * n_sub + 1)]
    return dict(coords=coords, elems=elems, elem_info=elem_info, load=load, sections=sections,
                contact_levels=contact_levels, thick=[p["t"] for p in parts], shared=shared,
                control=n(OUTER, CONTROL_SEC * n_sub, CONTROL_PT))


# ─────────────────────────────── 3. OpenSees 모델 ───────────────────────────────
def build_model(mesh) -> None:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for tag, (x, y, z) in mesh["coords"].items():
        ops.node(tag, x, y, z)
    G = E / (2 * (1 + NU))
    mats = {}
    for tag, (i, j, b, t, fy, nrm) in enumerate(mesh["elems"], start=1):
        if fy not in mats:
            mats[fy] = len(mats) + 1
            ops.uniaxialMaterial("Steel02", mats[fy], fy, E, HARDENING, 20.0, 0.925, 0.15)
        L = np.linalg.norm(np.subtract(mesh["coords"][i], mesh["coords"][j]))
        y0 = np.sqrt(b * t * L * L / (12 * (1 + NU)) / (b * t))        # 면내 I = b t L²/(12(1+ν))
        ops.geomTransf("Corotational", tag, float(nrm[0]), float(nrm[1]), 0.0)
        ops.section("Fiber", tag, "-GJ", G * b * t ** 3 / 3)
        for yf in (-y0, y0):
            for zf in -t / 2 + (np.arange(N_THICK) + 0.5) * t / N_THICK:
                ops.fiber(yf, zf, b * t / (2 * N_THICK), mats[fy])
        ops.beamIntegration("Lobatto", tag, tag, 3)
        ops.element("dispBeamColumn", tag, i, j, tag, tag)
    ends = {nd for s in (0, N_SEC - 1) for nodes in mesh["sections"][s].values() for nd in nodes}
    for nd in ends:                                                    # 양단 전 절점 고정
        ops.fix(nd, 1, 1, 1, 1, 1, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nd, f in mesh["load"].items():                                 # 참조 하중 합계 1 kN
        ops.load(nd, 0.0, -1000.0 * f, 0.0, 0.0, 0.0, 0.0)


# ─────────────────────────────── 4. 접촉 (y 방향) ───────────────────────────────
class Contact:
    """위 파트 절점 ↔ 바로 아래 파트 절점 y 방향 gap. 층 순서 Outer → Inner → Patch1 → Plate."""
    ORDER = [OUTER, INNER, PATCH1, PLATE]

    def __init__(self, mesh):
        self.mesh, self.pairs, self.tag = mesh, {}, 5_000_000

    def update(self) -> bool:
        """쌍 추가/제거. 바뀐 것이 있으면 True (해석 객체를 다시 만들어야 함)."""
        cur = {nd: np.add(self.mesh["coords"][nd][:2], ops.nodeDisp(nd)[:2]) for nd in self.mesh["coords"]}
        changed = False
        for (u, l), (tag, c) in list(self.pairs.items()):     # 벌어지고 옆으로 어긋난 쌍 제거
            if abs(cur[u][0] - cur[l][0]) > 2 * CONTACT_WIN and cur[u][1] - cur[l][1] > c + 0.15:
                ops.remove("element", tag)
                del self.pairs[(u, l)]
                changed = True
        for lev in self.mesh["contact_levels"]:
            for i, pu in enumerate(self.ORDER[:-1]):
                lower = [(nd, pl) for pl in self.ORDER[i + 1:] for nd in lev.get(pl, [])]
                for u in lev.get(pu, []):
                    ux, uy = cur[u]
                    cand = [(cur[nd][1], nd, pl) for nd, pl in lower
                            if abs(cur[nd][0] - ux) <= CONTACT_WIN and cur[nd][1] < uy]
                    if not cand:
                        continue
                    ly, lo, pl = max(cand)                             # 바로 아래 표면
                    c = 0.5 * (self.mesh["thick"][pu] + self.mesh["thick"][pl])
                    if (u, lo) in self.pairs or uy - ly - c > CONTACT_ACT:
                        continue
                    c = min(c, 0.9 * (uy - ly))
                    self.tag += 1
                    # 추가 시점 상태가 변형 0 → gap = 접촉거리 − 현재 거리 (압축 음수)
                    ops.uniaxialMaterial("ElasticPPGap", self.tag, CONTACT_K, -1e15, c - (uy - ly))
                    ops.element("zeroLength", self.tag, lo, u, "-mat", self.tag, "-dir", 2)
                    self.pairs[(u, lo)] = (self.tag, c)
                    changed = True
        return changed


# ─────────────────────────────── 5. 정적 해석 ───────────────────────────────
def run_static(mesh, plate_target, max_disp, du, force_target=None, log=print):
    """제어 절점을 −y로 du씩 누른다. 멈추는 조건: force_target(N)이 있으면 총 하중이 그 값에,
    없으면 Plate 최대 침입량이 plate_target에 닿을 때. 어느 쪽이든 제어 변위 max_disp가 상한.
    반환: 제어 변위, 총 하중(N), 섹션 절점 변위 이력, 절점 → 열 번호."""
    record = sorted({nd for sec in mesh["sections"].values() for nodes in sec.values() for nd in nodes})
    col = {nd: i for i, nd in enumerate(record)}
    plate_idx = sorted({col[nd] for sec in mesh["sections"].values() for nd in sec[PLATE]})
    contact = Contact(mesh)
    contact.update()

    def setup():
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1e-6, 50)

    setup()
    disp, force = [0.0], [0.0]
    U = [np.zeros((len(record), 2))]
    d, plate_max, t0 = 0.0, 0.0, time.time()

    def reached():
        return force[-1] >= force_target if force_target else plate_max >= plate_target

    while not reached() and d < max_disp - 1e-9:
        ok, step = -1, min(du, max_disp - d)
        while ok != 0 and step >= du / 64:                      # 실패하면 알고리즘 바꾸고, 그래도 안 되면 반으로
            for alg in ("Newton", "NewtonLineSearch", "KrylovNewton", "ModifiedNewton"):
                ops.algorithm(alg)
                ops.integrator("DisplacementControl", mesh["control"], 2, -step)
                ops.analysis("Static")
                ok = ops.analyze(1)
                if ok == 0:
                    break
            if ok != 0:
                step /= 2
        if ok != 0:
            log(f"  수렴 실패: 제어 변위 {d:.1f} mm에서 중단")
            break
        d += step
        disp.append(d)
        force.append(ops.getLoadFactor(1) * 1000.0)
        U.append(np.array([ops.nodeDisp(nd)[:2] for nd in record]))
        plate_max = float((-U[-1][plate_idx, 1]).max())
        if contact.update():
            ops.wipeAnalysis()
            setup()
        if len(disp) % 10 == 0:
            log(f"  제어 {d:6.1f} mm  {force[-1] / 1e3:7.1f} kN  Plate 침입 {plate_max:5.1f} mm  "
                f"접촉 {len(contact.pairs)}쌍  ({time.time() - t0:.0f}s)")
    stop = ("하중 목표 도달" if force_target else "Plate 목표 도달") if reached() else "제어 변위 상한 도달"
    log(f"  종료({stop}): 제어 {d:.1f} mm, Plate 침입 {plate_max:.1f} mm, {force[-1] / 1e3:.0f} kN")
    return np.array(disp), np.array(force), np.array(U), col


# ─────────────────────────────── 6. 후처리 ───────────────────────────────
def kabsch_residual(P, Q) -> float:
    """P → Q 2D 최적 강체변환(회전+이동) 후 최대 잔차 = 단면 찌그러짐."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    u, _, vt = np.linalg.svd(Pc.T @ Qc)
    R = vt.T @ np.diag([1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return float(np.max(np.linalg.norm(Qc - Pc @ R.T, axis=1)))


METRICS = ("plate_intrusion", "plate_distortion", "outer_intrusion", "d_crush")


def section_metrics(mesh, U, col) -> dict:
    """프레임 × 섹션 지표 (mm).

    plate_intrusion  : Plate(가장 안쪽, 탑승자 쪽) 절점의 −y 최대 이동량 — 최대 변형량 판단 기준
    plate_distortion : Plate 자체 찌그러짐 (Plate 절점에서 강체운동을 뺀 잔차)
    outer_intrusion  : Outer(바깥, 충돌면) 절점의 −y 최대 이동량 (참고)
    d_crush          : Outer 윗면 – Plate 사이 거리 감소 = 단면 높이 압궤 (참고)
    """
    out = {m: np.zeros((len(U), N_SEC)) for m in METRICS}
    xy0 = {nd: np.array(mesh["coords"][nd][:2]) for nd in col}
    for s, sec in mesh["sections"].items():
        plate = [col[nd] for nd in sec[PLATE]]
        outer = [col[nd] for nd in sec[OUTER]]
        P = np.array([xy0[nd] for nd in sec[PLATE]])
        cr = [col[sec[OUTER][k - 1]] for k in CROWN_PTS]
        pl = [col[sec[PLATE][k - 1]] for k in PLATE_PTS]
        cr0 = np.mean([xy0[sec[OUTER][k - 1]] for k in CROWN_PTS], 0)
        pl0 = np.mean([xy0[sec[PLATE][k - 1]] for k in PLATE_PTS], 0)
        D0 = np.linalg.norm(cr0 - pl0)
        for f, u in enumerate(U):
            out["plate_intrusion"][f, s] = max(0.0, (-u[plate, 1]).max())
            out["plate_distortion"][f, s] = kabsch_residual(P, P + u[plate])
            out["outer_intrusion"][f, s] = max(0.0, (-u[outer, 1]).max())
            out["d_crush"][f, s] = D0 - np.linalg.norm((cr0 + u[cr].mean(0)) - (pl0 + u[pl].mean(0)))
    return out


def force_states(force, forces_N) -> tuple[list[dict], float]:
    """하중 수준마다 정적 곡선의 어느 두 프레임 사이를 보간할지 정한다.

    1차 한계하중(이후 하중이 1% 넘게 떨어지는 첫 극대) 이하 → 상승 구간 보간 (stable).
    넘으면 → 일정 하중에서는 다음 평형으로 뛰어넘으므로, 한계 이후 처음 F ≥ F0인 상태 (snap_through).
    반환: [{F0, status, j0, j, w}] (not_reached면 j = None), 1차 한계하중(N)
    """
    F = force
    i_lim = next((i for i in range(1, len(F) - 1)
                  if F[i] >= F[i - 1] and F[i + 1] < F[i] and (F[i + 1:] < 0.99 * F[i]).any()), len(F) - 1)
    states = []
    for F0 in forces_N:
        if F0 <= F[i_lim]:
            j, status = int(np.argmax(F[: i_lim + 1] >= F0)), "stable"
        else:
            hit = np.flatnonzero(F[i_lim + 1:] >= F0)
            j, status = (i_lim + 1 + int(hit[0]), "snap_through") if len(hit) else (None, "not_reached")
        j0 = max(j - 1, 0) if j is not None else None
        w = 0.0 if j is None or j == j0 else (F0 - F[j0]) / (F[j] - F[j0])
        states.append(dict(F0=F0, status=status, j0=j0, j=j, w=w))
    return states, F[i_lim]


def interp(a, st):
    """프레임 배열 a에서 하중 수준 상태 st의 값 (선형 보간)."""
    return (1 - st["w"]) * a[st["j0"]] + st["w"] * a[st["j"]]


def deformation_table(states, disp, metrics, dz) -> pd.DataFrame:
    """(하중 수준 × 섹션) 변형량 표."""
    rows = []
    for st in states:
        ok = st["j"] is not None
        for s in range(N_SEC):
            rows.append(dict(F0_kN=round(st["F0"] / 1e3, 1), status=st["status"],
                             control_disp=interp(disp, st) if ok else np.nan, sec=s, z_mm=round(s * dz, 2),
                             **{m: interp(a, st)[s] if ok else np.nan for m, a in metrics.items()}))
    return pd.DataFrame(rows)


def deformed_shapes(mesh, parts, U, col, states) -> pd.DataFrame:
    """하중 수준별 섹션 단면 좌표 (변형 전 x0, y0 / 변형 후 x, y). 입력 CSV와 같은 point 번호.
    변형 전 좌표는 입력 CSV 값(병합된 플랜지 절점도 각 파트 원래 위치)에 절점 변위를 더한다."""
    rows = []
    for st in states:
        if st["j"] is None:
            continue
        u = interp(U, st)
        for s, sec in mesh["sections"].items():
            for pid, nodes in sec.items():
                for k, nd in enumerate(nodes, start=1):
                    x0, y0 = parts[pid]["xy"][s, k - 1]
                    ux, uy = u[col[nd]]
                    rows.append(dict(F0_kN=round(st["F0"] / 1e3, 1), status=st["status"], sec=s,
                                     part=parts[pid]["name"], point_idx=k, x0_mm=x0, y0_mm=y0,
                                     x_mm=x0 + ux, y_mm=y0 + uy))
    return pd.DataFrame(rows)


def plot_shapes(shapes: pd.DataFrame, out_dir: Path, name: str) -> list[Path]:
    """하중 수준마다 한 장: 17개 섹션의 변형 전(회색 점선)·후(파트별 색) 단면."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(PART_NAMES, ["C0", "C1", "C2", "C3", "C4"]))
    paths = []
    for F0, df in shapes.groupby("F0_kN"):
        fig, axes = plt.subplots(3, 6, figsize=(24, 10))
        for s, ax in enumerate(axes.flat):
            if s >= N_SEC:
                ax.axis("off")
                continue
            d = df[df["sec"] == s]
            for part, g in d.groupby("part", sort=False):
                ax.plot(g["x0_mm"], g["y0_mm"], ":", color="0.6", lw=0.8)
                ax.plot(g["x_mm"], g["y_mm"], "-", color=colors[part], lw=1.2, label=part)
            ax.set_title(f"sec {s}" + (" (load)" if LOAD_SECS[0] <= s <= LOAD_SECS[1] else ""), fontsize=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"{name}: deformed sections at F0 = {F0:.1f} kN ({df['status'].iloc[0]})  "
                     f"— dotted: undeformed", fontsize=12)
        fig.tight_layout()
        path = out_dir / f"{name}_shapes_F{F0:.0f}kN.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


# ─────────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="단면 형상 CSV")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR), help="결과 폴더 (기본 FEM/results)")
    ap.add_argument("--name", default=None, help="결과 파일 이름 앞부분 (기본: CSV 파일 이름)")
    ap.add_argument("--forces", type=float, nargs="+", metavar="kN",
                    help="결과를 뽑을 하중 수준 (기본: 해석에서 도달한 최대 하중까지 6단계)")
    ap.add_argument("--plate-target", type=float, default=50.0,
                    help="Plate 최대 침입량이 이 값(mm)에 닿을 때까지 하중을 올린다")
    ap.add_argument("--force-target", type=float, default=None, metavar="kN",
                    help="총 하중이 이 값에 닿을 때까지 올린다 (주면 --plate-target 대신 사용)")
    ap.add_argument("--max-disp", type=float, default=300.0, help="제어 절점 변위 상한 (mm)")
    ap.add_argument("--n-sub", type=int, default=4, help="섹션 사이 분할 수")
    ap.add_argument("--du", type=float, default=1.0, help="변위 증분 (mm)")
    ap.add_argument("--quick", action="store_true", help="n_sub=1, 2 mm 증분, Plate 목표 30 mm")
    a = ap.parse_args()
    if a.quick:
        a.n_sub, a.du, a.plate_target = 1, 2.0, min(a.plate_target, 30.0)

    t0 = time.time()
    dz = HEIGHT / (N_SEC - 1)
    parts = read_parts(a.csv)
    mesh = build_mesh(parts, a.n_sub, dz)
    goal = f"총 하중 {a.force_target:.0f} kN" if a.force_target else f"Plate 침입 {a.plate_target:.0f} mm"
    print(f"[mesh] 절점 {len(mesh['coords'])}, 요소 {len(mesh['elems'])}  "
          f"({goal}까지 하중 증가, 제어 변위 상한 {a.max_disp:.0f} mm)")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.csv).stem + ("_quick" if a.quick else "")
    ops.logFile(str(out_dir / f"{name}_opensees.log"), "-noEcho")        # OpenSees 경고는 파일로
    build_model(mesh)
    disp, force, U, col = run_static(mesh, a.plate_target, a.max_disp, a.du,
                                     force_target=a.force_target * 1e3 if a.force_target else None)

    # 하중 수준: 지정이 없으면 도달한 최대 하중까지 6등분 (모든 수준에 결과가 나온다)
    forces = [F * 1e3 for F in a.forces] if a.forces else list(np.linspace(1, 6, 6) / 6 * force.max())
    metrics = section_metrics(mesh, U, col)
    states, F_lim = force_states(force, forces)
    table = deformation_table(states, disp, metrics, dz)
    shapes = deformed_shapes(mesh, parts, U, col, states)
    table.to_csv(out_dir / f"{name}_deformation.csv", index=False, float_format="%.3f")
    shapes.to_csv(out_dir / f"{name}_shapes.csv", index=False, float_format="%.3f")
    pd.DataFrame(dict(control_disp_mm=disp, force_kN=force / 1e3,
                      plate_intrusion_max_mm=metrics["plate_intrusion"].max(axis=1))).to_csv(
        out_dir / f"{name}_curve.csv", index=False, float_format="%.3f")
    pngs = plot_shapes(shapes, out_dir, name)

    print(f"[done] {time.time() - t0:.0f}s  1차 한계하중 {F_lim / 1e3:.0f} kN  → {out_dir / name}_*"
          f"  (단면 그림 {len(pngs)}장)")
    print("섹션별 Plate 침입량 (mm) — 최대 변형량 기준")
    print(table.pivot(index="sec", columns="F0_kN", values="plate_intrusion").round(1).to_string())
    worst = table.loc[table.groupby("F0_kN")["plate_intrusion"].idxmax()]
    print("하중 수준별 최대 변형량 (Plate):")
    print(worst[["F0_kN", "status", "sec", "plate_intrusion"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
