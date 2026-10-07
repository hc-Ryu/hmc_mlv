"""B-pillar 단면 CSV → 격자 FEM(OpenSeesPy) → 섹션별 변형량 CSV (v2, 단일 파일).

bpillar_fem_demo.py(v1.21: 5파트 × 17섹션 고정)를 일반화한 판. 바뀐 점:
    1. 입력 일반화 — 섹션 수·파트 수를 CSV에서 읽는다 (v1.22: 7파트 × 15섹션, v1.21 입력도 동작).
       일부 섹션에서 삭제된 파트(Patch1 sec 1/9/11–13, Patch4 sec 3–4 없음)는 floor_idx로 위치를 맞추고,
       존재 섹션 구간마다 따로 메쉬한다. 섹션 s의 파트는 z ∈ [s−½, s+½]·dz 구간을 대표한다고 보고
       구간 양 끝을 반 피치씩 연장한다 (sec 0, 마지막 섹션에서는 자름) — 섹션 1개짜리 Patch도 폭 dz의 판이 된다.
    2. 원주 방향 strip 폭 = 실제 판의 종방향 기여 길이 (위·아래 종방향 요소 길이의 평균).
       v1은 수직 간격 dz/n_sub를 썼는데, 단면이 섹션마다 크게 벌어지는 하부(flare, 연결선 기울기 최대 66°)에서는
       판 길이가 dz/cos θ라 재료량·강성이 최대 2.4배 적게 들어갔다.
    3. 섹션 사이 보간 = 같은 point 번호끼리 PCHIP(단조 보존 3차, 과도 진동 없음) — 하부 포물선 flare를
       매끄럽게 잇는다. --kink-secs 섹션(기본 1)에서는 곡선을 끊어 의도된 단차(초기 형상 v8의 sec 0 ×1.3 확대)를
       직선 + 꺾임으로 남긴다. --interp linear 이면 v1과 같은 선형 보간.
       덧판(Patch1–4)은 붙은 파트(Plate 또는 Outer)의 보간 형상 + 섹션별 오프셋 보간으로 만들어 항상 판에 붙어 있다.

사용 (conda env fem — Python 3.12 + openseespy):
    python bpillar_fem_v2.py                                     # 기본 입력 (v1.22 후처리 단면)
    python bpillar_fem_v2.py --quick                             # 거친 격자(n_sub = 2)로 빠르게
    python bpillar_fem_v2.py --csv 설계.csv --plate-target 80 --forces 100 200 300
    python bpillar_fem_v2.py --csv 설계.csv --energy-target 30            # 외력 일 30 kJ까지 (같은 충돌 에너지 비교)

종료 조건 (우선순위 순): --energy-target(외력 일 kJ) → --force-target(총 하중 kN) → --plate-target(Plate 침입 mm, 기본).
    충돌 시험은 대차 질량·속도가 같아 입력 에너지(½mv²)가 같으므로, 설계 비교에는 --energy-target가 시험 조건에 가깝다.
    외력 일 = Σ(하중 절점 힘 · y 변위 증분)의 누적 — 분포 하중이라 총 하중 × 제어 변위보다 작다.

하중·경계 (v1과 같은 물리 위치 — 섹션 피치 65.73 mm 동일, v1.22는 위 2개 섹션만 삭제)
    하중: sec 5–11 Outer 윗면 −y 균일 압력, 변위제어 기준 = sec 8 Outer pt 15. 경계: sec 0, 마지막 섹션 전 절점 고정.

출력 (--out-dir, 파일 이름 앞부분 = CSV 파일 이름, --quick이면 _quick 추가)
    <name>_deformation.csv   하중 수준(F0) 또는 에너지 수준(E, --energy-target) × 섹션별 변형량 (mm)
                             — 열 의미는 bpillar_fem_demo.py와 같고 E_kJ(그 시점 외력 일) 열 추가
    <name>_shapes.csv        수준별 변형된 섹션 단면 좌표
    <name>_shapes_F*kN.png   수준별 섹션 단면 그림 (점선 = 변형 전, 에너지 기준이면 _shapes_E*kJ.png)
    <name>_curve.csv         제어 변위 – 총 하중 – 외력 일 – Plate 최대 침입량
    <name>_mesh.csv          레벨(z)별 파트 존재·종방향 연결선 최대 기울기·strip 폭 보정 배율 (메쉬 점검용)
    <name>_opensees.log      OpenSees 경고

모델 (단위: mm, N, MPa) — 요소·재료·접촉은 v1과 같음
    - 요소: 섹션 안(원주) 선분과 레벨 사이(같은 point 번호끼리) 선분 → 판 strip 파이버 보.
      strip 단면 = 폭 b × 두께 t, 두께 방향 6층, 면내 I = b·t·L²/(12(1+ν)).
      ※ 이 등가식은 직교 격자 가정이라 연결선이 크게 기운 하부(sec 0–2)에서는 면내 전단 강성이 근사다.
    - 파트 접합: 플랜지 point 쌍을 한 절점으로 병합. Patch4는 양 끝 point(pt1, pt14)만 Outer pt9, pt22에 병합하고
      가운데는 접촉만 둔다 (CGNN은 접촉만 — 고정단에 닿지 않는 Patch4 구간의 강체 모드를 막기 위한 FEM 전용 접합).
    - 접촉: 위 파트 절점과 바로 아래 파트 절점 사이 y 방향 압축 전용 gap. 층 순서 Outer → Patch4 → Inner → Patch1 → Plate.
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
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "02. post_processing" / "results" / "v22" / "final_section_v1_22_0_ref_post.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DZ = 1051.7 / 16                    # 섹션 피치 (sec 0 sill ~ sec 16 roof rail 1051.7 mm 기준)
E, NU, HARDENING = 210000.0, 0.3, 0.01
N_THICK = 6                         # strip 두께 방향 파이버 층수

PART_NAMES = ["Outer", "Plate", "Inner", "Patch1", "Patch2", "Patch3", "Patch4"]
PART_FY = [1470.0, 980.0, 1470.0, 980.0, 440.0, 440.0, 980.0]
PART_COLORS = ["#FF5722", "#FFAA00", "#4CAF50", "#2196F3", "#9C27B0", "#CE93D8", "#E91E63"]
OUTER, PLATE, INNER, PATCH1, PATCH2, PATCH3, PATCH4 = range(7)
# 붙은 파트를 따라가는 덧판 (덧판, 기준 파트, 기준 파트의 첫 point 번호) — 덧판 point k ↔ 기준 point (첫+k−1)
FOLLOWERS = {PATCH1: (PLATE, 10), PATCH2: (PLATE, 22), PATCH3: (PLATE, 7), PATCH4: (OUTER, 9)}
# 플랜지 접합 (part_a, point_a, part_b, point_b) — CGNN FLANGE_PAIR_SPEC
FLANGE_PAIRS = [
    (OUTER, 1, PLATE, 1), (OUTER, 2, PLATE, 2), (OUTER, 3, PLATE, 3),
    (OUTER, 28, PLATE, 28), (OUTER, 29, PLATE, 29), (OUTER, 30, PLATE, 30),
    (PLATE, 7, INNER, 1), (PLATE, 8, INNER, 2), (PLATE, 9, INNER, 3),
    (PLATE, 22, INNER, 16), (PLATE, 23, INNER, 17), (PLATE, 24, INNER, 18),
    (PLATE, 10, PATCH1, 1), (PLATE, 21, PATCH1, 12),
    (PLATE, 22, PATCH2, 1), (PLATE, 23, PATCH2, 2), (PLATE, 24, PATCH2, 3),
    (PLATE, 7, PATCH3, 1), (PLATE, 8, PATCH3, 2), (PLATE, 9, PATCH3, 3),
    # FEM 전용: Patch4 양 끝을 Outer에 접합 (CGNN은 접촉만). 고정단에 닿지 않는 Patch4 구간(v23 sec 5–6)이
    # x·z 방향 강체 모드로 떠다니지 않게 한다. 가운데 point는 접촉만.
    (OUTER, 9, PATCH4, 1), (OUTER, 22, PATCH4, 14),
]
HUB = [1, 0, 2, 3, 4, 5, 6]         # 병합 대표 우선순위 (Plate → Outer → Inner → Patch)
LOAD_SECS = (5, 11)                 # 하중 구간 섹션
CONTROL_SEC, CONTROL_PT = 8, 15     # 정적 변위제어 기준 절점 (Outer)
CROWN_PTS = range(13, 19)           # d_crush 기준: Outer 윗면 point
PLATE_PTS = range(13, 19)           #               그 아래 Plate point
CONTACT_K = 2.0e5                   # 접촉 penalty (N/mm)
CONTACT_WIN = 6.0                   # 접촉 짝 찾기 x 창 (mm)
CONTACT_ACT = 3.0                   # 간격 − 접촉거리 < 이 값이면 접촉 요소 생성 (mm)


# ─────────────────────────────── 1. CSV ───────────────────────────────
def read_parts(path) -> list[dict]:
    """CSV → 파트 목록 [{name, t, fy, xy (n_sec, n_pt, 2)}]. 파트 블록 끝 구분 행(point_idx = 0)에 두께.
    파트가 없는 섹션의 xy는 NaN."""
    df = pd.read_csv(path)
    df["block"] = (df["point_idx"] == 0).cumsum().shift(fill_value=0)
    n_sec = int(df.loc[df["point_idx"] > 0, "floor_idx"].max()) + 1
    parts = []
    for pid, blk in df.groupby("block"):
        d = blk[blk["point_idx"] > 0]
        floors, pts = d["floor_idx"].astype(int).to_numpy(), d["point_idx"].astype(int).to_numpy()
        xy = np.full((n_sec, pts.max(), 2), np.nan)
        xy[floors, pts - 1] = d[["x_mm", "y_mm"]].to_numpy(float)
        parts.append(dict(name=PART_NAMES[pid], t=float(blk.loc[blk["point_idx"] == 0, "t_mm"].iloc[0]),
                          fy=PART_FY[pid], xy=xy))
    return parts


def alive(p) -> np.ndarray:
    return ~np.isnan(p["xy"][:, 0, 0])


def contiguous_runs(mask) -> list[np.ndarray]:
    """True가 이어진 섹션 구간들의 인덱스 배열."""
    idx = np.flatnonzero(mask)
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []


# ─────────────────────────────── 2. 섹션 사이 보간 ───────────────────────────────
def pchip(xk, yk, x) -> np.ndarray:
    """단조 보존 3차 Hermite 보간 (Fritsch–Carlson, scipy PchipInterpolator와 같은 기울기 규칙).
    yk: (m, ...) — 첫 축을 따라 보간, x는 [xk[0], xk[-1]] 안."""
    xk, yk, x = np.asarray(xk, float), np.asarray(yk, float), np.asarray(x, float)
    if len(xk) == 2:
        w = ((x - xk[0]) / (xk[1] - xk[0])).reshape(-1, *([1] * (yk.ndim - 1)))
        return (1 - w) * yk[0] + w * yk[1]
    h = np.diff(xk).reshape(-1, *([1] * (yk.ndim - 1)))
    delta = np.diff(yk, axis=0) / h
    d = np.zeros_like(yk)
    w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
    same = delta[:-1] * delta[1:] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        d[1:-1] = np.where(same, (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:]), 0.0)

    def end(h0, h1, d0, d1):
        e = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        e = np.where(np.sign(e) != np.sign(d0), 0.0, e)
        return np.where((np.sign(d0) != np.sign(d1)) & (np.abs(e) > 3 * np.abs(d0)), 3 * d0, e)
    d[0], d[-1] = end(h[0], h[1], delta[0], delta[1]), end(h[-1], h[-2], delta[-1], delta[-2])

    i = np.clip(np.searchsorted(xk, x, side="right") - 1, 0, len(xk) - 2)
    hh = h[i]
    t = ((x - xk[i]).reshape(-1, *([1] * (yk.ndim - 1)))) / hh
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * yk[i] + h10 * hh * d[i] + h01 * yk[i + 1] + h11 * hh * d[i + 1]


def interp_sections(secs, vals, s_query, kinks, method) -> np.ndarray:
    """섹션 secs의 값 vals (m, ...)를 섹션 좌표 s_query에서 보간. kinks 섹션에서 곡선을 끊는다.
    s_query가 secs 범위 밖이면 가장 가까운 끝 값 (덧판 반 피치 연장용)."""
    s_query = np.atleast_1d(np.asarray(s_query, float))
    out = np.empty((len(s_query), *vals.shape[1:]))
    if len(secs) == 1:
        out[:] = vals[0]
        return out
    cuts = [0] + [i for i, s in enumerate(secs) if s in kinks and 0 < i < len(secs) - 1] + [len(secs) - 1]
    sq = np.clip(s_query, secs[0], secs[-1])
    for a, b in zip(cuts[:-1], cuts[1:]):
        m = (sq >= secs[a]) & (sq <= secs[b])
        if not m.any():
            continue
        xs, ys = np.asarray(secs[a:b + 1], float), vals[a:b + 1]
        if method == "linear" or len(xs) == 2:
            j = np.clip(np.searchsorted(xs, sq[m], side="right") - 1, 0, len(xs) - 2)
            w = ((sq[m] - xs[j]) / (xs[j + 1] - xs[j])).reshape(-1, *([1] * (ys.ndim - 1)))
            out[m] = (1 - w) * ys[j] + w * ys[j + 1]
        else:
            out[m] = pchip(xs, ys, sq[m])
    return out


def part_levels(parts, n_sub, kinks, method) -> list[dict]:
    """파트별 {레벨: xy (n_pt, 2)}. 레벨 lev의 z = lev·dz/n_sub.
    기준 파트(Outer/Plate/Inner)는 전 레벨, 덧판은 존재 섹션 구간 ± 반 피치를 기준 파트 + 오프셋 보간으로."""
    n_sec = len(parts[OUTER]["xy"])
    last, half = (n_sec - 1) * n_sub, n_sub // 2
    geo = [dict() for _ in parts]
    for pid, p in enumerate(parts):
        if pid in FOLLOWERS:
            continue
        secs = np.flatnonzero(alive(p))
        levs = np.arange(secs[0] * n_sub, secs[-1] * n_sub + 1)
        for lev, xy in zip(levs, interp_sections(secs, p["xy"][secs], levs / n_sub, kinks, method)):
            geo[pid][int(lev)] = xy
    for pid, (base, k0) in FOLLOWERS.items():
        if pid >= len(parts):
            continue
        p = parts[pid]
        n = p["xy"].shape[1]
        for run in contiguous_runs(alive(p)):
            off = p["xy"][run] - parts[base]["xy"][run, k0 - 1:k0 - 1 + n]
            levs = np.arange(max(run[0] * n_sub - half, 0), min(run[-1] * n_sub + half, last) + 1)
            offs = interp_sections(run, off, levs / n_sub, kinks, method)
            for lev, o in zip(levs, offs):
                geo[pid][int(lev)] = geo[base][int(lev)][k0 - 1:k0 - 1 + n] + o
    return geo


# ─────────────────────────────── 3. 메쉬 ───────────────────────────────
def build_mesh(parts, n_sub, kinks=(1,), method="pchip") -> dict:
    """절점(병합 후), 요소, 섹션별 절점을 만든다. 절점 번호 = pid*100000 + lev*100 + point."""
    n_sec = len(parts[OUTER]["xy"])
    dzl = DZ / n_sub
    last = (n_sec - 1) * n_sub
    geo = part_levels(parts, n_sub, kinks, method)

    raw = {}
    for pid, g in enumerate(geo):
        for lev, xy in g.items():
            for k, (x, y) in enumerate(xy, start=1):
                raw[pid * 100000 + lev * 100 + k] = (x, y, lev * dzl)

    parent = {t: t for t in raw}                                 # 플랜지 병합: union-find

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t
    for pa, ka, pb, kb in FLANGE_PAIRS:
        if max(pa, pb) >= len(parts):
            continue
        for lev in set(geo[pa]) & set(geo[pb]):
            ra, rb = find(pa * 100000 + lev * 100 + ka), find(pb * 100000 + lev * 100 + kb)
            if ra != rb:
                lo, hi = sorted([ra, rb], key=lambda r: (HUB[r // 100000], r))
                parent[hi] = lo
    node = {t: find(t) for t in raw}
    coords = {r: raw[r] for r in set(node.values())}

    def n(pid, lev, k):
        return node[pid * 100000 + lev * 100 + k]

    def p3(pid, lev):
        xy = geo[pid][lev]
        return np.c_[xy, np.full(len(xy), lev * dzl)]

    def tributary(xy):                                           # 원주 방향 기여 폭
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        return np.r_[seg, 0] / 2 + np.r_[0, seg] / 2

    def normals(xy):                                             # point별 판 법선 (면내)
        d = np.diff(xy, axis=0)
        sn = np.stack([-d[:, 1], d[:, 0]], 1) / np.linalg.norm(d, axis=1)[:, None]
        pn = np.r_[sn, sn[-1:]] + np.r_[sn[:1], sn]
        return sn, pn / np.linalg.norm(pn, axis=1)[:, None]

    elems, elem_info, mesh_rows = [], [], []
    for pid, g in enumerate(geo):
        p = parts[pid]
        for lev in sorted(g):
            xy = g[lev]
            sn, pn = normals(xy)
            up, dn = lev + 1 in g, lev - 1 in g
            L_up = np.linalg.norm(p3(pid, lev + 1) - p3(pid, lev), axis=1) if up else np.zeros(len(xy))
            L_dn = np.linalg.norm(p3(pid, lev) - p3(pid, lev - 1), axis=1) if dn else np.zeros(len(xy))
            trib_z = 0.5 * (L_up + L_dn)                         # 점별 종방향 기여 길이 (실제 판 길이)
            b_c = 0.5 * (trib_z[:-1] + trib_z[1:])
            for k in range(1, len(xy)):                          # 원주 방향
                i, j = n(pid, lev, k), n(pid, lev, k + 1)
                if i != j:
                    elems.append((i, j, b_c[k - 1], p["t"], p["fy"], sn[k - 1]))
                    elem_info.append((pid, "circ"))
            if up:                                               # 종방향 (같은 point끼리)
                b_l = 0.5 * (tributary(xy) + tributary(g[lev + 1]))
                for k in range(1, len(xy) + 1):
                    elems.append((n(pid, lev, k), n(pid, lev + 1, k), b_l[k - 1], p["t"], p["fy"], pn[k - 1]))
                    elem_info.append((pid, "long"))
                incl = np.degrees(np.arccos(np.clip(dzl / L_up, -1, 1)))
                mesh_rows.append(dict(part=p["name"], lev=lev, z0_mm=lev * dzl, z1_mm=(lev + 1) * dzl,
                                      max_incl_deg=incl.max(), max_strip_scale=(L_up / dzl).max()))

    # 하중: 하중 구간 모든 레벨의 Outer 윗면(|n_y| ≥ 0.5, 플랜지 제외)에 x 투영 폭 비율
    load = {}
    levs = np.arange(LOAD_SECS[0] * n_sub, LOAD_SECS[1] * n_sub + 1)
    w_lev = np.ones(len(levs))
    w_lev[[0, -1]] = 0.5
    for lev, wl in zip(levs, w_lev / w_lev.sum()):
        xy = geo[OUTER][int(lev)]
        _, pn = normals(xy)
        dx = np.abs(np.diff(xy[:, 0]))
        bx = np.r_[dx, 0] / 2 + np.r_[0, dx] / 2
        pts = [k for k in range(6, 26) if abs(pn[k - 1, 1]) >= 0.5]
        for k in pts:
            nd = n(OUTER, int(lev), k)
            load[nd] = load.get(nd, 0.0) + wl * bx[k - 1] / bx[np.array(pts) - 1].sum()

    # 섹션별 절점 (입력 CSV에 그 파트가 있는 섹션만)
    sections = {s: {pid: [n(pid, s * n_sub, k) for k in range(1, p["xy"].shape[1] + 1)]
                    for pid, p in enumerate(parts) if alive(p)[s]} for s in range(n_sec)}
    shared = {r for r, cnt in Counter(node.values()).items() if cnt > 1}
    contact_levels = [{pid: [n(pid, lev, k) for k in range(1, p["xy"].shape[1] + 1) if n(pid, lev, k) not in shared]
                       for pid, p in enumerate(parts) if lev in geo[pid]}
                      for lev in range(last + 1)]
    ends = {n(pid, lev, k) for pid, g in enumerate(geo) for lev in (0, last) if lev in g
            for k in range(1, parts[pid]["xy"].shape[1] + 1)}
    return dict(coords=coords, elems=elems, elem_info=elem_info, load=load, sections=sections,
                contact_levels=contact_levels, thick=[p["t"] for p in parts], shared=shared, ends=ends,
                control=n(OUTER, CONTROL_SEC * n_sub, CONTROL_PT), n_sec=n_sec,
                mesh_table=pd.DataFrame(mesh_rows))


# ─────────────────────────────── 4. OpenSees 모델 ───────────────────────────────
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
    for nd in mesh["ends"]:                                            # 양단 전 절점 고정
        ops.fix(nd, 1, 1, 1, 1, 1, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nd, f in mesh["load"].items():                                 # 참조 하중 합계 1 kN
        ops.load(nd, 0.0, -1000.0 * f, 0.0, 0.0, 0.0, 0.0)


# ─────────────────────────────── 5. 접촉 (y 방향) ───────────────────────────────
class Contact:
    """위 파트 절점 ↔ 바로 아래 파트 절점 y 방향 gap. 층 순서 Outer → Patch4 → Inner → Patch1 → Plate."""
    ORDER = [OUTER, PATCH4, INNER, PATCH1, PLATE]

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


# ─────────────────────────────── 6. 정적 해석 ───────────────────────────────
def run_static(mesh, plate_target, max_disp, du, force_target=None, energy_target=None, log=print):
    """제어 절점을 −y로 du씩 누른다. 멈추는 조건 (우선순위 순): energy_target(N·mm)이 있으면 외력 일이 그 값에,
    force_target(N)이 있으면 총 하중이 그 값에, 둘 다 없으면 Plate 최대 침입량이 plate_target에 닿을 때.
    어느 쪽이든 제어 변위 max_disp가 상한.
    외력 일 = Σ (하중 절점 힘 · y 변위 증분), 사다리꼴 적분 — 분포 하중이라 총 하중 × 제어 변위와 다르다.
    반환: 제어 변위, 총 하중(N), 외력 일(N·mm), 섹션 절점 변위 이력, 절점 → 열 번호."""
    record = sorted({nd for sec in mesh["sections"].values() for nodes in sec.values() for nd in nodes})
    col = {nd: i for i, nd in enumerate(record)}
    plate_idx = sorted({col[nd] for sec in mesh["sections"].values() for nd in sec[PLATE]})
    load_nodes = list(mesh["load"])
    p_ref = -1000.0 * np.array([mesh["load"][nd] for nd in load_nodes])   # 하중계수 1일 때 절점 y 힘 (N)
    contact = Contact(mesh)
    contact.update()

    def setup():
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1e-6, 50)

    setup()
    disp, force, energy = [0.0], [0.0], [0.0]
    U = [np.zeros((len(record), 2))]
    uy_load = np.zeros(len(load_nodes))
    d, plate_max, t0 = 0.0, 0.0, time.time()

    def reached():
        if energy_target:
            return energy[-1] >= energy_target
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
        uy_new = np.array([ops.nodeDisp(nd, 2) for nd in load_nodes])
        energy.append(energy[-1] + 0.5 * (force[-2] + force[-1]) / 1000.0 * float(p_ref @ (uy_new - uy_load)))
        uy_load = uy_new
        U.append(np.array([ops.nodeDisp(nd)[:2] for nd in record]))
        plate_max = float((-U[-1][plate_idx, 1]).max())
        if contact.update():
            ops.wipeAnalysis()
            setup()
        if len(disp) % 10 == 0:
            log(f"  제어 {d:6.1f} mm  {force[-1] / 1e3:7.1f} kN  {energy[-1] / 1e6:6.2f} kJ  "
                f"Plate 침입 {plate_max:5.1f} mm  접촉 {len(contact.pairs)}쌍  ({time.time() - t0:.0f}s)", flush=True)
    goal = "에너지" if energy_target else "하중" if force_target else "Plate"
    stop = f"{goal} 목표 도달" if reached() else "제어 변위 상한 도달"
    log(f"  종료({stop}): 제어 {d:.1f} mm, Plate 침입 {plate_max:.1f} mm, {force[-1] / 1e3:.0f} kN, "
        f"외력 일 {energy[-1] / 1e6:.2f} kJ")
    return np.array(disp), np.array(force), np.array(energy), np.array(U), col


# ─────────────────────────────── 7. 후처리 ───────────────────────────────
def kabsch_residual(P, Q) -> float:
    """P → Q 2D 최적 강체변환(회전+이동) 후 최대 잔차 = 단면 찌그러짐."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    u, _, vt = np.linalg.svd(Pc.T @ Qc)
    R = vt.T @ np.diag([1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return float(np.max(np.linalg.norm(Qc - Pc @ R.T, axis=1)))


METRICS = ("plate_intrusion", "plate_distortion", "outer_intrusion", "d_crush")


def section_metrics(mesh, U, col) -> dict:
    """프레임 × 섹션 지표 (mm). 의미는 bpillar_fem_demo.section_metrics와 같음."""
    out = {m: np.zeros((len(U), mesh["n_sec"])) for m in METRICS}
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


def limit_index(F) -> int:
    """1차 한계하중 프레임: 하중이 꺾여 내려간 뒤 1 % 넘게 떨어지는 첫 극대. 없으면 마지막 프레임."""
    return next((i for i in range(1, len(F) - 1)
                 if F[i] >= F[i - 1] and F[i + 1] < F[i] and (F[i + 1:] < 0.99 * F[i]).any()), len(F) - 1)


def force_states(force, energy, forces_N) -> tuple[list[dict], float]:
    """하중 수준마다 정적 곡선의 어느 두 프레임 사이를 보간할지 정한다 (1차 한계하중 기준 stable/snap_through)."""
    F = force
    i_lim = limit_index(F)
    states = []
    for F0 in forces_N:
        if F0 <= F[i_lim]:
            j, status = int(np.argmax(F[: i_lim + 1] >= F0)), "stable"
        else:
            hit = np.flatnonzero(F[i_lim + 1:] >= F0)
            j, status = (i_lim + 1 + int(hit[0]), "snap_through") if len(hit) else (None, "not_reached")
        j0 = max(j - 1, 0) if j is not None else None
        w = 0.0 if j is None or j == j0 else (F0 - F[j0]) / (F[j] - F[j0])
        st = dict(F0=F0, status=status, j0=j0, j=j, w=w)
        st["E0"] = interp(energy, st) if j is not None else np.nan
        states.append(st)
    return states, F[i_lim]


def energy_states(force, energy, energies_Nmm) -> tuple[list[dict], float]:
    """에너지 수준마다 외력 일이 그 값에 처음 닿는 두 프레임 사이를 보간한다 (같은 충돌 에너지 비교용).
    status는 그 지점이 1차 한계하중 전(stable)인지 후(snap_through)인지."""
    i_lim = limit_index(force)
    states = []
    for E0 in energies_Nmm:
        hit = np.flatnonzero(energy >= E0)
        j = int(hit[0]) if len(hit) else None
        j0 = max(j - 1, 0) if j is not None else None
        w = 0.0 if j is None or j == j0 else (E0 - energy[j0]) / (energy[j] - energy[j0])
        status = "not_reached" if j is None else "stable" if j <= i_lim else "snap_through"
        st = dict(E0=E0, status=status, j0=j0, j=j, w=w)
        st["F0"] = interp(force, st) if j is not None else np.nan
        states.append(st)
    return states, force[i_lim]


def interp(a, st):
    """프레임 배열 a에서 하중 수준 상태 st의 값 (선형 보간)."""
    return (1 - st["w"]) * a[st["j0"]] + st["w"] * a[st["j"]]


def deformation_table(states, disp, metrics, n_sec) -> pd.DataFrame:
    """(하중·에너지 수준 × 섹션) 변형량 표."""
    rows = []
    for st in states:
        ok = st["j"] is not None
        for s in range(n_sec):
            rows.append(dict(F0_kN=round(st["F0"] / 1e3, 1), E_kJ=round(st["E0"] / 1e6, 3), status=st["status"],
                             control_disp=interp(disp, st) if ok else np.nan, sec=s, z_mm=round(s * DZ, 2),
                             **{m: interp(a, st)[s] if ok else np.nan for m, a in metrics.items()}))
    return pd.DataFrame(rows)


def deformed_shapes(mesh, parts, U, col, states) -> pd.DataFrame:
    """하중 수준별 섹션 단면 좌표 (변형 전 x0, y0 = 입력 CSV / 변형 후 x, y)."""
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
                    rows.append(dict(F0_kN=round(st["F0"] / 1e3, 1), E_kJ=round(st["E0"] / 1e6, 3),
                                     status=st["status"], sec=s,
                                     part=parts[pid]["name"], point_idx=k, x0_mm=x0, y0_mm=y0,
                                     x_mm=x0 + ux, y_mm=y0 + uy))
    return pd.DataFrame(rows)


def plot_shapes(shapes: pd.DataFrame, n_sec, out_dir: Path, name: str, by="F0_kN") -> list[Path]:
    """수준(by = F0_kN 하중 또는 E_kJ 에너지)마다 한 장: 섹션별 변형 전(회색 점선)·후(파트별 색) 단면."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(PART_NAMES, PART_COLORS))
    ncol = 6
    paths = []
    for _, df in shapes.groupby(by):
        F0, E0 = df["F0_kN"].iloc[0], df["E_kJ"].iloc[0]
        label, tag = (f"F0 = {F0:.1f} kN (E = {E0:.2f} kJ)", f"F{F0:.0f}kN") if by == "F0_kN" else \
                     (f"E = {E0:.2f} kJ (F = {F0:.1f} kN)", f"E{E0:.2f}kJ")
        fig, axes = plt.subplots(int(np.ceil(n_sec / ncol)), ncol, figsize=(24, 3.4 * np.ceil(n_sec / ncol)))
        for s, ax in enumerate(axes.flat):
            if s >= n_sec:
                ax.axis("off")
                continue
            d = df[df["sec"] == s]
            for part, g in d.groupby("part", sort=False):
                ax.plot(g["x0_mm"], g["y0_mm"], ":", color="0.6", lw=0.8)
                ax.plot(g["x_mm"], g["y_mm"], "-", color=colors[part], lw=1.2, label=part)
            ax.set_title(f"sec {s}" + (" (load)" if LOAD_SECS[0] <= s <= LOAD_SECS[1] else ""), fontsize=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
        handles = {}
        for ax in axes.flat:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(lab, h)
        axes.flat[0].legend(handles.values(), handles.keys(), fontsize=7)
        fig.suptitle(f"{name}: deformed sections at {label} ({df['status'].iloc[0]})  "
                     f"— dotted: undeformed", fontsize=12)
        fig.tight_layout()
        path = out_dir / f"{name}_shapes_{tag}.png"
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
    ap.add_argument("--energy-target", type=float, default=None, metavar="kJ",
                    help="외력 일(흡수 에너지)이 이 값에 닿을 때까지 올린다 — 같은 충돌 에너지 비교용 "
                         "(주면 --force-target, --plate-target 대신 사용, 결과도 에너지 수준별)")
    ap.add_argument("--energies", type=float, nargs="+", metavar="kJ",
                    help="--energy-target일 때 결과를 뽑을 에너지 수준 (기본: 목표 에너지까지 6단계)")
    ap.add_argument("--max-disp", type=float, default=300.0, help="제어 절점 변위 상한 (mm)")
    ap.add_argument("--n-sub", type=int, default=4, help="섹션 사이 분할 수 (짝수 — 덧판 반 피치 연장)")
    ap.add_argument("--du", type=float, default=1.0, help="변위 증분 (mm)")
    ap.add_argument("--interp", choices=("pchip", "linear"), default="pchip", help="섹션 사이 보간")
    ap.add_argument("--kink-secs", type=int, nargs="*", default=[1],
                    help="보간 곡선을 끊는 섹션 (의도된 단차, 기본 1 = sec 0 확대 단차)")
    ap.add_argument("--quick", action="store_true", help="n_sub=2, 2 mm 증분, Plate 목표 30 mm")
    a = ap.parse_args()
    if a.quick:
        a.n_sub, a.du, a.plate_target = 2, 2.0, min(a.plate_target, 30.0)
    if a.n_sub % 2:
        ap.error("--n-sub는 짝수여야 합니다 (덧판 존재 구간을 섹션 ± 반 피치로 연장)")

    t0 = time.time()
    parts = read_parts(a.csv)
    mesh = build_mesh(parts, a.n_sub, set(a.kink_secs), a.interp)
    n_sec = mesh["n_sec"]
    goal = (f"외력 일 {a.energy_target:.2f} kJ" if a.energy_target else
            f"총 하중 {a.force_target:.0f} kN" if a.force_target else f"Plate 침입 {a.plate_target:.0f} mm")
    mt = mesh["mesh_table"]
    print(f"[input] {Path(a.csv).name}: {len(parts)}파트 × {n_sec}섹션, 높이 {(n_sec - 1) * DZ:.1f} mm, "
          f"보간 {a.interp} (꺾임 섹션 {sorted(a.kink_secs)})")
    print(f"[mesh] 절점 {len(mesh['coords'])}, 요소 {len(mesh['elems'])}, n_sub {a.n_sub}  "
          f"(종방향 연결선 최대 기울기 {mt['max_incl_deg'].max():.1f}°, strip 폭 보정 최대 ×{mt['max_strip_scale'].max():.2f})  "
          f"({goal}까지 하중 증가, 제어 변위 상한 {a.max_disp:.0f} mm)", flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.csv).stem + ("_quick" if a.quick else "")
    mt.to_csv(out_dir / f"{name}_mesh.csv", index=False, float_format="%.3f")
    ops.logFile(str(out_dir / f"{name}_opensees.log"), "-noEcho")        # OpenSees 경고는 파일로
    build_model(mesh)
    disp, force, energy, U, col = run_static(
        mesh, a.plate_target, a.max_disp, a.du,
        force_target=a.force_target * 1e3 if a.force_target else None,
        energy_target=a.energy_target * 1e6 if a.energy_target else None)

    metrics = section_metrics(mesh, U, col)
    if a.energy_target:                                  # 에너지 수준별 (kJ → N·mm)
        levels = [E * 1e6 for E in a.energies] if a.energies else list(np.linspace(1, 6, 6) / 6 * a.energy_target * 1e6)
        states, F_lim = energy_states(force, energy, levels)
        key, what = "E_kJ", "에너지"
    else:
        forces = [F * 1e3 for F in a.forces] if a.forces else list(np.linspace(1, 6, 6) / 6 * force.max())
        states, F_lim = force_states(force, energy, forces)
        key, what = "F0_kN", "하중"
    table = deformation_table(states, disp, metrics, n_sec)
    shapes = deformed_shapes(mesh, parts, U, col, states)
    table.to_csv(out_dir / f"{name}_deformation.csv", index=False, float_format="%.3f")
    shapes.to_csv(out_dir / f"{name}_shapes.csv", index=False, float_format="%.3f")
    pd.DataFrame(dict(control_disp_mm=disp, force_kN=force / 1e3, energy_kJ=energy / 1e6,
                      plate_intrusion_max_mm=metrics["plate_intrusion"].max(axis=1))).to_csv(
        out_dir / f"{name}_curve.csv", index=False, float_format="%.4f")
    pngs = plot_shapes(shapes, n_sec, out_dir, name, by=key)

    print(f"[done] {time.time() - t0:.0f}s  1차 한계하중 {F_lim / 1e3:.0f} kN  외력 일 {energy[-1] / 1e6:.2f} kJ"
          f"  → {out_dir / name}_*  (단면 그림 {len(pngs)}장)")
    print("섹션별 Plate 침입량 (mm) — 최대 변형량 기준")
    print(table.pivot(index="sec", columns=key, values="plate_intrusion").round(1).to_string())
    worst = table.loc[table.dropna(subset=["plate_intrusion"]).groupby(key)["plate_intrusion"].idxmax()]
    print(f"{what} 수준별 최대 변형량 (Plate):")
    print(worst[["E_kJ", "F0_kN", "status", "sec", "plate_intrusion"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
