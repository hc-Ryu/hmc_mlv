"""후처리 계산: 섹션별 변형량, 관통 판정, 하중 수준별 표, 파트별 변형 시작 (idea_v1.md §6, §7).

이 모듈은 계산만 한다(DataFrame/dict 반환). 파일 저장은 pipeline.py, 그림은 plots.py.

섹션 지표 (프레임마다 계산)
  delta_dir  : max_n (−u_y)                  충돌 방향 최대 침입량 (대표 지표)
  delta_rel  : 양단(sec 0 / 끝 섹션) 평균 u_y를 잇는 기준선 대비 상대 침입량
  delta_dist : 섹션 절점에 2D 강체변환(Kabsch)을 맞춘 뒤 남는 잔차 최대 노름 → 단면 찌그러짐
  d_crush    : Outer crown 도심 – Plate 대응 구간 도심 거리 감소량 (단면 높이 압궤)
  crown_gap  : Outer 몸체 절점 각각의 x에서 아래 표면(Inner/Patch1/Plate) 높이를 뺀 중심선 간격 최소값
  n_cross    : 판 중심선(폴리라인) 선분끼리 교차하는 쌍의 수 (> 0 이면 관통)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_SECTIONS
from .geometry import Mesh
from .io_csv import INNER, OUTER, OUTER_BODY_PTS, PATCH1, PLATE
from .solve_ops import History

SECTION_METRICS = ["delta_dir", "delta_rel", "delta_dist", "d_crush"]
GAP_TOL = -1.0          # crown_gap 허용치 (mm): penalty 침투 + 절점-절점 이산화


# ── 기하 도우미 ─────────────────────────────────────────────

def _kabsch_residual(P: np.ndarray, Q: np.ndarray) -> float:
    """P(초기) → Q(현재) 2D 최적 강체변환 후 최대 잔차."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return float(np.max(np.linalg.norm(Qc - Pc @ R.T, axis=1)))


class _Frames:
    """History를 배열로 풀어 두고 (파트, 섹션, point) → 기록 인덱스 조회를 제공한다."""

    def __init__(self, mesh: Mesh, hist: History):
        self.mesh = mesh
        self.times, self.force, self.U = hist.as_arrays()
        self.col = {nd: i for i, nd in enumerate(hist.nodes)}
        self.xy0 = np.array([mesh.coords[nd][:2] for nd in hist.nodes])

    def idx(self, pid: int, sec: int, pts) -> np.ndarray | None:
        lev = self.mesh.level_of_sec(sec)
        if not self.mesh.has(pid, lev):
            return None
        return np.array([self.col[self.mesh.node(pid, lev, k)] for k in pts])

    def part_idx(self, pid: int, sec: int) -> np.ndarray | None:
        return self.idx(pid, sec, range(1, self.mesh.parts[pid].n_pt + 1))

    def xy(self, f: int, idx: np.ndarray) -> np.ndarray:
        """프레임 f의 현재 2D 좌표."""
        return self.xy0[idx] + self.U[f, idx, :2]


def section_crossings(mesh: Mesh, hist: History) -> np.ndarray:
    """(프레임, 섹션) 판 중심선 선분 교차 쌍의 수.

    다른 파트 선분끼리, 같은 파트는 인접하지 않은 선분끼리 검사한다. 절점을 공유하는 선분
    (병합된 플랜지, 인접 선분)은 제외한다. 교차가 있으면 관통(x·y·자기 어느 방향이든)이다.
    """
    fr = _Frames(mesh, hist)
    out = np.zeros((len(fr.times), N_SECTIONS), dtype=int)
    for s in range(N_SECTIONS):
        lev = mesh.level_of_sec(s)
        seg, pid, k = [], [], []
        for p in mesh.parts:
            if not mesh.has(p.pid, lev):
                continue
            ns = [mesh.node(p.pid, lev, j) for j in range(1, p.n_pt + 1)]
            for j in range(p.n_pt - 1):
                if ns[j] != ns[j + 1]:
                    seg.append((ns[j], ns[j + 1]))
                    pid.append(p.pid)
                    k.append(j)
        seg, pid, k = np.array(seg), np.array(pid), np.array(k)
        ia = np.array([fr.col[a] for a in seg[:, 0]])
        ib = np.array([fr.col[b] for b in seg[:, 1]])
        i1, i2 = np.triu_indices(len(seg), 1)
        share = ((seg[i1, 0] == seg[i2, 0]) | (seg[i1, 0] == seg[i2, 1]) |
                 (seg[i1, 1] == seg[i2, 0]) | (seg[i1, 1] == seg[i2, 1]))
        adjacent = (pid[i1] == pid[i2]) & (np.abs(k[i1] - k[i2]) < 2)
        i1, i2 = i1[~share & ~adjacent], i2[~share & ~adjacent]
        for f in range(len(fr.times)):
            A, B = fr.xy(f, ia), fr.xy(f, ib)
            p, r = A[i1], B[i1] - A[i1]
            q, w = A[i2], B[i2] - A[i2]
            den = r[:, 0] * w[:, 1] - r[:, 1] * w[:, 0]
            ok = np.abs(den) > 1e-12
            safe = np.where(ok, den, 1.0)
            qp = q - p
            t = (qp[:, 0] * w[:, 1] - qp[:, 1] * w[:, 0]) / safe
            u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / safe
            eps = 1e-6
            out[f, s] = int(np.sum(ok & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)))
    return out


def _crown_gap(fr: _Frames, s: int) -> np.ndarray:
    """프레임별: Outer 몸체 절점 x에서 아래 표면(Inner/Patch1/Plate 폴리라인) 높이를 뺀 간격의 최소값."""
    o_idx = fr.idx(OUTER, s, OUTER_BODY_PTS)
    lower = [i for i in (fr.part_idx(pid, s) for pid in (INNER, PATCH1, PLATE)) if i is not None]
    gap = np.empty(len(fr.times))
    for f in range(len(fr.times)):
        o = fr.xy(f, o_idx)
        X = o[:, :1]
        surf = np.full(len(o), -np.inf)
        for li in lower:
            q = fr.xy(f, li)
            x1, y1, x2, y2 = q[:-1, 0], q[:-1, 1], q[1:, 0], q[1:, 1]
            inside = (X >= np.minimum(x1, x2)) & (X <= np.maximum(x1, x2))
            dx = np.where(np.abs(x2 - x1) < 1e-9, 1e-9, x2 - x1)
            ys = y1 + (X - x1) / dx * (y2 - y1)
            surf = np.maximum(surf, np.where(inside, ys, -np.inf).max(axis=1))
        g = o[:, 1] - surf
        gap[f] = g[np.isfinite(g)].min() if np.isfinite(g).any() else np.inf
    return gap


# ── 섹션 지표 ───────────────────────────────────────────────

def section_metrics(mesh: Mesh, hist: History) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(섹션별 전 구간 최대값 표, 프레임 × 섹션 시계열 표)."""
    cfg = mesh.cfg
    fr = _Frames(mesh, hist)
    times, force, U = fr.times, fr.force, fr.U
    sec_idx = {s: np.array([fr.col[nd] for nd in mesh.section_nodes(s)]) for s in range(N_SECTIONS)}
    cross = section_crossings(mesh, hist)
    uy_end0 = U[:, sec_idx[0], 1].mean(axis=1)
    uy_end1 = U[:, sec_idx[N_SECTIONS - 1], 1].mean(axis=1)

    rows, series = [], []
    for s in range(N_SECTIONS):
        idx = sec_idx[s]
        uy = U[:, idx, 1]
        zr = s / (N_SECTIONS - 1)
        uref = (1 - zr) * uy_end0 + zr * uy_end1
        d_dir = (-uy).max(axis=1)
        d_rel = (-(uy - uref[:, None])).max(axis=1)
        umag = np.linalg.norm(U[:, idx, :], axis=2).max(axis=1)
        P = fr.xy0[idx]
        d_dist = np.array([_kabsch_residual(P, fr.xy(f, idx)) for f in range(len(times))])

        crown, plate = fr.idx(OUTER, s, cfg.crown_pts), fr.idx(PLATE, s, cfg.plate_crush_pts)
        D = np.array([np.linalg.norm(fr.xy(f, crown).mean(0) - fr.xy(f, plate).mean(0))
                      for f in range(len(times))])
        d_crush = D[0] - D
        gap = _crown_gap(fr, s)

        f_max = int(np.argmax(d_dir))
        rows.append(dict(sec=s, z_mm=s * cfg.dz,
                         delta_dir=d_dir.max(), delta_rel=d_rel.max(), delta_dist=d_dist.max(),
                         d_crush=d_crush.max(), max_u=umag.max(), t_at_max=times[f_max],
                         delta_dir_final=d_dir[-1], delta_dist_final=d_dist[-1],
                         min_crown_gap=gap.min(), max_crossings=int(cross[:, s].max())))
        series.append(pd.DataFrame(dict(frame=np.arange(len(times)), t=times, force=force, sec=s,
                                        delta_dir=d_dir, delta_rel=d_rel, delta_dist=d_dist,
                                        d_crush=d_crush, crown_gap=gap, n_cross=cross[:, s])))
    return pd.DataFrame(rows), pd.concat(series, ignore_index=True)


# ── 정적 곡선 → 하중 수준별 표 ─────────────────────────────

def static_force_table(series: pd.DataFrame, alphas, P_c: float) -> tuple[pd.DataFrame, dict]:
    """정적 변위제어 곡선에서 하중 F0 = α·P_c 별 섹션 변형을 읽는다 (일정 하중의 준정적 해).

    status
      stable          : F0 ≤ F_lim1(1차 한계하중). 상승 구간에서 F = F0인 상태를 선형 보간
      snap_through    : F0 > F_lim1. 일정 하중에서는 뛰어넘어(snap-through) 다음 평형으로 간다.
                        유효 구간 안에 F = F0인 다음 평형이 있으면 그 상태
      invalid_contact : 다음 평형이 관통 이후에만 존재 → 판정 불가
      no_equilibrium  : 해석 범위 안에 F = F0 상태가 없음 (범위를 늘려야 함)
    반환: (α별 wide 표, 한계값 dict)
    """
    curve = series[series["sec"] == 0].sort_values("frame")
    F = curve["force"].to_numpy()
    d = curve["t"].to_numpy()
    gap = series.pivot(index="frame", columns="sec", values="crown_gap").min(axis=1).to_numpy()
    n_cross = series.pivot(index="frame", columns="sec", values="n_cross").sum(axis=1).to_numpy()
    bad = np.flatnonzero((gap < GAP_TOL) | (n_cross > 0))
    i_valid = int(bad[0]) - 1 if len(bad) else len(F) - 1

    i_lim = len(F) - 1          # 1차 한계하중: 이후 하중이 1% 넘게 떨어지는 첫 극대
    for i in range(1, len(F) - 1):
        if F[i] >= F[i - 1] and F[i + 1] < F[i] and np.any(F[i + 1:] < 0.99 * F[i]):
            i_lim = i
            break
    k0 = F[1] / d[1] if len(F) > 1 and d[1] > 0 else np.nan
    sec_k = np.divide(F[1:], d[1:], out=np.full(len(F) - 1, np.nan), where=d[1:] > 0)
    nl = np.flatnonzero(sec_k < 0.95 * k0)

    wide = {m: series.pivot(index="frame", columns="sec", values=m) for m in SECTION_METRICS}

    def state(j0, j, F0):
        w = 0.0 if j == j0 else (F0 - F[j0]) / (F[j] - F[j0])
        st = {"control_disp": (1 - w) * d[j0] + w * d[j]}
        for m in SECTION_METRICS:
            v = (1 - w) * wide[m].iloc[j0] + w * wide[m].iloc[j]
            st.update({f"{m}_s{s}": val for s, val in v.items()})
        return st

    rows = []
    for al in alphas:
        F0 = al * P_c
        row = dict(alpha=al, F0_kN=F0 / 1e3)
        if F0 <= F[i_lim]:
            j = int(np.argmax(F[: i_lim + 1] >= F0))
            row.update(status="stable", **state(max(j - 1, 0), j, F0))
        else:
            hit = np.flatnonzero(F[i_lim + 1:] >= F0)
            if not len(hit):
                row["status"] = "invalid_contact" if i_valid < len(F) - 1 else "no_equilibrium"
            elif i_lim + 1 + hit[0] > i_valid:
                row["status"] = "invalid_contact"
            else:
                j = i_lim + 1 + int(hit[0])
                row.update(status="snap_through", **state(j - 1, j, F0))
        rows.append(row)

    limits = dict(F_lim1_kN=F[i_lim] / 1e3, control_disp_at_lim1=float(d[i_lim]),
                  alpha_lim1=F[i_lim] / P_c,
                  F_linear_limit_kN=float(F[nl[0] + 1]) / 1e3 if len(nl) else float("nan"),
                  valid_until_control_disp=float(d[i_valid]),
                  F_at_valid_end_kN=float(F[i_valid]) / 1e3,
                  F_max_in_range_kN=float(F.max()) / 1e3, P_c_kN=P_c / 1e3)
    return pd.DataFrame(rows), {k: float(v) for k, v in limits.items()}


def deformation_long(force_table: pd.DataFrame, dz: float) -> pd.DataFrame:
    """α별 wide 표 → 하중 수준 × 섹션 한 행씩인 정돈된 표 (데모의 최종 출력)."""
    rows = []
    for _, r in force_table.iterrows():
        for s in range(N_SECTIONS):
            row = dict(F0_kN=round(r["F0_kN"], 1), alpha=r["alpha"], status=r["status"],
                       sec=s, z_mm=round(s * dz, 2))
            for m in SECTION_METRICS:
                v = r.get(f"{m}_s{s}", np.nan)
                row[m] = 0.0 if abs(v) < 1e-6 else v        # 고정단의 −0, 1e-14 같은 수치 잡음 정리
            rows.append(row)
    return pd.DataFrame(rows)


# ── 파트별 ──────────────────────────────────────────────────

def part_metrics(mesh: Mesh, hist: History) -> pd.DataFrame:
    """프레임 × 섹션 × 파트: 파트 자체 찌그러짐, 최대 이동량, 최대 침입량, 항복 비율.
    sec = −1 행은 파트 전체 요소 기준 항복 비율."""
    fr = _Frames(mesh, hist)
    yr, ym = hist.extra.get("yield_r"), hist.extra.get("yield_meta")
    rows = []
    for s in range(N_SECTIONS):
        for p in mesh.parts:
            idx = fr.part_idx(p.pid, s)
            if idx is None:
                continue
            P = fr.xy0[idx]
            sel = (ym["pid"] == p.pid) & (ym["sec"] == s) & ym["beam"] if yr is not None else None
            for f in range(len(fr.times)):
                u = fr.U[f, idx]
                row = dict(frame=f, t=fr.times[f], force=fr.force[f], sec=s, part=p.name,
                           distortion=_kabsch_residual(P, P + u[:, :2]) if len(idx) >= 3 else 0.0,
                           max_u=float(np.linalg.norm(u, axis=1).max()),
                           intrusion=float((-u[:, 1]).max()))
                if sel is not None and sel.any():
                    r = np.asarray(yr[f])[sel]
                    row.update(yield_frac=float(np.mean(r >= 1.0)), max_r=float(r.max()))
                rows.append(row)
    if yr is not None:
        for p in mesh.parts:
            sel = (ym["pid"] == p.pid) & ym["beam"]
            for f in range(len(fr.times)):
                r = np.asarray(yr[f])[sel]
                rows.append(dict(frame=f, t=fr.times[f], force=fr.force[f], sec=-1, part=p.name,
                                 yield_frac=float(np.mean(r >= 1.0)), max_r=float(r.max())))
    return pd.DataFrame(rows)


def part_onset(pm: pd.DataFrame, P_c: float, distortion_levels=(1.0, 5.0),
               yield_levels=(0.05, 0.20)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """파트별 변형 시작 하중: 최초 항복, 항복 비율 ≥ 5/20 %, 찌그러짐 ≥ 1/5 mm (양 끝 제외 섹션 중 최대).
    반환: (시작 하중 표, 파트 × 프레임 이력)."""
    inner = pm[(pm["sec"] >= 1) & (pm["sec"] <= N_SECTIONS - 2)]
    g = inner.groupby(["part", "frame"], sort=False).agg(
        t=("t", "first"), force=("force", "first"),
        distortion=("distortion", "max"), intrusion=("intrusion", "max"))
    if "yield_frac" in pm:
        whole = pm[pm["sec"] == -1].set_index(["part", "frame"])
        g["yield_frac"] = whole["yield_frac"]
        g["max_r"] = whole["max_r"]
    g = g.reset_index()

    rows = []
    for part, d in g.groupby("part", sort=False):
        d = d.sort_values("frame")
        row = dict(part=part)

        def first(mask, key):
            i = np.flatnonzero(mask.to_numpy())
            row[f"{key}_F_kN"] = d["force"].iloc[i[0]] / 1e3 if len(i) else np.nan
            if len(i):
                row[f"{key}_alpha"] = d["force"].iloc[i[0]] / P_c
                row[f"{key}_disp"] = d["t"].iloc[i[0]]

        if "max_r" in d:
            first(d["max_r"] >= 1.0, "first_yield")
        if "yield_frac" in d:
            for lv in yield_levels:
                first(d["yield_frac"] >= lv, f"yield{int(lv * 100)}pct")
        for lv in distortion_levels:
            first(d["distortion"] >= lv, f"distort{lv:g}mm")
        row["final_distortion_mm"] = d["distortion"].iloc[-1]
        row["final_yield_frac"] = d["yield_frac"].iloc[-1] if "yield_frac" in d else np.nan
        rows.append(row)
    return pd.DataFrame(rows), g


# ── explicit α 스윕 ─────────────────────────────────────────

def sweep_table(cases: list[tuple[float, float, pd.DataFrame]]) -> pd.DataFrame:
    """explicit α 스윕: (alpha, F0, 섹션 최대값 표) 목록 → 하나의 표."""
    frames = []
    for alpha, F0, tab in cases:
        t = tab.copy()
        t.insert(0, "F0_kN", F0 / 1e3)
        t.insert(0, "alpha", alpha)
        frames.append(t)
    return pd.concat(frames, ignore_index=True).sort_values(["alpha", "sec"])


# ── 이력 저장/복원 ──────────────────────────────────────────

def history_arrays(hist: History) -> dict:
    """history.npz에 저장할 배열."""
    times, force, U = hist.as_arrays()
    out = dict(times=times, force=force, U=U, nodes=np.array(hist.nodes),
               kinetic=np.array(hist.kinetic), ext_work=np.array(hist.ext_work))
    for key in ("n_contact", "n_contact_x", "imp_sec_force"):
        if key in hist.extra:
            out[key] = np.array(hist.extra[key])
    if "yield_r" in hist.extra:
        m = hist.extra["yield_meta"]
        out.update(yield_r=np.asarray(hist.extra["yield_r"], dtype=np.float32),
                   elem_pid=m["pid"], elem_sec=m["sec"], elem_beam=m["beam"])
    return out


def history_from_npz(h) -> History:
    """history.npz → History (재처리용)."""
    hist = History(nodes=list(h["nodes"]), times=list(h["times"]), force=list(h["force"]),
                   U=list(h["U"]), kinetic=list(h["kinetic"]), ext_work=list(h["ext_work"]))
    for key in ("n_contact", "n_contact_x", "imp_sec_force"):
        if key in h.files:
            hist.extra[key] = list(h[key])
    if "yield_r" in h.files:
        hist.extra["yield_r"] = list(h["yield_r"])
        hist.extra["yield_meta"] = dict(pid=h["elem_pid"], sec=h["elem_sec"], beam=h["elem_beam"])
    return hist
