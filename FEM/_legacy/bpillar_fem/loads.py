"""하중 분배(하중 구간 섹션의 Outer 절점)와 시간 이력 (idea_v1.md §5.1–5.2, §7.7).

반환하는 분배 계수는 합이 1이다. 절점 하중 = -y 방향 * F0 * 계수 * ts(t).
"""
from __future__ import annotations

import numpy as np

from .geometry import Mesh, _level_xy, _tributary
from .io_csv import OUTER


def load_levels(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """하중 레벨과 레벨별 가중치(합 1)."""
    cfg = mesh.cfg
    n = cfg.n_sub
    s_a, s_b = cfg.load_secs
    if cfg.load_dist == "sections":
        levs = np.array([s * n for s in range(s_a, s_b + 1)])
        w = np.ones(len(levs))
    elif cfg.load_dist == "band":
        levs = np.arange(s_a * n, s_b * n + 1)
        w = np.ones(len(levs))
        w[[0, -1]] = 0.5                       # 사다리꼴 적분 → 균일 띠하중
    else:
        raise ValueError(cfg.load_dist)
    if cfg.weights == "hann":
        xi = (levs - levs[0]) / max(levs[-1] - levs[0], 1)
        w = w * np.sin(np.pi * (0.1 + 0.8 * xi)) ** 2    # 양 끝도 0이 되지 않도록 여유
    elif cfg.weights != "uniform":
        raise ValueError(cfg.weights)
    return levs, w / w.sum()


def top_points(xy: np.ndarray, normal_min: float = 0.5) -> np.ndarray:
    """Outer 윗면 point 번호(1-base): 국부 법선의 |n_y| ≥ normal_min, 플랜지(pt 1–5, 26–30) 제외."""
    from .geometry import _point_normals
    n = _point_normals(xy)
    return np.array([k + 1 for k in range(len(xy)) if abs(n[k, 1]) >= normal_min and 5 < k + 1 < 26])


def _projected_width(xy: np.ndarray) -> np.ndarray:
    """x 방향 투영 기여 폭 (양 끝 반쪽)."""
    dx = np.abs(np.diff(xy[:, 0]))
    b = np.zeros(len(xy))
    b[:-1] += 0.5 * dx
    b[1:] += 0.5 * dx
    return b


def nodal_fractions(mesh: Mesh) -> dict[int, float]:
    """절점 -> 하중 분배 계수 (합 1).

    load_mode
      'crown6'  : crown_pts(기본 pt 13–18)에 원주 기여 폭 비율 (RBE3 등가)
      'top'     : Outer 윗면 전체(|n_y| ≥ 0.5, 플랜지 제외)에 x 투영 폭 비율 = 균일 압력
      'impactor': 분배 계수 없음 (강체 임팩터와의 접촉으로 결정, solve_ops 참고)
    """
    cfg = mesh.cfg
    if cfg.load_mode == "impactor":
        return {}
    outer = mesh.parts[OUTER]
    out = {}
    levs, w = load_levels(mesh)
    for lev, wl in zip(levs, w):
        xy = _level_xy(outer, int(lev), cfg.n_sub)
        if cfg.load_mode == "crown6":
            pts = np.array(cfg.crown_pts)
            b = _tributary(xy)[pts - 1]
        elif cfg.load_mode == "top":
            pts = top_points(xy)
            b = _projected_width(xy)[pts - 1]
        else:
            raise ValueError(cfg.load_mode)
        for k, f in zip(pts, b / b.sum()):
            nd = mesh.node(OUTER, int(lev), int(k))
            out[nd] = out.get(nd, 0.0) + wl * f
    return out


def section_shares(mesh: Mesh) -> dict[int, float]:
    """보고용: 섹션별 하중 비율 (레버 룰로 양쪽 섹션에 선형 분배 — 정적 등가)."""
    levs, w = load_levels(mesh)
    n = mesh.cfg.n_sub
    shares = {}
    for lev, wl in zip(levs, w):
        s, j = divmod(int(lev), n)
        f = j / n
        shares[s] = shares.get(s, 0.0) + float(wl) * (1.0 - f)
        if f > 0:
            shares[s + 1] = shares.get(s + 1, 0.0) + float(wl) * f
    return shares


def time_history(cfg) -> tuple[list[float], list[float], float]:
    """(times, values, t_end). values는 0..1 (F0 배율)."""
    tr, th, tf = cfg.t_ramp, cfg.t_hold, cfg.t_free
    if cfg.profile == "P1":
        t = [0.0, tr, tr + th]
        v = [0.0, 1.0, 1.0]
    elif cfg.profile == "P2":
        t = [0.0, tr, tr + th, 2 * tr + th, 2 * tr + th + tf]
        v = [0.0, 1.0, 1.0, 0.0, 0.0]
    elif cfg.profile == "P3":
        t = list(np.linspace(0.0, cfg.t_pulse, 41))
        v = list(np.sin(np.pi * np.array(t) / cfg.t_pulse))
        t += [cfg.t_pulse + tf]
        v += [0.0]
    else:
        raise ValueError(cfg.profile)
    return [float(x) for x in t], [float(x) for x in v], float(t[-1])
