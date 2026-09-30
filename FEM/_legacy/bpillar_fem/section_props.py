"""단면 성질과 기준 붕괴하중 (idea_v1.md §5.4).

CSV 단면만으로 계산하므로 설계안(CSV)이 바뀌어도 그대로 쓸 수 있다.
v1_21_0_x1.3에서 소성 모멘트는 리포트 값과 0.4% 이내로 일치한다.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .io_csv import Part, read_parts


def plastic_moment(parts: list[Part], sec: int) -> float:
    """섹션 sec의 소성 모멘트 Mp (N·mm), 충돌 방향(y) 굽힘.

    각 폴리라인 선분을 면적 L·t, 항복응력 fy인 점으로 보고, 인장·압축 합력이 같아지는
    소성 중립축 y_p를 찾은 뒤 Mp = Σ fy·A·|y − y_p| 로 계산한다.
    """
    y, force = [], []
    for p in parts:
        if sec > p.last_sec:
            continue
        xy = p.xy[sec]
        length = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        y.append(0.5 * (xy[1:, 1] + xy[:-1, 1]))
        force.append(length * p.t * p.fy)
    y, force = np.concatenate(y), np.concatenate(force)
    order = np.argsort(y)
    cum = np.cumsum(force[order])
    y_p = y[order][np.searchsorted(cum, 0.5 * cum[-1])]
    return float(np.sum(force * np.abs(y - y_p)))


def collapse_load(parts: list[Part], height: float, load_secs: tuple, hinge_sec: int) -> float:
    """이상화 3힌지 붕괴하중 P_c (N): 양단 고정, 힌지 = 양단 + hinge_sec, 하중 = load_secs 구간 균일.

    가상일: P · (하중 구간 평균 처짐) = Mp(0)·θ1 + Mp(h)·(θ1 + θ2) + Mp(끝)·θ2.
    국부좌굴을 무시한 완전소성 상한이다.
    """
    n_sec = max(p.last_sec for p in parts) + 1
    dz = height / (n_sec - 1)
    z_h = hinge_sec * dz
    theta1, theta2 = 1.0 / z_h, 1.0 / (height - z_h)          # 힌지 처짐 1일 때 회전각
    internal = (plastic_moment(parts, 0) * theta1
                + plastic_moment(parts, hinge_sec) * (theta1 + theta2)
                + plastic_moment(parts, n_sec - 1) * theta2)
    z = np.linspace(load_secs[0] * dz, load_secs[1] * dz, 401)
    shape = np.where(z <= z_h, z * theta1, (height - z) * theta2)
    return internal / float(shape.mean())


@lru_cache(maxsize=8)
def collapse_load_from_csv(csv_path: str, height: float, load_secs: tuple, hinge_sec: int) -> float:
    return collapse_load(read_parts(csv_path), height, load_secs, hinge_sec)
