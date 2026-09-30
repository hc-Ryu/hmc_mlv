"""빠른 회귀 테스트 (수 초):  conda activate fem && python -m pytest tests -q

기본 설계 CSV(v1_21_0_x1.3)를 쓴다. 해석 테스트는 거친 격자(n_sub=1)·짧은 변위만 돈다.
"""
from __future__ import annotations

import numpy as np
import pytest

from bpillar_fem import loads, post
from bpillar_fem.config import N_SECTIONS, Config
from bpillar_fem.geometry import build_mesh
from bpillar_fem.io_csv import OUTER, read_parts
from bpillar_fem.section_props import plastic_moment


@pytest.fixture(scope="module")
def parts():
    return read_parts(Config().csv_path)


def test_parse(parts):
    assert [p.name for p in parts] == ["Outer", "Plate", "Inner", "Patch1", "Patch2"]
    assert [p.n_pt for p in parts] == [30, 30, 18, 12, 3]
    assert parts[3].last_sec == 8                                   # Patch1은 sec 0–8만
    area = sum((np.linalg.norm(np.diff(p.xy, axis=1), axis=2).sum(axis=1) * p.t).sum() for p in parts)
    assert area == pytest.approx(24128.1, abs=0.5)                 # 리포트 §3


def test_mesh(parts):
    mesh = build_mesh(parts, Config(n_sub=1))
    assert len(mesh.rep) - len(mesh.coords) == 273                  # 병합 = 리포트 §6 접합 쌍 수
    assert sum(mesh.mass.values()) * 1e3 == pytest.approx(11.70, abs=0.01)   # kg
    assert len(mesh.section_nodes(0)) > 0 and mesh.dt > 0


def test_section_props(parts):
    assert plastic_moment(parts, 0) == pytest.approx(69_902_672, rel=0.01)   # 리포트 §1
    assert plastic_moment(parts, 16) == pytest.approx(27_203_696, rel=0.01)
    assert Config().P_c == pytest.approx(445.5e3, rel=0.01)


@pytest.mark.parametrize("mode", ["top", "crown6"])
def test_load_distribution(parts, mode):
    mesh = build_mesh(parts, Config(n_sub=2, load_mode=mode))
    frac = loads.nodal_fractions(mesh)
    assert sum(frac.values()) == pytest.approx(1.0)
    assert all(nd // 100000 == OUTER for nd in frac)                # 하중은 Outer 절점에만
    shares = loads.section_shares(mesh)
    assert shares[5] == pytest.approx(shares[11]) and sum(shares.values()) == pytest.approx(1.0)


def test_patch_calibrated_shear():
    from bpillar_fem import patch_test
    g, e = patch_test.run(a=8.0, c=16.4, mode="calibrated", nx=6, nz=4)
    assert g == pytest.approx(1.0, abs=0.01) and e == pytest.approx(1.0, abs=0.01)


def test_static_smoke():
    from bpillar_fem import solve_ops
    cfg = Config(n_sub=1, static_target=4.0, static_du=2.0, contact=False)
    mesh, hist = solve_ops.run_static(cfg, log=lambda *_: None)
    assert len(hist.times) == 3 and hist.force[-1] > hist.force[1] > 0
    table, series = post.section_metrics(mesh, hist)
    assert len(table) == N_SECTIONS
    assert abs(table.loc[0, "delta_dir"]) < 1e-9                     # 고정단은 움직이지 않음
    assert table["delta_dir"].idxmax() in range(*cfg.load_secs)     # 최대 침입은 하중 구간
    force_table, limits = post.static_force_table(series, [0.05], cfg.P_c)
    assert force_table.loc[0, "status"] == "stable"
    long = post.deformation_long(force_table, cfg.dz)
    assert len(long) == N_SECTIONS and long["delta_dir"].max() > 0
