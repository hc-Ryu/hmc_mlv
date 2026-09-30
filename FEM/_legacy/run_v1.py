"""B-pillar 격자 모델 충돌 해석 (데모): 설계 CSV → 섹션별 변형량.

기본 실행 (conda env fem, 약 20분):
    python run_v1.py
빠른 동작 확인 (거친 격자, 약 4분):
    python run_v1.py --quick
다른 설계안 / 하중 수준 지정:
    python run_v1.py --csv path/to/final_section_xxx.csv --forces 100 200 300
결과만 다시 계산 (해석 재실행 없음):
    python run_v1.py --reprocess results/<폴더>

주 출력: results/<이름>/section_deformation.csv (하중 수준 × 섹션별 변형량)
하중 방식(--load-mode): top(기본, Outer 윗면 균일 압력) | crown6(crown 6점) | impactor(강체 평판)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from bpillar_fem import pipeline, plots, post
from bpillar_fem.config import Config
from bpillar_fem.geometry import build_mesh, summary
from bpillar_fem.io_csv import read_parts
from bpillar_fem.loads import section_shares

DEFAULT_ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    io = ap.add_argument_group("입력 / 출력")
    io.add_argument("--csv", default=Config.csv_path, help="설계 단면 CSV (기본 v1_21_0_x1.3)")
    io.add_argument("--height", type=float, default=Config.height, help="sec 0 ~ 끝 섹션 높이 (mm)")
    io.add_argument("--out", default=Config.out_dir, help="결과 루트 폴더")
    io.add_argument("--name", default=None, help="결과 폴더 이름 (기본: CSV 이름_모드_하중방식)")
    io.add_argument("--info", action="store_true", help="메쉬·하중 요약만 출력")
    io.add_argument("--reprocess", default=None, metavar="DIR", help="저장된 결과의 후처리만 다시 실행")

    ld = ap.add_argument_group("하중")
    ld.add_argument("--load-mode", default=Config.load_mode, choices=["top", "crown6", "impactor"])
    lv = ld.add_mutually_exclusive_group()
    lv.add_argument("--forces", type=float, nargs="+", metavar="kN", help="결과를 뽑을 하중 수준 (kN)")
    lv.add_argument("--alphas", type=float, nargs="+", help=f"하중 수준 = α·P_c (기본 {DEFAULT_ALPHAS})")

    sv = ap.add_argument_group("풀이")
    sv.add_argument("--mode", default=Config.mode, choices=["static", "explicit"])
    sv.add_argument("--quick", action="store_true", help="거친 격자(n_sub=1)·짧은 범위로 빠르게 확인")
    sv.add_argument("--target", type=float, default=Config.static_target, help="정적: 제어 변위 (mm)")
    sv.add_argument("--du", type=float, default=Config.static_du, help="정적: 변위 증분 (mm)")
    sv.add_argument("--n-sub", type=int, default=Config.n_sub, help="섹션 사이 종방향 분할 수")
    sv.add_argument("--bc", default=Config.bc, choices=["fixed", "pinned"])
    sv.add_argument("--no-contact", action="store_true", help="접촉 끄기")
    sv.add_argument("--no-contact-x", action="store_true", help="x 방향 벽–벽 접촉만 끄기")

    ex = ap.add_argument_group("explicit 전용")
    ex.add_argument("--profile", default=Config.profile, choices=["P1", "P2", "P3"])
    ex.add_argument("--t-ramp", type=float, default=None, help=f"하중 램프 시간 (s, 기본 {Config.t_ramp})")
    ex.add_argument("--t-hold", type=float, default=None, help=f"하중 유지 시간 (s, 기본 {Config.t_hold})")
    ex.add_argument("--mass-scale", type=float, default=Config.mass_scale)
    ex.add_argument("--workers", type=int, default=1, help="α 여러 개를 병렬로")
    return ap.parse_args(argv)


def make_config(a: argparse.Namespace) -> Config:
    cfg = Config(csv_path=a.csv, height=a.height, out_dir=a.out, mode=a.mode,
                 load_mode=a.load_mode, n_sub=a.n_sub, bc=a.bc,
                 static_target=a.target, static_du=a.du,
                 contact=a.mode == "static" and not a.no_contact, contact_x=not a.no_contact_x,
                 profile=a.profile, mass_scale=a.mass_scale)
    if a.quick:
        cfg = replace(cfg, n_sub=1, static_target=min(a.target, 60.0), static_du=2.0,
                      t_ramp=0.002, t_hold=0.002, t_free=0.002, n_frames=100)
    # 명시한 시간 옵션은 --quick보다 우선한다
    if a.t_ramp is not None:
        cfg.t_ramp = a.t_ramp
    if a.t_hold is not None:
        cfg.t_hold = a.t_hold
    return cfg


def main(argv=None) -> int:
    try:        # Windows 콘솔(cp949)에서 출력 불가 문자로 해석이 중단되지 않도록
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    a = parse_args(argv)
    cfg = make_config(a)
    if cfg.mode == "explicit" and cfg.load_mode == "impactor":
        sys.exit("impactor 하중은 정적 해석 전용입니다 (--mode static)")

    alphas = (list(a.alphas) if a.alphas else
              [F * 1e3 / cfg.P_c for F in a.forces] if a.forces else
              DEFAULT_ALPHAS if cfg.mode == "static" else [1.0])

    if a.reprocess:
        pipeline.reprocess(Path(a.reprocess), alphas)
        return 0

    if a.info:
        mesh = build_mesh(read_parts(cfg.csv_path), cfg)
        print(summary(mesh))
        print(f"dz={cfg.dz:.3f} mm  P_c={cfg.P_c / 1e3:.1f} kN  load_mode={cfg.load_mode}")
        print("하중 수준 (kN):", [round(al * cfg.P_c / 1e3, 1) for al in alphas])
        print("섹션별 하중 비율:", {k: round(v, 4) for k, v in section_shares(mesh).items()})
        return 0

    base_name = a.name or (f"{Path(cfg.csv_path).stem}_{cfg.mode}_{cfg.load_mode}"
                           + (f"_{cfg.bc}" if cfg.bc != "fixed" else "") + ("_quick" if a.quick else ""))

    if cfg.mode == "static":
        pipeline.run_case(replace(cfg, alpha=0.0, run_name=base_name), alphas)
        return 0

    cfgs = [replace(cfg, alpha=al, run_name=f"{base_name}_{cfg.profile}_a{al:.2f}") for al in alphas]
    if a.workers > 1 and len(cfgs) > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(pipeline.run_case, cfgs))
    else:
        results = [pipeline.run_case(c) for c in cfgs]
    if len(results) > 1:
        root = Path(cfg.out_dir) / f"{base_name}_{cfg.profile}_sweep"
        root.mkdir(parents=True, exist_ok=True)
        sweep = post.sweep_table([(al, F0, tab) for al, F0, tab, _ in results])
        sweep.to_csv(root / "sweep_summary.csv", index=False, float_format="%.5g")
        plots.plot_sweep(root, sweep)
        print(f"sweep summary → {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
