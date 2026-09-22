# tests/test_segment_ids.py — compute_segment_ids() 1D CCL 단위 테스트 (command_v2.md §6-1)
# 실행: py tests/test_segment_ids.py  (CGNN 디렉터리에서; torch만 필요, 학습 없음)
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_design_v1_1 import (  # noqa: E402
    compute_segment_ids, count_thickness_groups, make_pruning_state_multi,
    STATE_ALIVE, STATE_DELETED, NUM_SECTIONS, CANDIDATE_PARTS, CONTINUOUS_PARTS,
)

A, D = STATE_ALIVE, STATE_DELETED


def make_state(col0, col1):
    ps = make_pruning_state_multi()
    ps['state'][:, 0] = torch.tensor(col0, dtype=torch.long)
    ps['state'][:, 1] = torch.tensor(col1, dtype=torch.long)
    return ps


def test_all_alive():
    ps = make_pruning_state_multi()
    seg = compute_segment_ids(ps)
    assert seg.shape == (NUM_SECTIONS, len(CANDIDATE_PARTS))
    assert seg.dtype == torch.long
    assert (seg == 0).all(), f"전부 ALIVE면 run 1개(전 칸 0)여야 함: {seg[:, 0].tolist()}"
    assert count_thickness_groups(seg) == len(CONTINUOUS_PARTS) + 2


def test_middle_deletion_splits_two_runs():
    col0 = [A] * NUM_SECTIONS
    for s in (5, 6, 7):
        col0[s] = D
    ps = make_state(col0, [A] * NUM_SECTIONS)
    seg = compute_segment_ids(ps)
    assert (seg[:5, 0] == 0).all(), "상단 run은 0"
    assert (seg[5:8, 0] == -1).all(), "DELETED 칸은 -1"
    assert (seg[8:, 0] == 1).all(), "하단 run은 1"
    assert (seg[:, 1] == 0).all(), "미분할 파트 열은 영향 없음"
    assert count_thickness_groups(seg) == len(CONTINUOUS_PARTS) + 2 + 1


def test_multiple_runs_and_edges():
    # 섹션 0 DELETED 시작, 중간 단일 삭제, 마지막 섹션 DELETED
    col0 = [D, A, A, D, A, A, A, D, A, A, A, A, A, A, A, A, D]
    ps = make_state(col0, [A] * NUM_SECTIONS)
    seg = compute_segment_ids(ps)[:, 0].tolist()
    expected = [-1, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, 2, 2, 2, 2, 2, -1]
    assert seg == expected, f"기대 {expected}, 실제 {seg}"


def test_all_deleted():
    ps = make_state([D] * NUM_SECTIONS, [A] * NUM_SECTIONS)
    seg = compute_segment_ids(ps)
    assert (seg[:, 0] == -1).all()
    assert count_thickness_groups(seg) == len(CONTINUOUS_PARTS) + 0 + 1


def test_run_monotone_no_reuse():
    # run 라벨은 -1 이후 다시 낮은 번호로 돌아가면 안 됨 (연결성 위반) — 랜덤 500케이스
    g = torch.Generator().manual_seed(0)
    for _ in range(500):
        col = torch.where(torch.rand(NUM_SECTIONS, generator=g) < 0.3,
                          torch.tensor(D), torch.tensor(A)).tolist()
        ps = make_state(col, [A] * NUM_SECTIONS)
        seg = compute_segment_ids(ps)[:, 0]
        alive_vals = seg[seg >= 0].tolist()
        # 생존 구간 라벨은 비내림차순이고, 라벨이 바뀌는 지점 사이에는 반드시 -1이 존재
        assert alive_vals == sorted(alive_vals), f"run 라벨 역행: {seg.tolist()}"
        for i in range(1, NUM_SECTIONS):
            if seg[i] >= 0 and seg[i - 1] >= 0:
                assert seg[i] == seg[i - 1], f"인접 생존 칸 라벨 불일치: {seg.tolist()}"


def test_composite_key_no_collision():
    # run_key = 10000 + part_id*(NUM_SECTIONS+1) + seg, dummy = 90000 + part_id — 전수 충돌 검사
    keys = set()
    for pid in CANDIDATE_PARTS:
        for seg in range(0, (NUM_SECTIONS + 1) // 2 + 1):   # 최대 run 수 9
            k = 10_000 + pid * (NUM_SECTIONS + 1) + seg
            assert k not in keys, f"키 충돌: {k}"
            keys.add(k)
        dummy = 90_000 + pid
        assert dummy not in keys
        keys.add(dummy)
    assert not (keys & set(CONTINUOUS_PARTS)), "연속 파트 키와 충돌"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} 테스트 통과")
