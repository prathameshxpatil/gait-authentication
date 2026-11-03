"""
Unit tests for Corrected Adaptive Stream Validation.
"""

import numpy as np
import pytest

from src.segmentation import GaitWindow
from src.adaptive_stream_validation import (
    build_operational_user_stream,
    _worker_evaluate_user_stream_and_exhaustive,
)


def _make_dummy_window(pid: int, win_idx: int, t_start: float = 0.0) -> GaitWindow:
    rng = np.random.default_rng(pid * 1000 + win_idx)
    data = rng.normal(loc=0.0, scale=1.0, size=(100, 1)).astype(np.float32)
    return GaitWindow(
        window_id=f"win_p{pid}_{win_idx}",
        participant_id=pid,
        session=2,
        label="G",
        segment_id=0,
        window_index=win_idx,
        start_time_rel=t_start,
        end_time_rel=t_start + 4.0,
        data=data,
        magnitude=np.squeeze(data),
    )


def test_build_operational_user_stream_structure():
    """Verify stream length, participant isolation, and chronological ordering."""
    pids = [1, 2, 3, 4, 5]
    val_dict = {
        p: [_make_dummy_window(p, i, t_start=i * 2.0) for i in range(10)]
        for p in pids
    }

    stream = build_operational_user_stream(
        uid=1,
        all_pids=pids,
        val_windows_dict=val_dict,
        seed=42,
    )

    # Must have 10 genuine and 10 impostor = 20 total
    assert len(stream) == 20
    gen_items = [item for item in stream if item[2] is True]
    imp_items = [item for item in stream if item[2] is False]

    assert len(gen_items) == 10
    assert len(imp_items) == 10

    # Genuine must be from uid=1, impostor must NOT be from uid=1
    for w, claimant, is_g, ts in gen_items:
        assert claimant == 1
        assert is_g is True

    for w, claimant, is_g, ts in imp_items:
        assert claimant != 1
        assert is_g is False

    # Timestamps must be sorted in non-decreasing order
    timestamps = [item[3] for item in stream]
    assert timestamps == sorted(timestamps)


def test_build_operational_user_stream_seed_diversity():
    """Verify different seeds produce different arrival orders for impostors."""
    pids = [1, 2, 3, 4, 5]
    val_dict = {
        p: [_make_dummy_window(p, i, t_start=i * 2.0) for i in range(10)]
        for p in pids
    }

    stream_42 = build_operational_user_stream(uid=1, all_pids=pids, val_windows_dict=val_dict, seed=42)
    stream_100 = build_operational_user_stream(uid=1, all_pids=pids, val_windows_dict=val_dict, seed=100)

    ts_42 = [item[3] for item in stream_42]
    ts_100 = [item[3] for item in stream_100]

    assert ts_42 != ts_100


def test_worker_evaluate_user_stream_and_exhaustive_synthetic():
    """Verify worker produces valid operational and exhaustive metrics without mutating permanently."""
    pid = 1
    pids = [1, 2, 3]
    val_dict = {
        p: [_make_dummy_window(p, i, t_start=i * 2.0) for i in range(6)]
        for p in pids
    }
    all_val = []
    for p in pids:
        all_val.extend(val_dict[p])
    val_pids = [w.participant_id for w in all_val]

    perm_tmpls = [val_dict[1][0].data, val_dict[1][1].data]
    stream = build_operational_user_stream(uid=1, all_pids=pids, val_windows_dict=val_dict, seed=42)

    perm_scores = np.array([0.10 if p == 1 else 0.25 for p in val_pids], dtype=np.float32)

    args = (
        pid,
        "dual_standard",
        perm_tmpls,
        stream,
        all_val,
        perm_scores,
        val_pids,
        0.1336,
        {
            "w_perm": 0.70,
            "max_adaptive_capacity": 4,
            "candidate_target_size": 3,
            "alpha_high_confidence": 0.70,
            "gamma_cluster_consistency": 0.85,
            "beta_anchor_guard": 1.00,
        },
    )

    ret_pid, res = _worker_evaluate_user_stream_and_exhaustive(args)
    assert ret_pid == 1
    assert "stream_micro_far_percent" in res or "stream_far_percent" in res
    assert "exh_far_percent" in res
    assert res["exh_n_gen"] == 6
    assert res["exh_n_imp"] == 12
    assert res["stream_n_gen"] == 6
    assert res["stream_n_imp"] == 6
