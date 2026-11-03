"""
test_fair_comparison.py — Unit tests for fair comparison worker functions and stream logic
"""

import numpy as np
import pytest

from src.segmentation import GaitWindow
from src.fair_adaptive_comparison import (
    _worker_precompute_gallery_perm,
    _worker_evaluate_gallery_stream,
)


def _make_dummy_val_window(pid: int, win_idx: int):
    t = np.linspace(0, 4 * np.pi, 400)
    data = np.sin(t).reshape(-1, 1).astype(np.float64)
    return GaitWindow(
        window_id=f"win_{pid}_{win_idx}",
        participant_id=pid,
        session=2,
        label="G",
        segment_id=0,
        window_index=win_idx,
        start_time_rel=float(win_idx * 2),
        end_time_rel=float(win_idx * 2 + 4),
        data=data,
        magnitude=np.squeeze(data),
    )


def test_precompute_worker():
    tmpl = [_make_dummy_val_window(1, 0).data]
    wins = [_make_dummy_val_window(1, i) for i in range(5)]
    win_data = [w.data for w in wins]

    pid, scores = _worker_precompute_gallery_perm((1, tmpl, win_data, 30, "sum_lengths"))
    assert pid == 1
    assert len(scores) == 5
    assert np.all(scores >= 0.0)


def test_stream_worker_fixed():
    tmpl = [_make_dummy_val_window(1, 0).data]
    wins = [_make_dummy_val_window(1, i) for i in range(4)]
    perm_scores = np.array([0.01, 0.02, 0.05, 0.20], dtype=np.float32)

    # 2 genuine (idx 0, 1) and 2 impostor (idx 2, 3)
    stream_meta = [(0, 1, True), (2, 2, False), (1, 1, True), (3, 2, False)]

    pid, res = _worker_evaluate_gallery_stream((
        1,
        "fixed",
        tmpl,
        stream_meta,
        wins,
        perm_scores,
        0.15,
        {},
    ))

    assert pid == 1
    assert res["n_genuine"] == 2
    assert res["n_impostor"] == 2
    assert res["true_accepts"] == 2
    assert res["false_rejects"] == 0
    assert res["false_accepts"] == 1  # idx 2 has score 0.05 <= 0.15
    assert res["total_updates"] == 0


def test_stream_worker_dual_standard():
    tmpl = [_make_dummy_val_window(1, 0).data]
    wins = [_make_dummy_val_window(1, i) for i in range(6)]
    perm_scores = np.full(6, 0.01, dtype=np.float32)

    # 3 genuine attempts non-overlapping
    stream_meta = [(0, 1, True), (2, 1, True), (4, 1, True)]

    pid, res = _worker_evaluate_gallery_stream((
        1,
        "dual_standard",
        tmpl,
        stream_meta,
        wins,
        perm_scores,
        0.15,
        {
            "w_perm": 0.70,
            "max_adaptive_capacity": 4,
            "candidate_target_size": 3,
            "alpha_high_confidence": 0.70,
            "gamma_cluster_consistency": 0.85,
            "beta_anchor_guard": 1.00,
        },
    ))

    assert pid == 1
    assert res["n_genuine"] == 3
    assert res["true_accepts"] == 3
    assert res["total_updates"] == 1
    assert res["genuine_updates"] == 1
    assert res["poisoned_updates"] == 0

