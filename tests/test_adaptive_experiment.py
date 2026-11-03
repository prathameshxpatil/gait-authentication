"""
test_adaptive_experiment.py — Unit tests for modular adaptive strategies and candidate buffer
"""

import numpy as np
import pytest

from src.segmentation import GaitWindow
from src.adaptive_experiment import (
    CandidateBuffer,
    FixedBaselineStrategy,
    NaiveAdaptiveStrategy,
    HighConfSingleStrategy,
    ControlledDualGalleryStrategy,
    build_validation_stream_for_user,
)


def _make_dummy_window(pid: int, session: int, seg_id: int, win_idx: int, t_start: float, t_end: float, offset: float = 0.0):
    t = np.linspace(0, 4 * np.pi, 400)
    data = (np.sin(t) + offset).reshape(-1, 1).astype(np.float64)
    return GaitWindow(
        window_id=f"win_p{pid}_s{session}_seg{seg_id}_{win_idx}",
        participant_id=pid,
        session=session,
        label="G",
        segment_id=seg_id,
        window_index=win_idx,
        start_time_rel=t_start,
        end_time_rel=t_end,
        data=data,
        magnitude=np.squeeze(data),
    )


def test_candidate_buffer_temporal_independence():
    buf = CandidateBuffer(target_size=3)
    w1 = _make_dummy_window(pid=1, session=2, seg_id=0, win_idx=0, t_start=0.0, t_end=4.0)
    w2 = _make_dummy_window(pid=1, session=2, seg_id=0, win_idx=1, t_start=2.0, t_end=6.0) # overlaps w1
    w3 = _make_dummy_window(pid=1, session=2, seg_id=0, win_idx=2, t_start=4.0, t_end=8.0) # disjoint w1

    appended1, _ = buf.add_candidate(w1, score=0.05, is_genuine=True)
    assert appended1 is True
    assert len(buf) == 1

    # Attempt to add overlapping window
    appended2, reason = buf.add_candidate(w2, score=0.04, is_genuine=True)
    assert appended2 is False
    assert reason == "overlapping_window"
    assert len(buf) == 1

    # Add non-overlapping window
    appended3, _ = buf.add_candidate(w3, score=0.05, is_genuine=True)
    assert appended3 is True
    assert len(buf) == 2


def test_fixed_baseline_never_updates():
    w_template = _make_dummy_window(pid=1, session=1, seg_id=0, win_idx=0, t_start=0.0, t_end=4.0)
    strat = FixedBaselineStrategy(participant_id=1, permanent_templates=[w_template.data])

    w_attempt = _make_dummy_window(pid=1, session=2, seg_id=0, win_idx=0, t_start=0.0, t_end=4.0)
    res = strat.evaluate_attempt(step=1, window=w_attempt, claimant_id=1, is_genuine=True, threshold=0.15)

    assert res.accepted is True
    assert res.updated is False
    assert strat.get_drift() == 0.0


def test_naive_adaptive_updates_and_poisons():
    w_template = _make_dummy_window(pid=1, session=1, seg_id=0, win_idx=0, t_start=0.0, t_end=4.0)
    strat = NaiveAdaptiveStrategy(participant_id=1, permanent_templates=[w_template.data])

    # Impostor attempt with similar gait
    w_imp = _make_dummy_window(pid=2, session=2, seg_id=0, win_idx=0, t_start=0.0, t_end=4.0, offset=0.05)
    res = strat.evaluate_attempt(step=1, window=w_imp, claimant_id=2, is_genuine=False, threshold=0.20)

    assert res.accepted is True
    assert res.updated is True
    assert res.is_poisoned is True
    assert strat.get_drift() > 0.0


def test_controlled_dual_gallery_blocks_single_impostor():
    w_template = _make_dummy_window(pid=1, session=1, seg_id=0, win_idx=0, t_start=0.0, t_end=4.0)
    strat = ControlledDualGalleryStrategy(
        name="test_dual",
        participant_id=1,
        permanent_templates=[w_template.data],
        w_perm=0.70,
        max_adaptive_capacity=4,
        candidate_target_size=3,
        alpha_high_confidence=0.70,
        gamma_cluster_consistency=0.85,
        beta_anchor_guard=1.00,
    )

    # Single impostor false acceptance should NOT update gallery (needs K=3 confirmations)
    w_imp = _make_dummy_window(pid=2, session=2, seg_id=0, win_idx=0, t_start=0.0, t_end=4.0, offset=0.02)
    res = strat.evaluate_attempt(step=1, window=w_imp, claimant_id=2, is_genuine=False, threshold=0.20)

    # Even if accepted, it should NOT have updated because buffer requires 3 candidates
    assert res.updated is False
    assert res.is_poisoned is False
    assert len(strat.adaptive_templates) == 0
