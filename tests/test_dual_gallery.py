"""
test_dual_gallery.py — Unit Tests for Candidate-Buffer Dual-Gallery Gait Adaptation
===================================================================================
Project : Gait-Based Biometric Authentication (Phase 3 Redesign)
"""

import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation import GaitWindow
from src.dual_gallery import (
    CandidateCluster,
    CandidateBuffer,
    DualGallery,
)
from src.dtw_authentication import multivariate_dtw_distance


def make_dummy_window(
    shape: tuple = (200, 1),
    offset: float = 0.0,
    scale: float = 1.0,
    participant_id: int = 1,
    window_id: str = "w0",
    segment_id: int = 0,
    window_index: int = 0,
    start_time_rel: float = 0.0,
    end_time_rel: float = 2.0,
    session: int = 2,
    label: str = "G",
) -> GaitWindow:
    """Helper to generate synthetic test gait windows."""
    t = np.linspace(0, 4 * np.pi, shape[0])
    data = (scale * np.sin(t) + offset).reshape(shape)
    return GaitWindow(
        window_id=window_id,
        participant_id=participant_id,
        session=session,
        label=label,
        segment_id=segment_id,
        window_index=window_index,
        start_time_rel=start_time_rel,
        end_time_rel=end_time_rel,
        data=data,
        magnitude=np.linalg.norm(data, axis=1) if data.ndim > 1 else data,
    )


def test_initial_dual_gallery_state():
    """Verify DualGallery initializes with 8 permanent templates and an empty adaptive gallery."""
    base_tmpls = [make_dummy_window(offset=i * 0.01).data for i in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        max_adaptive_capacity=4,
        w_perm=0.70,
    )

    assert len(dg.permanent_templates) == 8
    assert len(dg.adaptive_templates) == 0
    assert dg.compute_current_drift() == 0.0
    assert len(dg.candidate_buffer) == 0

    # Test distance computation with empty adaptive gallery
    test_w = make_dummy_window(offset=0.0)
    d_perm, d_adapt, d_comb = dg.compute_distances(test_w.data)
    assert d_perm == pytest.approx(d_adapt, abs=1e-6)
    assert d_comb == pytest.approx(d_perm, abs=1e-6)


def test_candidate_buffer_temporal_overlap_rejection():
    """Verify candidate buffer rejects overlapping windows and accepts non-overlapping windows."""
    cb = CandidateBuffer(target_size=3)

    w1 = make_dummy_window(window_index=1, start_time_rel=0.0, end_time_rel=2.0)
    w2_overlap = make_dummy_window(window_index=2, start_time_rel=1.0, end_time_rel=3.0)
    w3_non_overlap = make_dummy_window(window_index=3, start_time_rel=2.0, end_time_rel=4.0)

    # First window should be appended
    ok, reason = cb.add_candidate(w1, score=0.03, is_genuine=True)
    assert ok is True
    assert len(cb) == 1

    # Overlapping window (adjacent index=2, dt=1.0 < 2.0) must be rejected
    ok2, reason2 = cb.add_candidate(w2_overlap, score=0.03, is_genuine=True)
    assert ok2 is False
    assert reason2 == "overlapping_window"
    assert len(cb) == 1

    # Non-overlapping window (index=3, start=2.0 >= end=2.0) must be appended
    ok3, reason3 = cb.add_candidate(w3_non_overlap, score=0.03, is_genuine=True)
    assert ok3 is True
    assert len(cb) == 2


def test_candidate_buffer_quarantines_on_low_confidence():
    """Verify that buffer clears when an incoming window fails the high-confidence gate."""
    base_tmpls = [make_dummy_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        w_perm=0.70,
        candidate_target_size=3,
    )

    threshold = 0.10
    alpha = 0.70  # high conf limit = 0.07

    # Add 2 high confidence windows
    w1 = make_dummy_window(scale=1.0, window_index=1, start_time_rel=0.0, end_time_rel=2.0)
    w2 = make_dummy_window(scale=1.0, window_index=3, start_time_rel=2.0, end_time_rel=4.0)

    dg.evaluate_and_update(w1, threshold=threshold, alpha_high_confidence=alpha)
    dg.evaluate_and_update(w2, threshold=threshold, alpha_high_confidence=alpha)
    assert len(dg.candidate_buffer) == 2

    # Third window has distance ~0.08 (> alpha * threshold = 0.07): confidence deteriorates!
    w_low_conf = make_dummy_window(scale=1.45, window_index=5, start_time_rel=4.0, end_time_rel=6.0)
    acc, d_comb, _, _, promoted, meta = dg.evaluate_and_update(
        w_low_conf, threshold=threshold, alpha_high_confidence=alpha
    )

    # Buffer must be quarantined (cleared)
    assert len(dg.candidate_buffer) == 0
    assert promoted is False
    assert meta is not None
    assert meta["event_type"] == "buffer_quarantined_low_conf"
    assert meta["cleared_count"] == 2


def test_medoid_computation_accuracy():
    """Verify CandidateCluster computes medoid correctly as the cluster center."""
    # Create 3 candidates: c0 (scale 0.8), c1 (scale 1.0), c2 (scale 1.2)
    # c1 is central, minimizing sum of distances to c0 and c2
    c0 = make_dummy_window(scale=0.8, window_index=1)
    c1 = make_dummy_window(scale=1.0, window_index=3)
    c2 = make_dummy_window(scale=1.2, window_index=5)

    cluster = CandidateCluster(
        candidates=[c0, c1, c2],
        scores=[0.03, 0.02, 0.03],
        is_genuine_flags=[True, True, True],
    )

    m_idx, medoid_w, is_gen, avg_d = cluster.compute_medoid(dtw_radius=30, norm_mode="sum_lengths")
    assert m_idx == 1
    assert medoid_w == c1
    assert is_gen is True


def test_cluster_inconsistent_discarded():
    """Verify that candidate cluster with excessive pairwise DTW dispersion is discarded."""
    base_tmpls = [make_dummy_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
    )

    threshold = 0.10
    gamma = 0.50  # consistency limit = 0.05

    # 3 candidates that are mutually divergent (scale 0.7, 1.0, 1.3)
    c0 = make_dummy_window(scale=0.7, window_index=1, start_time_rel=0.0, end_time_rel=2.0)
    c1 = make_dummy_window(scale=1.0, window_index=3, start_time_rel=2.0, end_time_rel=4.0)
    c2 = make_dummy_window(scale=1.3, window_index=5, start_time_rel=4.0, end_time_rel=6.0)

    dg.evaluate_and_update(c0, threshold=threshold, gamma_cluster_consistency=gamma)
    dg.evaluate_and_update(c1, threshold=threshold, gamma_cluster_consistency=gamma)
    acc, _, _, _, promoted, meta = dg.evaluate_and_update(
        c2, threshold=threshold, gamma_cluster_consistency=gamma
    )

    # Dispersed cluster should fail consistency check and be discarded
    assert promoted is False
    assert len(dg.adaptive_templates) == 0
    assert len(dg.candidate_buffer) == 0
    assert meta is not None
    assert meta["event_type"] == "cluster_inconsistent_discarded"


def test_medoid_anchor_guard_blocks_outlier():
    """Verify that a candidate cluster diverging from permanent enrollment anchors is blocked."""
    # Permanent templates at scale 1.0
    base_tmpls = [make_dummy_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
    )

    threshold = 0.10
    alpha = 0.80
    beta = 0.30  # strict anchor limit = 0.03

    # Candidates cluster tightly together at scale 1.35, but deviate from base templates (~0.06 > beta*threshold)
    c0 = make_dummy_window(scale=1.35, window_index=1, start_time_rel=0.0, end_time_rel=2.0)
    c1 = make_dummy_window(scale=1.35, window_index=3, start_time_rel=2.0, end_time_rel=4.0)
    c2 = make_dummy_window(scale=1.35, window_index=5, start_time_rel=4.0, end_time_rel=6.0)

    dg.evaluate_and_update(c0, threshold=threshold, alpha_high_confidence=alpha, beta_anchor_guard=beta)
    dg.evaluate_and_update(c1, threshold=threshold, alpha_high_confidence=alpha, beta_anchor_guard=beta)
    acc, _, _, _, promoted, meta = dg.evaluate_and_update(
        c2, threshold=threshold, alpha_high_confidence=alpha, beta_anchor_guard=beta
    )

    assert promoted is False
    assert len(dg.adaptive_templates) == 0
    assert len(dg.candidate_buffer) == 0
    assert meta is not None
    assert meta["event_type"] == "medoid_anchor_guard_rejected"


def test_successful_genuine_cluster_promotion():
    """Verify that consistent high-confidence candidate cluster promotes medoid to adaptive gallery."""
    base_tmpls = [make_dummy_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        max_adaptive_capacity=4,
        candidate_target_size=3,
    )

    threshold = 0.10
    alpha = 0.80
    beta = 1.00
    gamma = 1.00

    c0 = make_dummy_window(scale=1.05, window_index=1, start_time_rel=0.0, end_time_rel=2.0)
    c1 = make_dummy_window(scale=1.06, window_index=3, start_time_rel=2.0, end_time_rel=4.0)
    c2 = make_dummy_window(scale=1.05, window_index=5, start_time_rel=4.0, end_time_rel=6.0)

    dg.evaluate_and_update(c0, threshold=threshold, alpha_high_confidence=alpha, beta_anchor_guard=beta, gamma_cluster_consistency=gamma)
    dg.evaluate_and_update(c1, threshold=threshold, alpha_high_confidence=alpha, beta_anchor_guard=beta, gamma_cluster_consistency=gamma)
    acc, _, _, _, promoted, meta = dg.evaluate_and_update(
        c2, threshold=threshold, alpha_high_confidence=alpha, beta_anchor_guard=beta, gamma_cluster_consistency=gamma
    )

    assert promoted is True
    assert len(dg.adaptive_templates) == 1
    assert len(dg.candidate_buffer) == 0
    assert meta is not None
    assert meta["event_type"] == "candidate_cluster_promoted"
    assert meta["is_poisoned"] is False
    assert dg.compute_current_drift() > 0.0


def test_adaptive_capacity_bounded_and_redundancy_replacement():
    """Verify adaptive gallery capacity is strictly bounded and uses redundancy replacement."""
    base_tmpls = [make_dummy_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        max_adaptive_capacity=2,  # bound capacity to 2
        candidate_target_size=1,  # immediate promotion for unit test
    )

    threshold = 0.10

    # Promote 1st template (scale 1.05)
    w1 = make_dummy_window(scale=1.05, window_index=1, start_time_rel=0.0, end_time_rel=2.0)
    dg.evaluate_and_update(w1, threshold=threshold)
    assert len(dg.adaptive_templates) == 1

    # Promote 2nd template (scale 1.10)
    w2 = make_dummy_window(scale=1.10, window_index=3, start_time_rel=2.0, end_time_rel=4.0)
    dg.evaluate_and_update(w2, threshold=threshold)
    assert len(dg.adaptive_templates) == 2

    # Promote 3rd template (scale 1.06 - closest to scale 1.05)
    w3 = make_dummy_window(scale=1.06, window_index=5, start_time_rel=4.0, end_time_rel=6.0)
    dg.evaluate_and_update(w3, threshold=threshold)
    # Capacity must remain 2
    assert len(dg.adaptive_templates) == 2

    # Permanent templates must remain strictly 8 and unchanged
    assert len(dg.permanent_templates) == 8
    for pt in dg.permanent_templates:
        assert np.allclose(pt, base_tmpls[0])


def test_adaptive_fifo_replacement():
    """Verify adaptive gallery uses FIFO replacement when configured."""
    base_tmpls = [make_dummy_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        max_adaptive_capacity=2,
        candidate_target_size=1,
        replacement_strategy="fifo",
    )

    threshold = 0.10
    # Promote 1st template (scale 1.05)
    w1 = make_dummy_window(scale=1.05, window_index=1, start_time_rel=0.0, end_time_rel=2.0)
    dg.evaluate_and_update(w1, threshold=threshold)

    # Promote 2nd template (scale 1.10)
    w2 = make_dummy_window(scale=1.10, window_index=3, start_time_rel=2.0, end_time_rel=4.0)
    dg.evaluate_and_update(w2, threshold=threshold)

    # Promote 3rd template (scale 1.15) -> FIFO should drop w1 and keep w2, w3
    w3 = make_dummy_window(scale=1.15, window_index=5, start_time_rel=4.0, end_time_rel=6.0)
    dg.evaluate_and_update(w3, threshold=threshold)

    assert len(dg.adaptive_templates) == 2
    # The oldest template (w1, scale 1.05) should have been popped
    assert np.allclose(dg.adaptive_templates[0], w2.data)
    assert np.allclose(dg.adaptive_templates[1], w3.data)


def test_dual_gallery_weighted_scoring():
    """Verify that combined distance combines permanent and adaptive galleries with w_perm."""
    base_tmpls = [make_dummy_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        max_adaptive_capacity=4,
        w_perm=0.70,
    )
    # Add an adaptive template at scale 1.2
    adapt_tmpl = make_dummy_window(scale=1.2).data
    dg.adaptive_templates.append(adapt_tmpl)

    # Test window matching closely with adaptive template
    test_w = make_dummy_window(scale=1.15)
    d_perm, d_adapt, d_comb = dg.compute_distances(test_w.data)

    expected_comb = 0.70 * d_perm + 0.30 * d_adapt
    assert d_comb == pytest.approx(expected_comb, abs=1e-6)
    assert d_adapt < d_perm  # closer to adaptive template
    assert d_comb < d_perm   # combined score benefited from adaptation


def test_impostor_quarantine_blocks_infiltration():
    """Verify that impostor low-confidence attempt flushes genuine candidate accumulation."""
    base_tmpls = [make_dummy_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
    )

    threshold = 0.08
    alpha = 0.70

    # Genuine user accumulates 2 high-confidence windows
    g1 = make_dummy_window(scale=1.0, window_index=1, start_time_rel=0.0, end_time_rel=2.0)
    g2 = make_dummy_window(scale=1.0, window_index=3, start_time_rel=2.0, end_time_rel=4.0)
    dg.evaluate_and_update(g1, threshold=threshold, alpha_high_confidence=alpha)
    dg.evaluate_and_update(g2, threshold=threshold, alpha_high_confidence=alpha)
    assert len(dg.candidate_buffer) == 2

    # Impostor attempt occurs (scale 1.8 -> d_comb ~ 0.15 > threshold):
    imp = make_dummy_window(scale=1.8, participant_id=2, window_index=5, start_time_rel=4.0, end_time_rel=6.0)
    acc, d_comb, _, _, promoted, meta = dg.evaluate_and_update(
        imp, threshold=threshold, alpha_high_confidence=alpha, is_genuine_groundtruth=False
    )

    assert acc is False
    assert promoted is False
    # Candidate buffer must be instantly flushed to prevent cross-claim contamination
    assert len(dg.candidate_buffer) == 0
    assert meta["event_type"] == "buffer_quarantined_low_conf"


def test_max_pairwise_consistency_strictness():
    """Verify that candidate cluster consistency strictly enforces maximum pairwise DTW distance."""
    base_tmpls = [make_dummy_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
    )

    threshold = 0.10
    alpha = 0.80
    # gamma = 0.50 => max allowable pairwise DTW is 0.05
    gamma = 0.50

    # 3 candidates: c0 and c1 are close (~0.02), but c2 is more distant from c0 (~0.07 > 0.05)
    # The mean pairwise distance might be ~0.04 < 0.05, but max pairwise is ~0.07 > 0.05!
    c0 = make_dummy_window(scale=1.00, window_index=1, start_time_rel=0.0, end_time_rel=2.0)
    c1 = make_dummy_window(scale=1.05, window_index=3, start_time_rel=2.0, end_time_rel=4.0)
    c2 = make_dummy_window(scale=1.35, window_index=5, start_time_rel=4.0, end_time_rel=6.0)

    dg.evaluate_and_update(c0, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma)
    dg.evaluate_and_update(c1, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma)
    acc, _, _, _, promoted, meta = dg.evaluate_and_update(
        c2, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma
    )

    # Must be discarded because maximum pairwise distance exceeded gamma * threshold
    assert promoted is False
    assert len(dg.adaptive_templates) == 0
    assert len(dg.candidate_buffer) == 0
    assert meta is not None
    assert meta["event_type"] == "cluster_inconsistent_discarded"


