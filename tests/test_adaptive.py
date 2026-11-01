"""
test_adaptive.py — Unit Tests for Adaptive Template Updating and Poisoning Protection
======================================================================================
Project : Gait-Based Biometric Authentication (Phase 3)

Covers:
  - Gallery initialization and enrollment preservation
  - Rejected sample protection
  - High-confidence and low-confidence update gating
  - Anchor drift guard
  - Impostor rejection
  - Capacity bounding
  - Multi-attempt confirmation counter (NEW)
  - Confirmation reset on confidence failure (NEW)
  - Non-overlapping window requirement (NEW)
  - Aggregate anchor verification: top2_mean (NEW)
  - Rejected updates never modify confirmation state (NEW)

Run with:
    pytest tests/test_adaptive.py -v
"""

import os
import sys
import numpy as np
import pytest

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.segmentation import GaitWindow
from src.adaptive_template import AdaptiveGallery, are_windows_overlapping


def make_dummy_window(
    data: np.ndarray,
    window_id: str = "w0",
    is_genuine: bool = True,
    segment_id: int = 0,
    window_index: int = 0,
    start_time_rel: float = 0.0,
    end_time_rel: float = 2.0,
    session: int = 2,
    label: str = "G",
) -> GaitWindow:
    """Helper to construct a dummy GaitWindow with full metadata control."""
    return GaitWindow(
        window_id=window_id,
        participant_id=1 if is_genuine else 2,
        session=session,
        label=label,
        segment_id=segment_id,
        window_index=window_index,
        start_time_rel=start_time_rel,
        end_time_rel=end_time_rel,
        data=data,
        magnitude=np.linalg.norm(data, axis=1) if data.ndim > 1 else data,
    )


# ===========================================================================
# SECTION 1: Original Tests (preserved)
# ===========================================================================

def test_initial_gallery_state():
    """Verify that AdaptiveGallery initializes with correct anchor and dynamic slot counts."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    initial_templates = [t + i * 0.01 for i in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
    )

    assert len(ag.anchors) == 4
    assert len(ag.dynamic_slots) == 4
    assert len(ag.current_templates) == 8
    assert ag.compute_current_drift() == pytest.approx(0.0, abs=0.01)


def test_rejected_sample_never_updates():
    """Verify that any sample where distance > threshold NEVER modifies the gallery."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    initial_templates = [t + i * 0.01 for i in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
    )

    diff_signal = np.ones((200, 1)) * 50.0
    w_reject = make_dummy_window(diff_signal, "w_bad")

    threshold = 0.10
    is_acc, score, updated = ag.evaluate_and_update(
        w_reject, threshold=threshold, alpha_high_confidence=0.70
    )

    assert not is_acc, "High-distance sample must be rejected"
    assert not updated, "Rejected sample must NEVER update gallery"
    assert len(ag.dynamic_slots) == 4


def test_accepted_high_confidence_triggers_update():
    """Verify that an authentic sample meeting the high-confidence threshold updates the gallery
    when required_confirmations=1 (default backward-compatible behavior)."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    initial_templates = [t + i * 0.01 for i in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
        required_confirmations=1,  # single-attempt mode
    )

    close_signal = t + 0.002
    w_close = make_dummy_window(close_signal, "w_good")

    threshold = 0.10
    is_acc, score, updated = ag.evaluate_and_update(
        w_close, threshold=threshold, alpha_high_confidence=0.85
    )

    assert is_acc, "Close sample must be accepted"
    assert score <= 0.85 * threshold, "Sample should be high confidence"
    assert updated, "High-confidence genuine accept must update the gallery"
    assert ag.genuine_updates_count == 1
    assert ag.poisoned_updates_count == 0


def test_accepted_low_confidence_never_updates():
    """Verify that a borderline accepted sample (below threshold, but above alpha*threshold) does NOT update."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    s = np.sin(t)
    initial_templates = [s.copy() for _ in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
    )

    w_mod = make_dummy_window(1.4 * s, "w_mod")

    threshold = 0.10
    alpha = 0.50
    is_acc, score, updated = ag.evaluate_and_update(
        w_mod, threshold=threshold, alpha_high_confidence=alpha
    )

    assert is_acc, "Should be accepted by general threshold"
    assert not updated, "Borderline sample failing high-confidence gate must NOT update"
    assert ag.genuine_updates_count == 0


def test_anchor_drift_guard_blocks_excessive_drift():
    """Verify that a candidate that is too far from original enrollment anchors is blocked from updating."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    s = np.sin(t)
    initial_templates = [s.copy() for _ in range(4)] + [1.3 * s for _ in range(4)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
    )

    w_drift = make_dummy_window(1.3 * s, "w_drift")

    threshold = 0.10
    beta_guard = 0.40
    is_acc, score, updated = ag.evaluate_and_update(
        w_drift, threshold=threshold, alpha_high_confidence=0.85, beta_anchor_guard=beta_guard
    )

    assert is_acc, "Candidate matching dynamic slot should be accepted"
    assert not updated, "Anchor drift guard must block candidate exceeding anchor distance limit"


def test_impostor_rejection_blocks_poisoning():
    """Verify that simulated impostor attempts are rejected and cannot modify gallery."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    initial_templates = [t + i * 0.01 for i in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
    )

    imp_signal = np.sin(10 * t)
    w_imp = make_dummy_window(imp_signal, "w_impostor", is_genuine=False)

    threshold = 0.10
    is_acc, score, updated = ag.evaluate_and_update(
        w_imp, threshold=threshold, alpha_high_confidence=0.70, is_genuine_groundtruth=False
    )

    assert not is_acc, "Impostor must be rejected"
    assert not updated, "Rejected impostor must NOT modify gallery"
    assert ag.poisoned_updates_count == 0


def test_capacity_is_strictly_bounded():
    """Verify that repeated updates never exceed maximum gallery capacity."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    initial_templates = [t + i * 0.01 for i in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
    )

    threshold = 0.20
    for i in range(25):
        sample = t + 0.001 * (i + 1)
        w = make_dummy_window(sample, f"w_{i}")
        ag.evaluate_and_update(w, threshold=threshold, alpha_high_confidence=0.90)

    assert len(ag.anchors) == 4, "Permanent anchors must remain intact"
    assert len(ag.dynamic_slots) == 4, "Dynamic slots must not exceed 4"
    assert len(ag.current_templates) == 8, "Total gallery must strictly equal max_capacity"


# ===========================================================================
# SECTION 2: Window Overlap Detection Tests
# ===========================================================================

def test_windows_overlapping_adjacent_indices():
    """Adjacent window indices (diff=1) with same segment/label/session ARE overlapping (50% overlap)."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    w1 = make_dummy_window(t, "w0", segment_id=0, window_index=5, start_time_rel=5.0, end_time_rel=7.0)
    w2 = make_dummy_window(t, "w1", segment_id=0, window_index=6, start_time_rel=6.0, end_time_rel=8.0)
    assert are_windows_overlapping(w1, w2) is True


def test_windows_not_overlapping_separated_indices():
    """Windows with index diff >= 2 with same segment/label/session are NOT overlapping."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    w1 = make_dummy_window(t, "w0", segment_id=0, window_index=3, start_time_rel=3.0, end_time_rel=5.0)
    w2 = make_dummy_window(t, "w1", segment_id=0, window_index=5, start_time_rel=5.0, end_time_rel=7.0)
    assert are_windows_overlapping(w1, w2) is False


def test_windows_not_overlapping_different_segments():
    """Windows from different segments are NEVER overlapping regardless of indices."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    w1 = make_dummy_window(t, "w0", segment_id=0, window_index=5)
    w2 = make_dummy_window(t, "w1", segment_id=1, window_index=5)
    assert are_windows_overlapping(w1, w2) is False


def test_windows_not_overlapping_none_input():
    """None input always returns False."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    w1 = make_dummy_window(t, "w0")
    assert are_windows_overlapping(None, w1) is False
    assert are_windows_overlapping(w1, None) is False
    assert are_windows_overlapping(None, None) is False


# ===========================================================================
# SECTION 3: Multi-Attempt Confirmation Gate Tests
# ===========================================================================

def test_three_confirmation_requires_three_attempts():
    """Gallery with required_confirmations=3 must NOT update on first or second high-confidence attempt."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    initial_templates = [t + i * 0.01 for i in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
        required_confirmations=3,
    )

    threshold = 0.20
    alpha = 0.90

    # Attempt 1: high-confidence, non-overlapping (window_index=0)
    w1 = make_dummy_window(t + 0.001, "w1", window_index=0, start_time_rel=0.0, end_time_rel=2.0)
    acc1, _, upd1 = ag.evaluate_and_update(w1, threshold=threshold, alpha_high_confidence=alpha)
    assert acc1, "Should be accepted"
    assert not upd1, "First confirmation alone must NOT trigger update"
    assert ag.genuine_updates_count == 0

    # Attempt 2: high-confidence, non-overlapping (window_index=2, gap of 1 window)
    w2 = make_dummy_window(t + 0.002, "w2", window_index=2, start_time_rel=2.0, end_time_rel=4.0)
    acc2, _, upd2 = ag.evaluate_and_update(w2, threshold=threshold, alpha_high_confidence=alpha)
    assert acc2, "Should be accepted"
    assert not upd2, "Two confirmations is not enough for 3-confirmation gate"
    assert ag.genuine_updates_count == 0

    # Attempt 3: high-confidence, non-overlapping (window_index=4)
    w3 = make_dummy_window(t + 0.003, "w3", window_index=4, start_time_rel=4.0, end_time_rel=6.0)
    acc3, _, upd3 = ag.evaluate_and_update(w3, threshold=threshold, alpha_high_confidence=alpha)
    assert acc3, "Should be accepted"
    assert upd3, "Third non-overlapping confirmation MUST trigger update"
    assert ag.genuine_updates_count == 1


def test_confirmation_resets_on_confidence_failure():
    """Any attempt failing the high-confidence gate must reset the confirmation counter to zero."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    s = np.sin(t)
    initial_templates = [s.copy() for _ in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
        required_confirmations=3,
    )

    threshold = 0.10
    alpha = 0.90  # update threshold = 0.09

    # Two high-confidence attempts (close to templates)
    w1 = make_dummy_window(s + 0.001, "w1", window_index=0, start_time_rel=0.0, end_time_rel=2.0)
    ag.evaluate_and_update(w1, threshold=threshold, alpha_high_confidence=alpha)
    assert len(ag.confirmation_buffer) == 1

    w2 = make_dummy_window(s + 0.002, "w2", window_index=2, start_time_rel=2.0, end_time_rel=4.0)
    ag.evaluate_and_update(w2, threshold=threshold, alpha_high_confidence=alpha)
    assert len(ag.confirmation_buffer) == 2

    # Low-confidence attempt (borderline, accepted but not high-confidence)
    # 1.5 * s has distance ~0.097, which is > alpha * threshold = 0.09, but < threshold = 0.10
    w_low = make_dummy_window(1.5 * s, "w_low", window_index=4, start_time_rel=4.0, end_time_rel=6.0)
    acc_low, score_low, upd_low = ag.evaluate_and_update(w_low, threshold=threshold, alpha_high_confidence=alpha)

    # It should be accepted (distance < threshold) but fail high-confidence gate, resetting counter
    assert acc_low, "Should still be accepted by the general threshold"
    assert not upd_low, "Must not update"
    assert len(ag.confirmation_buffer) == 0, "Confirmation buffer must be reset after confidence failure"

    # Now try 3 more high-confidence attempts — all must restart from zero
    w3 = make_dummy_window(s + 0.003, "w3", window_index=6, start_time_rel=6.0, end_time_rel=8.0)
    ag.evaluate_and_update(w3, threshold=threshold, alpha_high_confidence=alpha)
    assert len(ag.confirmation_buffer) == 1, "Should restart from 1"
    assert ag.genuine_updates_count == 0, "No update yet"


def test_overlapping_windows_do_not_count_as_confirmations():
    """Adjacent overlapping windows (index diff = 1) must NOT count as separate confirmations."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    initial_templates = [t + i * 0.01 for i in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
        required_confirmations=3,
    )

    threshold = 0.20
    alpha = 0.90

    # Attempt 1: window_index=0
    w1 = make_dummy_window(t + 0.001, "w1", window_index=0, start_time_rel=0.0, end_time_rel=2.0)
    ag.evaluate_and_update(w1, threshold=threshold, alpha_high_confidence=alpha)
    assert len(ag.confirmation_buffer) == 1

    # Attempt 2: window_index=1 (OVERLAPPING with window_index=0 at 50% overlap)
    w2 = make_dummy_window(t + 0.001, "w2", window_index=1, start_time_rel=1.0, end_time_rel=3.0)
    ag.evaluate_and_update(w2, threshold=threshold, alpha_high_confidence=alpha)
    assert len(ag.confirmation_buffer) == 1, "Overlapping window must not increment confirmation count"

    # Attempt 3: window_index=2 (NOT overlapping with window_index=1, but buffer only has w1)
    w3 = make_dummy_window(t + 0.001, "w3", window_index=2, start_time_rel=2.0, end_time_rel=4.0)
    ag.evaluate_and_update(w3, threshold=threshold, alpha_high_confidence=alpha)
    # w3 does NOT overlap with w1 (last in buffer), so it counts
    assert len(ag.confirmation_buffer) == 2

    # No update yet (need 3)
    assert ag.genuine_updates_count == 0


def test_rejected_sample_resets_confirmation_counter():
    """A rejected sample (distance > threshold) must reset the confirmation counter."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    initial_templates = [t + i * 0.01 for i in range(8)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
        required_confirmations=3,
    )

    threshold = 0.10
    alpha = 0.90

    # Two valid confirmations
    w1 = make_dummy_window(t + 0.001, "w1", window_index=0, start_time_rel=0.0, end_time_rel=2.0)
    ag.evaluate_and_update(w1, threshold=threshold, alpha_high_confidence=alpha)
    w2 = make_dummy_window(t + 0.001, "w2", window_index=2, start_time_rel=2.0, end_time_rel=4.0)
    ag.evaluate_and_update(w2, threshold=threshold, alpha_high_confidence=alpha)
    assert len(ag.confirmation_buffer) == 2

    # Rejection (impostor signal, very different)
    w_bad = make_dummy_window(np.ones((200, 1)) * 50.0, "w_bad", window_index=4)
    acc, _, upd = ag.evaluate_and_update(w_bad, threshold=threshold, alpha_high_confidence=alpha)
    assert not acc, "Must be rejected"
    assert not upd, "Must not update"
    assert len(ag.confirmation_buffer) == 0, "Counter must be reset after rejection"


# ===========================================================================
# SECTION 4: Aggregate Anchor Verification Tests
# ===========================================================================

def test_top2_mean_anchor_aggregation():
    """Verify top2_mean anchor aggregation returns mean of 2 closest anchor distances."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    s = np.sin(t)

    # 4 anchors at base amplitude, 4 dynamic at 1.3x
    initial_templates = [s.copy() for _ in range(4)] + [1.3 * s for _ in range(4)]

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
        anchor_aggregation="top2_mean",
    )

    # Compute distances to anchors using different aggregation methods
    test_data = 1.1 * s  # slightly shifted
    min_dist = ag.compute_anchor_aggregate_distance(test_data, aggregation="min")
    top2_dist = ag.compute_anchor_aggregate_distance(test_data, aggregation="top2_mean")
    mean_dist = ag.compute_anchor_aggregate_distance(test_data, aggregation="mean")

    # top2_mean should be >= min and <= mean (it averages the 2 closest)
    assert top2_dist >= min_dist, "top2_mean must be >= min anchor distance"
    assert top2_dist <= mean_dist, "top2_mean must be <= mean anchor distance"


def test_top2_mean_blocks_single_anchor_outlier():
    """A candidate close to ONE outlier anchor but far from others should fail top2_mean check."""
    t = np.linspace(0, 4 * np.pi, 200)[:, None]
    s = np.sin(t)

    # Create anchors: 3 at base amplitude + 1 outlier at 1.5x amplitude
    anchors = [s.copy(), s.copy(), s.copy(), 1.5 * s]
    dyn_slots = [s.copy() for _ in range(4)]
    initial_templates = anchors + dyn_slots

    ag = AdaptiveGallery(
        participant_id=1,
        initial_templates=initial_templates,
        n_permanent_anchors=4,
        max_capacity=8,
        anchor_aggregation="top2_mean",
    )

    # Candidate at 1.5x matches the outlier anchor exactly, but is far from the 3 base anchors
    candidate = 1.5 * s
    min_dist = ag.compute_anchor_aggregate_distance(candidate, aggregation="min")
    top2_dist = ag.compute_anchor_aggregate_distance(candidate, aggregation="top2_mean")

    # min might be ~0 (matches outlier), but top2_mean averages with next closest (base anchor, ~0.097)
    assert top2_dist > min_dist + 0.01, "top2_mean should be noticeably larger than min for outlier match"
