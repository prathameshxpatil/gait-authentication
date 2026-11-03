"""
test_adaptive_scenarios.py — Comprehensive Unit & Integration Tests for Adaptive Gait Authentication
==================================================================================================
Project : Gait-Based Biometric Authentication
Tasks   : 
  - Genuine stable gait scenario
  - Genuine gradual gait change scenario
  - Impostor entry into candidate buffer & isolation
  - Repeated impostor attempts & buffer flush
  - Mixed genuine and impostor attempts (cluster consistency rejection)
  - Candidate buffer overflow and expiration
  - Rejected candidate cluster recovery
  - Permanent gallery immutability
  - Candidate confirmation bypass prevention
  - Stateful stream processing without resets
"""

import pytest
import numpy as np

from src.segmentation import GaitWindow
from src.dual_gallery import (
    DualGallery,
    CandidateBuffer,
    AdaptiveAuthenticationPipeline,
)


def make_test_window(
    participant_id: int = 1,
    window_id: str = "w0",
    segment_id: int = 0,
    window_index: int = 0,
    start_time_rel: float = 0.0,
    end_time_rel: float = 4.0,
    shape: tuple = (400, 1),
    offset: float = 0.0,
    scale: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> GaitWindow:
    """Helper to generate realistic synthetic gait windows for testing."""
    t = np.linspace(0, 4 * np.pi, shape[0])
    base = scale * np.sin(t) + offset
    if noise_std > 0:
        rng = np.random.default_rng(seed + window_index)
        base += rng.normal(0, noise_std, size=shape[0])
    data = base.reshape(shape).astype(np.float64)
    return GaitWindow(
        window_id=window_id,
        participant_id=participant_id,
        session=2,
        label="G",
        segment_id=segment_id,
        window_index=window_index,
        start_time_rel=start_time_rel,
        end_time_rel=end_time_rel,
        data=data,
        magnitude=np.squeeze(data),
    )


# ---------------------------------------------------------------------------
# Scenario 1: Permanent Gallery Immutability
# ---------------------------------------------------------------------------

def test_permanent_gallery_immutability():
    """Verify permanent templates cannot be overwritten, modified in-place, or reassigned."""
    base_tmpls = [make_test_window(window_index=i, offset=0.01 * i).data for i in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        max_adaptive_capacity=4,
        w_perm=0.70,
    )

    # 1. Permanent templates is a tuple (cannot append, pop, or replace items)
    assert isinstance(dg.permanent_templates, tuple)
    with pytest.raises(TypeError):
        dg.permanent_templates[0] = np.zeros_like(base_tmpls[0])

    # 2. In-place numpy array mutation must raise ValueError because flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        dg.permanent_templates[0][0, 0] = 999.0

    # 3. Verify original template values remain strictly unchanged
    assert dg.permanent_templates[0][0, 0] == pytest.approx(base_tmpls[0][0, 0], abs=1e-7)


# ---------------------------------------------------------------------------
# Scenario 2: Candidate Confirmation Bypass Prevention
# ---------------------------------------------------------------------------

def test_candidate_confirmation_cannot_be_bypassed():
    """Verify that incoming windows cannot update the gallery without full buffer confirmation."""
    base_tmpls = [make_test_window().data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
    )

    threshold = 0.15
    alpha = 0.70

    # Submit 1 high-confidence window (score ~ 0.0)
    w1 = make_test_window(window_index=0, start_time_rel=0.0, end_time_rel=4.0)
    acc1, _, _, _, promoted1, _ = dg.evaluate_and_update(w1, threshold=threshold, alpha_high_confidence=alpha)

    assert acc1 is True
    assert promoted1 is False
    assert len(dg.adaptive_templates) == 0  # Cannot bypass K=3 confirmation!
    assert len(dg.candidate_buffer) == 1

    # Submit a 2nd non-overlapping high-confidence window
    w2 = make_test_window(window_index=2, start_time_rel=4.0, end_time_rel=8.0)
    acc2, _, _, _, promoted2, _ = dg.evaluate_and_update(w2, threshold=threshold, alpha_high_confidence=alpha)

    assert acc2 is True
    assert promoted2 is False
    assert len(dg.adaptive_templates) == 0  # Still 2/3 candidates; not promoted!
    assert len(dg.candidate_buffer) == 2


# ---------------------------------------------------------------------------
# Scenario 3: Genuine User with Stable Gait
# ---------------------------------------------------------------------------

def test_genuine_user_stable_gait_promotion():
    """Verify genuine stable gait smoothly accumulates K=3 candidates and promotes medoid."""
    base_tmpls = [make_test_window().data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
        max_adaptive_capacity=4,
    )

    threshold = 0.15
    alpha = 0.70

    # 3 non-overlapping genuine windows from the same gait pattern
    w1 = make_test_window(window_index=0, start_time_rel=0.0, end_time_rel=4.0)
    w2 = make_test_window(window_index=2, start_time_rel=4.0, end_time_rel=8.0)
    w3 = make_test_window(window_index=4, start_time_rel=8.0, end_time_rel=12.0)

    dg.evaluate_and_update(w1, threshold=threshold, alpha_high_confidence=alpha, is_genuine_groundtruth=True)
    dg.evaluate_and_update(w2, threshold=threshold, alpha_high_confidence=alpha, is_genuine_groundtruth=True)
    acc3, _, _, _, promoted3, meta3 = dg.evaluate_and_update(
        w3, threshold=threshold, alpha_high_confidence=alpha, is_genuine_groundtruth=True
    )

    assert acc3 is True
    assert promoted3 is True
    assert len(dg.adaptive_templates) == 1
    assert len(dg.candidate_buffer) == 0  # Buffer cleared after successful promotion
    assert meta3["event_type"] == "candidate_cluster_promoted"
    assert meta3["is_poisoned"] is False

    # Drift should be minimal since gait is identical/stable
    assert dg.compute_current_drift() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Scenario 4: Genuine User Whose Gait Gradually Changes Across Sessions
# ---------------------------------------------------------------------------

def test_genuine_user_gradual_gait_change():
    """Verify genuine user with mild session drift adapts safely, lowering matching distance."""
    # Permanent gallery at scale 1.0
    base_tmpls = [make_test_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
        max_adaptive_capacity=4,
        w_perm=0.70,
    )

    threshold = 0.15
    alpha = 0.75
    beta = 1.00

    # Session 2 gait has scale 1.05 (mild natural drift)
    w1 = make_test_window(scale=1.05, window_index=0, start_time_rel=0.0, end_time_rel=4.0)
    w2 = make_test_window(scale=1.05, window_index=2, start_time_rel=4.0, end_time_rel=8.0)
    w3 = make_test_window(scale=1.05, window_index=4, start_time_rel=8.0, end_time_rel=12.0)

    # Initial distance before adaptation
    d_perm_init, _, d_comb_init = dg.compute_distances(w1.data)

    dg.evaluate_and_update(w1, threshold=threshold, alpha_high_confidence=alpha, beta_anchor_guard=beta)
    dg.evaluate_and_update(w2, threshold=threshold, alpha_high_confidence=alpha, beta_anchor_guard=beta)
    _, _, _, _, promoted, meta = dg.evaluate_and_update(
        w3, threshold=threshold, alpha_high_confidence=alpha, beta_anchor_guard=beta
    )

    assert promoted is True
    assert len(dg.adaptive_templates) == 1
    assert meta["is_poisoned"] is False

    # After adaptation, combined distance to subsequent 1.05 windows must improve (decrease)
    w_next = make_test_window(scale=1.05, window_index=6, start_time_rel=12.0, end_time_rel=16.0)
    d_perm_after, d_adapt_after, d_comb_after = dg.compute_distances(w_next.data)

    assert d_adapt_after < d_perm_after
    assert d_comb_after < d_comb_init  # Combined distance successfully lowered!


# ---------------------------------------------------------------------------
# Scenario 5: Impostor Attempting to Enter Candidate Buffer
# ---------------------------------------------------------------------------

def test_impostor_cannot_enter_buffer_without_high_confidence():
    """Verify an impostor false-acceptance with distance > alpha * threshold cannot enter buffer."""
    base_tmpls = [make_test_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
    )

    threshold = 0.15
    alpha = 0.70  # high confidence limit = 0.105

    # Impostor attempt with scale 1.60 (distance ~ 0.117, accepted under 0.15, but > 0.105)
    w_imp = make_test_window(participant_id=2, scale=1.60, window_index=0)
    is_acc, d_comb, _, _, promoted, _ = dg.evaluate_and_update(
        w_imp, threshold=threshold, alpha_high_confidence=alpha, is_genuine_groundtruth=False
    )

    assert is_acc is True  # False acceptance
    assert promoted is False
    assert len(dg.candidate_buffer) == 0  # Impostor blocked from buffer!


# ---------------------------------------------------------------------------
# Scenario 6: Repeated Impostor Attempts Flush the Buffer
# ---------------------------------------------------------------------------

def test_repeated_impostors_flush_quarantined_candidates():
    """Verify that an impostor attack immediately flushes any previously buffered genuine candidates."""
    base_tmpls = [make_test_window().data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
    )

    threshold = 0.15
    alpha = 0.70

    # 1 genuine candidate enters buffer
    w_gen = make_test_window(window_index=0, start_time_rel=0.0, end_time_rel=4.0)
    dg.evaluate_and_update(w_gen, threshold=threshold, alpha_high_confidence=alpha)
    assert len(dg.candidate_buffer) == 1

    # Impostor attack with high distance (> alpha * threshold)
    w_imp = make_test_window(participant_id=99, scale=1.8, window_index=1, start_time_rel=4.0, end_time_rel=8.0)
    acc, _, _, _, promoted, meta = dg.evaluate_and_update(
        w_imp, threshold=threshold, alpha_high_confidence=alpha, is_genuine_groundtruth=False
    )

    # Buffer must be immediately flushed to protect the identity
    assert acc is False
    assert len(dg.candidate_buffer) == 0
    assert meta is not None
    assert meta["event_type"] == "buffer_quarantined_low_conf"
    assert meta["cleared_count"] == 1


# ---------------------------------------------------------------------------
# Scenario 7: Mixed Genuine and Impostor Attempts (Cluster Inconsistency)
# ---------------------------------------------------------------------------

def test_mixed_genuine_and_impostor_cluster_rejected():
    """Verify that a candidate buffer with mixed genuine and impostor windows fails consistency check."""
    base_tmpls = [make_test_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
    )

    threshold = 0.15
    alpha = 0.75  # high conf limit = 0.1125
    gamma = 0.45  # Consistency limit = 0.0675

    # Two genuine windows (scale 1.0)
    w_g1 = make_test_window(participant_id=1, scale=1.0, window_index=0, start_time_rel=0.0, end_time_rel=4.0)
    w_g2 = make_test_window(participant_id=1, scale=1.0, window_index=2, start_time_rel=4.0, end_time_rel=8.0)

    # One impostor window with scale 1.45 (distance to scale 1.0 is ~0.081 > 0.0675)
    w_imp = make_test_window(participant_id=2, scale=1.45, window_index=4, start_time_rel=8.0, end_time_rel=12.0)

    dg.evaluate_and_update(w_g1, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma)
    dg.evaluate_and_update(w_g2, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma)
    acc, _, _, _, promoted, meta = dg.evaluate_and_update(
        w_imp, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma, is_genuine_groundtruth=False
    )

    # Inconsistent cluster must be rejected and flushed! Zero poisoning!
    assert promoted is False
    assert len(dg.adaptive_templates) == 0
    assert len(dg.candidate_buffer) == 0
    assert meta["event_type"] == "cluster_inconsistent_discarded"


# ---------------------------------------------------------------------------
# Scenario 8: Candidate Buffer Expiration & Overflow Protection
# ---------------------------------------------------------------------------

def test_candidate_buffer_expiration_and_overflow():
    """Verify that stale candidates expire and buffer bounds overflow cleanly."""
    buf = CandidateBuffer(target_size=3, max_age_steps=5, max_capacity=4)

    # Candidate at step 1
    w1 = make_test_window(window_index=0, start_time_rel=0.0, end_time_rel=4.0)
    buf.add_candidate(w1, score=0.04, step=1, is_genuine=True)
    assert len(buf) == 1

    # At step 3, candidate is 2 steps old (<= 5), not expired
    exp1 = buf.expire_old_candidates(current_step=3)
    assert exp1 == 0
    assert len(buf) == 1

    # At step 10, candidate is 9 steps old (> 5), must be expired
    exp2 = buf.expire_old_candidates(current_step=10)
    assert exp2 == 1
    assert len(buf) == 0

    # Test overflow: max_capacity = 4
    for i in range(6):
        w = make_test_window(window_index=i * 2, start_time_rel=float(i * 4), end_time_rel=float((i + 1) * 4))
        buf.add_candidate(w, score=0.04, step=10 + i, is_genuine=True)
    # Must not exceed max_capacity of 4
    assert len(buf) == 4


# ---------------------------------------------------------------------------
# Scenario 9: Rejected Cluster Recovery
# ---------------------------------------------------------------------------

def test_rejected_cluster_clean_recovery():
    """Verify that after an inconsistent cluster is discarded, subsequent clean genuine gait recovers and updates."""
    base_tmpls = [make_test_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
        max_adaptive_capacity=4,
    )

    threshold = 0.15
    alpha = 0.85
    gamma = 0.50

    # 1. Trigger inconsistent cluster rejection
    w_bad1 = make_test_window(scale=0.7, window_index=0, start_time_rel=0.0, end_time_rel=4.0)
    w_bad2 = make_test_window(scale=1.0, window_index=2, start_time_rel=4.0, end_time_rel=8.0)
    w_bad3 = make_test_window(scale=1.4, window_index=4, start_time_rel=8.0, end_time_rel=12.0)

    dg.evaluate_and_update(w_bad1, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma)
    dg.evaluate_and_update(w_bad2, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma)
    _, _, _, _, promoted_bad, meta_bad = dg.evaluate_and_update(
        w_bad3, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma
    )

    assert promoted_bad is False
    assert meta_bad["event_type"] == "cluster_inconsistent_discarded"
    assert len(dg.candidate_buffer) == 0

    # 2. Recovery: genuine consistent walking arrives
    w_good1 = make_test_window(scale=1.0, window_index=6, start_time_rel=12.0, end_time_rel=16.0)
    w_good2 = make_test_window(scale=1.0, window_index=8, start_time_rel=16.0, end_time_rel=20.0)
    w_good3 = make_test_window(scale=1.0, window_index=10, start_time_rel=20.0, end_time_rel=24.0)

    dg.evaluate_and_update(w_good1, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma)
    dg.evaluate_and_update(w_good2, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma)
    _, _, _, _, promoted_good, meta_good = dg.evaluate_and_update(
        w_good3, threshold=threshold, alpha_high_confidence=alpha, gamma_cluster_consistency=gamma
    )

    # Clean cluster must successfully promote
    assert promoted_good is True
    assert len(dg.adaptive_templates) == 1
    assert meta_good["event_type"] == "candidate_cluster_promoted"


# ---------------------------------------------------------------------------
# Scenario 10: Stateful Stream Processing without Reset
# ---------------------------------------------------------------------------

def test_stateful_stream_pipeline_processing():
    """Verify AdaptiveAuthenticationPipeline maintains state across an extended chronological stream."""
    base_tmpls = [make_test_window(scale=1.0).data for _ in range(8)]
    dg = DualGallery(
        participant_id=1,
        permanent_templates=base_tmpls,
        candidate_target_size=3,
        max_adaptive_capacity=4,
    )

    pipeline = AdaptiveAuthenticationPipeline(
        gallery=dg,
        threshold=0.15,
        alpha_high_confidence=0.75,
        beta_anchor_guard=1.00,
        gamma_cluster_consistency=0.85,
    )

    # Construct chronological stream of 10 genuine attempts and 2 impostor attempts
    stream = []
    # 3 genuine attempts (triggers 1st promotion)
    for i in range(3):
        w = make_test_window(participant_id=1, scale=1.02, window_index=i * 2, start_time_rel=i * 4.0, end_time_rel=(i + 1) * 4.0)
        stream.append((w, 1, True))

    # 1 impostor attempt (fails high confidence, cannot promote)
    w_imp = make_test_window(participant_id=2, scale=1.40, window_index=6, start_time_rel=12.0, end_time_rel=16.0)
    stream.append((w_imp, 2, False))

    # 3 more genuine attempts (triggers 2nd promotion)
    for i in range(4, 7):
        w = make_test_window(participant_id=1, scale=1.04, window_index=i * 2, start_time_rel=i * 4.0, end_time_rel=(i + 1) * 4.0)
        stream.append((w, 1, True))

    results = pipeline.authenticate_stream(stream)

    assert len(results) == 7
    assert pipeline.step == 7

    # Check audit summary
    summary = pipeline.get_summary()
    assert summary["total_auth_attempts"] == 7
    assert summary["total_promotions"] == 2
    assert summary["genuine_promotions"] == 2
    assert summary["poisoned_promotions"] == 0
    assert summary["active_adaptive_templates"] == 2
    assert len(dg.permanent_templates) == 8  # Permanent templates untouched!
