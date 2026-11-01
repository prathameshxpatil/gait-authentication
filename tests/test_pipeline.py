"""
test_pipeline.py — Unit Tests for Gait Preprocessing and Multivariate DTW
==========================================================================
Project : Gait-Based Biometric Authentication (Phase 2)

Run tests with:
    pytest tests/test_pipeline.py -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    butter_lowpass_filter,
    split_at_discontinuities,
    preprocess_stream_segment,
)
from src.segmentation import (
    GaitWindow,
    extract_windows_from_segment,
    partition_enrolment_and_test,
)
from src.experiments import partition_enrol_val_test
from src.dtw_authentication import (
    multivariate_dtw_distance,
    build_enrolment_galleries,
    compute_verification_score,
)
from src.evaluation import compute_far_frr_curve


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

def test_butter_lowpass_filter_attenuates_high_frequencies():
    """Verify that the 20 Hz lowpass filter attenuates a 40 Hz component while preserving 2 Hz."""
    fs = 100.0
    t = np.arange(0, 3.0, 1.0 / fs)  # 3 seconds @ 100 Hz
    # Signal: 2 Hz gait-like sine wave + 40 Hz high-frequency noise
    gait_freq = 2.0
    noise_freq = 40.0
    signal_low = np.sin(2 * np.pi * gait_freq * t)
    signal_high = 0.5 * np.sin(2 * np.pi * noise_freq * t)
    combined = signal_low + signal_high

    filtered = butter_lowpass_filter(combined, cutoff=20.0, fs=fs, order=4)

    assert len(filtered) == len(combined)
    # The high frequency noise variance should be substantially reduced
    noise_power_orig = np.var(signal_high)
    residual_diff = np.var(filtered - signal_low)
    assert residual_diff < noise_power_orig * 0.1, "Filter should attenuate >90% of 40Hz noise"


def test_split_at_discontinuities_no_interpolation():
    """Verify that a recording with an intentional 2.0s gap is split into 2 continuous segments."""
    # Segment 1: t = 0 to 1s (100 samples)
    t1 = np.arange(0, 1.0, 0.010)
    # 2.0-second gap: next sample is at t = 3.0s
    t2 = np.arange(3.0, 4.0, 0.010)

    timestamps = np.concatenate([t1, t2])
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "X": np.random.randn(len(timestamps)),
        "Y": np.random.randn(len(timestamps)),
        "Z": np.random.randn(len(timestamps)),
        "Label": ["G"] * len(timestamps),
    })

    segments = split_at_discontinuities(df, timestamp_col="Timestamp", gap_threshold_sec=0.5)

    assert len(segments) == 2, "Should split into exactly 2 continuous segments"
    assert len(segments[0]) == len(t1)
    assert len(segments[1]) == len(t2)
    # Verify no synthetic samples were interpolated
    assert len(segments[0]) + len(segments[1]) == len(df)


def test_timestamp_normalization():
    """Verify that relative elapsed time starts at t=0.0."""
    t_raw = np.array([271130.5, 271130.51, 271130.52, 271130.53, 271130.54])
    df = pd.DataFrame({
        "Timestamp": t_raw,
        "X": [0.1, 0.2, 0.1, 0.2, 0.1],
        "Y": [-0.8, -0.8, -0.8, -0.8, -0.8],
        "Z": [0.0, 0.1, 0.0, 0.1, 0.0],
    })
    processed = preprocess_stream_segment(
        df, participant_id=1, session=1, label="G", segment_id=0
    )
    assert processed["time_rel"].iloc[0] == pytest.approx(0.0)
    assert processed["time_rel"].iloc[-1] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Segmentation Tests
# ---------------------------------------------------------------------------

def test_sliding_window_extraction_and_discard_incomplete():
    """Test 200-sample windowing with 50% overlap and incomplete window discard."""
    # 450 samples -> floor((450 - 200)/100) + 1 = 3 windows of length 200
    n_samples = 450
    t = np.arange(n_samples) * 0.010
    df = pd.DataFrame({
        "Timestamp": t,
        "time_rel": t,
        "X": np.zeros(n_samples),
        "Y": np.ones(n_samples),
        "Z": np.zeros(n_samples),
        "a_mag": np.ones(n_samples),
        "participant_id": [1] * n_samples,
        "session": [1] * n_samples,
        "label": ["G"] * n_samples,
        "segment_id": [0] * n_samples,
    })

    windows = extract_windows_from_segment(df, window_size=200, step_size=100)
    assert len(windows) == 3
    for w in windows:
        assert w.data.shape == (200, 3)
        assert len(w.magnitude) == 200


def test_zero_leakage_partitioning():
    """Verify strict partition: Session 1 in enrolment, Session 2 in test."""
    w1 = GaitWindow("w1", participant_id=1, session=1, label="G", segment_id=0,
                    window_index=0, start_time_rel=0.0, end_time_rel=2.0,
                    data=np.zeros((200, 3)), magnitude=np.zeros(200))
    w2 = GaitWindow("w2", participant_id=1, session=2, label="G", segment_id=0,
                    window_index=0, start_time_rel=0.0, end_time_rel=2.0,
                    data=np.zeros((200, 3)), magnitude=np.zeros(200))

    enrol, test = partition_enrolment_and_test([w1, w2])
    assert w1 in enrol[1]
    assert w2 not in enrol[1]
    assert w2 in test[1]
    assert w1 not in test[1]


def test_partition_enrol_val_test_discards_boundary_overlap():
    """Verify that the first test window is discarded when it shares samples with the final validation window."""
    dummy_data = np.zeros((200, 3))
    dummy_mag = np.zeros(200)

    # 4 windows from the same continuous segment with 50% overlap (step 1.0s, duration 2.0s)
    w_val0 = GaitWindow("val0", participant_id=1, session=2, label="G", segment_id=0,
                        window_index=0, start_time_rel=0.0, end_time_rel=2.0,
                        data=dummy_data, magnitude=dummy_mag)
    w_val1 = GaitWindow("val1", participant_id=1, session=2, label="G", segment_id=0,
                        window_index=1, start_time_rel=1.0, end_time_rel=3.0,
                        data=dummy_data, magnitude=dummy_mag)
    w_test_overlap = GaitWindow("test_overlap", participant_id=1, session=2, label="G", segment_id=0,
                                window_index=2, start_time_rel=2.0, end_time_rel=4.0,
                                data=dummy_data, magnitude=dummy_mag)
    w_test_clean = GaitWindow("test_clean", participant_id=1, session=2, label="G", segment_id=0,
                              window_index=3, start_time_rel=3.0, end_time_rel=5.0,
                              data=dummy_data, magnitude=dummy_mag)

    enrol, val, test = partition_enrol_val_test([w_val0, w_val1, w_test_overlap, w_test_clean], val_ratio=0.50)

    # Validation should keep exactly the first 2 windows (split_point = 2)
    assert len(val[1]) == 2
    assert val[1][0].window_id == "val0"
    assert val[1][1].window_id == "val1"

    # w_test_overlap starts at 2.0s < w_val1.end_time_rel (3.0s), so it must be discarded
    # w_test_clean starts at 3.0s >= w_val1.end_time_rel (3.0s), so it is retained
    assert len(test[1]) == 1
    assert test[1][0].window_id == "test_clean"
    assert test[1][0].start_time_rel >= val[1][-1].end_time_rel


def test_partition_enrol_val_test_retains_different_segment():
    """Verify that test windows are NOT discarded if they belong to different continuous segments."""
    dummy_data = np.zeros((200, 3))
    dummy_mag = np.zeros(200)

    # w1 in segment 0, w2 in segment 1 (e.g. after a turn or pause)
    w1 = GaitWindow("w1", participant_id=1, session=2, label="G", segment_id=0,
                    window_index=0, start_time_rel=0.0, end_time_rel=2.0,
                    data=dummy_data, magnitude=dummy_mag)
    w2 = GaitWindow("w2", participant_id=1, session=2, label="R", segment_id=1,
                    window_index=0, start_time_rel=0.0, end_time_rel=2.0,
                    data=dummy_data, magnitude=dummy_mag)

    enrol, val, test = partition_enrol_val_test([w1, w2], val_ratio=0.50)

    # split_point = 1 -> val=[w1], test=[w2]. Because they are different segments, w2 is retained.
    assert len(val[1]) == 1
    assert val[1][0].window_id == "w1"
    assert len(test[1]) == 1
    assert test[1][0].window_id == "w2"


def test_partition_enrol_val_test_bounds_checks():
    """Verify robustness to empty inputs and participants with insufficient windows."""
    dummy_data = np.zeros((200, 3))
    dummy_mag = np.zeros(200)

    # 1. Empty input
    enrol, val, test = partition_enrol_val_test([])
    assert enrol == {} and val == {} and test == {}

    # 2. Single window in Session 2 (split_point = 0)
    w_single = GaitWindow("single", participant_id=2, session=2, label="G", segment_id=0,
                          window_index=0, start_time_rel=0.0, end_time_rel=2.0,
                          data=dummy_data, magnitude=dummy_mag)
    enrol, val, test = partition_enrol_val_test([w_single], val_ratio=0.50)
    assert val[2] == []
    assert len(test[2]) == 1

    # 3. Two overlapping windows where discarding exhausts the test set
    w1 = GaitWindow("w1", participant_id=3, session=2, label="G", segment_id=0,
                    window_index=0, start_time_rel=0.0, end_time_rel=2.0,
                    data=dummy_data, magnitude=dummy_mag)
    w2 = GaitWindow("w2", participant_id=3, session=2, label="G", segment_id=0,
                    window_index=1, start_time_rel=1.0, end_time_rel=3.0,
                    data=dummy_data, magnitude=dummy_mag)
    enrol, val, test = partition_enrol_val_test([w1, w2], val_ratio=0.50)
    assert len(val[3]) == 1
    assert test[3] == []


# ---------------------------------------------------------------------------
# DTW Tests
# ---------------------------------------------------------------------------

def test_dtw_identity():
    """DTW distance between identical signals must be exactly 0.0."""
    t = np.linspace(0, 2 * np.pi, 200)
    sig = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])

    dist = multivariate_dtw_distance(sig, sig, window_radius=30, normalize=True)
    assert dist == pytest.approx(0.0, abs=1e-7)


def test_dtw_symmetry():
    """DTW distance must be symmetric: d(A, B) == d(B, A)."""
    t = np.linspace(0, 2 * np.pi, 200)
    sig_a = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])
    sig_b = np.column_stack([np.sin(t + 0.3), np.cos(t + 0.3), np.sin(2 * t + 0.3)])

    d_ab = multivariate_dtw_distance(sig_a, sig_b, window_radius=30, normalize=True)
    d_ba = multivariate_dtw_distance(sig_b, sig_a, window_radius=30, normalize=True)
    assert d_ab == pytest.approx(d_ba, rel=1e-4)


def test_dtw_time_warp_sensitivity():
    """A time-warped version of the same gait signal should be closer than noise."""
    t = np.linspace(0, 4 * np.pi, 200)
    base = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])

    # Warped version of the base signal
    t_warped = t ** 1.05
    t_warped = t_warped * (t[-1] / t_warped[-1])
    warped = np.column_stack([np.sin(t_warped), np.cos(t_warped), np.sin(2 * t_warped)])

    # Completely different high-frequency signal
    different = np.column_stack([np.sin(10 * t), np.cos(10 * t), np.sin(20 * t)])

    d_warped = multivariate_dtw_distance(base, warped, window_radius=30, normalize=True)
    d_diff = multivariate_dtw_distance(base, different, window_radius=30, normalize=True)

    assert d_warped < d_diff, "Time-warped same signal should have smaller DTW distance than distinct signal"


# ---------------------------------------------------------------------------
# Biometric Evaluation Tests
# ---------------------------------------------------------------------------

def test_far_frr_eer_calculation():
    """Test FAR, FRR, and EER calculation with known synthetic score distributions."""
    # Synthetic scores: genuines centered at 0.2, impostors centered at 0.8
    rng = np.random.default_rng(123)
    gen_scores = rng.normal(loc=0.2, scale=0.05, size=200).tolist()
    imp_scores = rng.normal(loc=0.8, scale=0.05, size=200).tolist()

    res = compute_far_frr_curve(gen_scores, imp_scores, n_thresholds=500)

    # In perfectly separable distributions, EER should be virtually 0%
    assert res["eer"] < 0.05
    assert 0.4 < res["eer_threshold"] < 0.6
    assert res["eer_accuracy"] > 0.95

