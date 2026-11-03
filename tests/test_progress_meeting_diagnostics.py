"""
tests/test_progress_meeting_diagnostics.py — Unit Tests for Adaptation Diagnostics & Participant Analysis.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.segmentation import GaitWindow
from src.adaptation_diagnostics import _worker_diagnostics_user_stream
from src.adaptive_stream_validation import build_operational_user_stream
from src.participant_analysis import run_participant_analysis


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


def test_diagnostics_worker_stage_accounting():
    """Verify that stage counters sum up correctly and capture genuine progression."""
    pid = 1
    pids = [1, 2, 3]
    val_dict = {
        p: [_make_dummy_window(p, i, t_start=i * 2.0) for i in range(10)]
        for p in pids
    }
    perm_tmpls = [val_dict[1][0].data, val_dict[1][1].data]
    stream = build_operational_user_stream(uid=1, all_pids=pids, val_windows_dict=val_dict, seed=42)

    args = (
        pid,
        "Dual-Gallery Conservative",
        perm_tmpls,
        stream,
        0.1336,
        {
            "w_perm": 0.80,
            "max_adaptive_capacity": 4,
            "candidate_target_size": 3,
            "alpha_high_confidence": 0.65,
            "gamma_cluster_consistency": 0.80,
            "beta_anchor_guard": 0.85,
        },
    )

    ret_pid, counts = _worker_diagnostics_user_stream(args)
    assert ret_pid == 1
    # Check total attempts match stream
    assert counts["gen_attempts"] == 10
    assert counts["imp_attempts"] == 10

    # Total genuine attempts must equal rejected_auth + accepted_low_conf + passed_high_conf
    sum_gen_auth = (
        counts["gen_rejected_auth"]
        + counts["gen_accepted_low_conf"]
        + counts["gen_passed_high_conf"]
    )
    assert sum_gen_auth == counts["gen_attempts"]

    # Passed high conf must equal entered_buffer + rejected_overlap
    assert counts["gen_entered_buffer"] + counts["gen_rejected_overlap"] == counts["gen_passed_high_conf"]

    # Impostor promotions must be 0 in synthetic baseline
    assert counts["impostor_promotions"] == 0


def test_participant_analysis_aggregation(tmp_path):
    """Verify participant analysis outputs correct delta and summary metrics."""
    # Create synthetic per_participant dataframe
    rows = []
    for pid in [1, 2]:
        for strat, far, frr, bal in [
            ("Fixed Baseline", 20.0, 30.0, 75.0),
            ("Dual-Gallery Conservative", 18.0, 28.0, 77.0),
            ("Dual-Gallery Standard", 15.0, 25.0, 80.0),
        ]:
            for seed in [42, 100]:
                rows.append({
                    "strategy": strat,
                    "seed": seed,
                    "participant_id": pid,
                    "stream_genuine": 100,
                    "stream_impostor": 100,
                    "stream_far_percent": far,
                    "stream_frr_percent": frr,
                    "stream_balanced_accuracy": bal,
                    "total_updates": 1 if "Dual" in strat else 0,
                    "genuine_updates": 1 if "Dual" in strat else 0,
                    "poisoned_updates": 0,
                    "final_drift": 0.05 if "Dual" in strat else 0.0,
                    "exh_genuine": 100,
                    "exh_impostor": 5400,
                    "exh_far_percent": far,
                    "exh_frr_percent": frr,
                    "exh_balanced_accuracy": bal,
                })

    df = pd.DataFrame(rows)
    input_csv = tmp_path / "per_participant.csv"
    df.to_csv(input_csv, index=False)

    df_pivot = run_participant_analysis(input_csv=str(input_csv), output_dir=str(tmp_path))
    assert len(df_pivot) == 2
    # Verify delta calculation
    assert "cons_delta_bal_acc" in df_pivot.columns
    # 77.0 - 75.0 = +2.0
    assert np.isclose(df_pivot.loc[df_pivot["participant_id"] == 1, "cons_delta_bal_acc"].values[0], 2.0)

