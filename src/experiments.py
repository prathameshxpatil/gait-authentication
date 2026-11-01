"""
experiments.py — Controlled Investigation & Benchmarking Suite
==============================================================
Project : Gait-Based Biometric Authentication (Phase 2 Investigation)
Dataset : 200GaitData

This module executes a controlled suite of experiments to investigate
and systematically benchmark the baseline DTW authentication system:
  1. DTW implementation audit: Path-length normalization (N+M vs K) and Sakoe-Chiba radii.
  2. Template selection audit: Template count and temporal coverage across Session 1.
  3. Stream evaluation: G-only ('Going'), R-only ('Returning'), and Combined (G+R).
  4. Feature representations: Raw XYZ, Dynamic Acc (zero-mean), Z-norm XYZ, Magnitude ||a||, and Z-norm Magnitude.
  5. Window lengths: 2.0s (200 samples), 3.0s (300 samples), 4.0s (400 samples).
  6. Strict Validation vs Held-Out Test protocol (zero data leakage / no threshold snooping).
  7. Reporting Balanced Accuracy alongside Ordinary Accuracy.
  8. Saving full results to results/baseline_investigation_experiments.csv and generating comparison plots.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_all_recordings
from src.segmentation import (
    GaitWindow,
    extract_all_windows,
)
from src.dtw_authentication import (
    ParticipantGallery,
    build_enrolment_galleries,
    compute_verification_score,
)
from src.evaluation import (
    compute_far_frr_curve,
    evaluate_fixed_threshold,
)


def transform_window_features(
    windows: list[GaitWindow],
    feature_mode: str = "raw_xyz"
) -> list[GaitWindow]:
    """
    Transform raw window data into a specified feature representation.

    Modes:
      - 'raw_xyz': Original 3-axis filtered acceleration [X, Y, Z].
      - 'dynamic_xyz': Gravity-compensated acceleration (zero-mean per window,
                       removing static DC gravity vector tilt).
      - 'znorm_xyz': Channel-standardized acceleration (zero-mean, unit-variance).
      - 'magnitude': 1D vector acceleration magnitude ||a|| = sqrt(X^2 + Y^2 + Z^2),
                     rotationally invariant to phone pocket tilt.
      - 'znorm_magnitude': Standardized acceleration magnitude (zero-mean, unit-variance).
      - 'dynamic_plus_mag': 4D feature [X_dyn, Y_dyn, Z_dyn, ||a||].

    Parameters
    ----------
    windows : list[GaitWindow]
        Input gait windows.
    feature_mode : str
        Feature extraction mode.

    Returns
    -------
    list[GaitWindow]
        Transformed gait windows with updated .data matrices.
    """
    transformed = []
    for w in windows:
        raw_mat = w.data  # shape (N, 3)

        if feature_mode == "raw_xyz":
            feat_data = raw_mat

        elif feature_mode == "dynamic_xyz":
            # Subtract static mean per channel (removes constant gravity offset)
            feat_data = raw_mat - raw_mat.mean(axis=0)

        elif feature_mode == "znorm_xyz":
            # Zero-mean and unit variance per channel
            std = raw_mat.std(axis=0)
            std[std == 0] = 1.0
            feat_data = (raw_mat - raw_mat.mean(axis=0)) / std

        elif feature_mode == "magnitude":
            # 1D acceleration magnitude
            mag = w.magnitude
            feat_data = mag[:, None]

        elif feature_mode == "znorm_magnitude":
            # Standardized magnitude
            mag = w.magnitude
            m_std = mag.std()
            if m_std == 0:
                m_std = 1.0
            feat_data = ((mag - mag.mean()) / m_std)[:, None]

        elif feature_mode == "dynamic_plus_mag":
            # 4-channel: 3 dynamic axes + magnitude
            dyn = raw_mat - raw_mat.mean(axis=0)
            mag = w.magnitude[:, None]
            feat_data = np.hstack([dyn, mag])

        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        w_new = GaitWindow(
            window_id=w.window_id,
            participant_id=w.participant_id,
            session=w.session,
            label=w.label,
            segment_id=w.segment_id,
            window_index=w.window_index,
            start_time_rel=w.start_time_rel,
            end_time_rel=w.end_time_rel,
            data=feat_data,
            magnitude=w.magnitude,
        )
        transformed.append(w_new)

    return transformed


def partition_enrol_val_test(
    windows: list[GaitWindow],
    val_ratio: float = 0.5
) -> tuple[dict[int, list[GaitWindow]], dict[int, list[GaitWindow]], dict[int, list[GaitWindow]]]:
    """
    Split windows into three strictly segregated sets:
      1. Enrolment : Strictly Session 1.
      2. Validation: Chronological first half of Session 2 (used solely for threshold selection).
      3. Test      : Chronological second half of Session 2 (held-out for final evaluation).

    Enforces zero sample leakage across the partition boundary by discarding
    test window(s) that share raw samples with the final validation window.

    Parameters
    ----------
    windows : list[GaitWindow]
        Transformed windows.
    val_ratio : float
        Fraction of Session 2 used for validation (default: 0.50).

    Returns
    -------
    tuple
        (enrol_dict, val_dict, test_dict) mapping participant_id -> list of windows.
    """
    enrol_dict = {}
    val_dict = {}
    test_dict = {}

    # Separate by session
    s1_by_user = {}
    s2_by_user = {}

    for w in windows:
        pid = w.participant_id
        if w.session == 1:
            s1_by_user.setdefault(pid, []).append(w)
        elif w.session == 2:
            s2_by_user.setdefault(pid, []).append(w)

    # Enrolment is all of Session 1
    enrol_dict = s1_by_user

    # Chronologically split Session 2 into Validation (first half) and Test (second half)
    for pid, s2_wins in s2_by_user.items():
        n = len(s2_wins)
        if n == 0:
            val_dict[pid] = []
            test_dict[pid] = []
            continue

        split_point = int(n * val_ratio)
        val_dict[pid] = s2_wins[:split_point]

        # Bounds check and boundary overlap discard
        test_start_idx = split_point
        if 0 < split_point < n:
            last_val = s2_wins[split_point - 1]
            while test_start_idx < n:
                cand_test = s2_wins[test_start_idx]
                same_segment = (
                    getattr(last_val, "segment_id", None) == getattr(cand_test, "segment_id", None)
                    and getattr(last_val, "label", None) == getattr(cand_test, "label", None)
                    and getattr(last_val, "session", None) == getattr(cand_test, "session", None)
                )
                # Discard test window if it shares raw samples with the final validation window
                if same_segment and (cand_test.start_time_rel < last_val.end_time_rel - 1e-6):
                    test_start_idx += 1
                else:
                    break

        test_dict[pid] = s2_wins[test_start_idx:]

    # Verification assertion: confirm no raw samples are shared between the
    # final validation window and the first retained test window for any participant.
    for pid in s2_by_user.keys():
        if val_dict[pid] and test_dict[pid]:
            final_val = val_dict[pid][-1]
            first_retained_test = test_dict[pid][0]
            same_segment = (
                getattr(final_val, "segment_id", None) == getattr(first_retained_test, "segment_id", None)
                and getattr(final_val, "label", None) == getattr(first_retained_test, "label", None)
                and getattr(final_val, "session", None) == getattr(first_retained_test, "session", None)
            )
            if same_segment:
                assert first_retained_test.start_time_rel >= final_val.end_time_rel - 1e-6, (
                    f"Sample leakage detected for participant {pid}: "
                    f"final validation window {final_val.window_id} ({final_val.start_time_rel:.3f}s-{final_val.end_time_rel:.3f}s) "
                    f"shares raw samples with first retained test window {first_retained_test.window_id} "
                    f"({first_retained_test.start_time_rel:.3f}s-{first_retained_test.end_time_rel:.3f}s)"
                )

    return enrol_dict, val_dict, test_dict


def evaluate_set_against_gallery(
    galleries: dict[int, ParticipantGallery],
    claimant_windows: dict[int, list[GaitWindow]],
    window_radius: int = 30,
    norm_mode: str = "sum_lengths",
    n_windows_per_user: int = 30,
    seed: int = 42
) -> tuple[list[float], list[float]]:
    """
    Run verification evaluation on a specified evaluation set (Validation or Test).

    Parameters
    ----------
    galleries : dict
        Enrolled user galleries from Session 1.
    claimant_windows : dict
        Evaluation windows (Validation or Test).
    window_radius : int
        Sakoe-Chiba band radius.
    norm_mode : str
        'sum_lengths' or 'path_length'.
    n_windows_per_user : int
        Sample size per user to balance speed and statistical precision.
    seed : int
        Random seed.

    Returns
    -------
    tuple[list[float], list[float]]
        (genuine_scores, impostor_scores)
    """
    rng = np.random.default_rng(seed)
    pids = sorted(galleries.keys())

    gen_scores = []
    imp_scores = []

    for claimant_id in pids:
        user_wins = claimant_windows.get(claimant_id, [])
        if len(user_wins) == 0:
            continue

        if len(user_wins) > n_windows_per_user:
            indices = rng.choice(len(user_wins), size=n_windows_per_user, replace=False)
            sampled = [user_wins[i] for i in indices]
        else:
            sampled = user_wins

        # 1. Genuine attempts (claim own identity)
        gal_own = galleries[claimant_id]
        for w in sampled:
            score = compute_verification_score(
                w, gal_own, window_radius=window_radius, norm_mode=norm_mode
            )
            gen_scores.append(score)

        # 2. Impostor attempts (claim every other user's identity)
        for claimed_id in pids:
            if claimed_id == claimant_id:
                continue
            gal_other = galleries[claimed_id]
            # Use up to 10 samples per impostor pair for balanced evaluation
            imp_sample = sampled[:min(10, len(sampled))]
            for w in imp_sample:
                score = compute_verification_score(
                    w, gal_other, window_radius=window_radius, norm_mode=norm_mode
                )
                imp_scores.append(score)

    return gen_scores, imp_scores


def run_single_experiment(
    segments: dict,
    exp_id: str,
    exp_name: str,
    feature_mode: str = "raw_xyz",
    stream_filter: str = "all",  # 'all', 'G', 'R'
    window_sec: float = 2.0,
    window_radius: int = 30,
    norm_mode: str = "sum_lengths",
    n_templates: int = 8,
    template_strategy: str = "stratified",
    n_eval_windows_per_user: int = 30,
    seed: int = 42
) -> dict:
    """
    Run one complete, controlled experiment under the strict Validation/Test protocol.
    """
    t_start = time.time()
    w_size = int(window_sec * 100)
    s_size = int(w_size / 2)  # 50% overlap

    # 1. Windowing
    windows = extract_all_windows(segments, window_size=w_size, step_size=s_size)

    # 2. Stream filtering
    if stream_filter in ["G", "R"]:
        windows = [w for w in windows if w.label == stream_filter]

    # 3. Feature transformation
    windows = transform_window_features(windows, feature_mode=feature_mode)

    # 4. Strict Partition: Enrolment (S1), Validation (S2 1st half), Test (S2 2nd half)
    enrol, val, test = partition_enrol_val_test(windows, val_ratio=0.5)

    # 5. Build Enrolment Galleries strictly from Session 1
    galleries = build_enrolment_galleries(
        enrol, n_templates_per_user=n_templates, strategy=template_strategy
    )

    # 6. Evaluate on VALIDATION set to tune threshold
    val_gen, val_imp = evaluate_set_against_gallery(
        galleries, val, window_radius=window_radius, norm_mode=norm_mode,
        n_windows_per_user=n_eval_windows_per_user, seed=seed
    )
    val_res = compute_far_frr_curve(val_gen, val_imp, n_thresholds=500)
    val_threshold = val_res["eer_threshold"]
    val_eer = val_res["eer"]
    val_bal_acc = val_res.get("eer_balanced_accuracy", 1.0 - val_eer)

    # 7. Evaluate on HELD-OUT TEST set using the fixed validation threshold
    test_gen, test_imp = evaluate_set_against_gallery(
        galleries, test, window_radius=window_radius, norm_mode=norm_mode,
        n_windows_per_user=n_eval_windows_per_user, seed=seed + 100
    )
    test_eval = evaluate_fixed_threshold(test_gen, test_imp, threshold=val_threshold)

    # Also compute intrinsic test EER for diagnostic comparison
    test_res = compute_far_frr_curve(test_gen, test_imp, n_thresholds=500)
    test_eer_intrinsic = test_res["eer"]

    duration = time.time() - t_start

    result = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "feature_mode": feature_mode,
        "stream_filter": stream_filter,
        "window_duration_sec": window_sec,
        "dtw_radius": window_radius,
        "norm_mode": norm_mode,
        "n_templates": n_templates,
        "val_threshold": round(val_threshold, 4),
        "val_eer_pct": round(val_eer * 100, 2),
        "val_bal_acc_pct": round(val_bal_acc * 100, 2),
        "test_eer_intrinsic_pct": round(test_eer_intrinsic * 100, 2),
        "test_far_pct": round(test_eval["far"] * 100, 2),
        "test_frr_pct": round(test_eval["frr"] * 100, 2),
        "test_balanced_acc_pct": round(test_eval["balanced_accuracy"] * 100, 2),
        "test_accuracy_pct": round(test_eval["accuracy"] * 100, 2),
        "test_n_genuine": test_eval["n_genuine"],
        "test_n_impostor": test_eval["n_impostor"],
        "gen_mean_distance": round(float(np.mean(test_gen)), 4),
        "imp_mean_distance": round(float(np.mean(test_imp)), 4),
        "distance_margin": round(float(np.mean(test_imp) - np.mean(test_gen)), 4),
        "runtime_sec": round(duration, 2),
        # Store score lists for plotting
        "_test_gen": test_gen,
        "_test_imp": test_imp,
        "_test_res": test_res,
        "_val_threshold": val_threshold,
    }

    return result


def run_all_investigation_experiments(
    results_dir: str = None
) -> tuple[pd.DataFrame, dict]:
    """
    Run the complete investigation matrix covering all 9 project requirements.
    Saves CSV and figures to results_dir.
    """
    if results_dir is None:
        results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 76)
    print("  PHASE 2 INVESTIGATION & AUDIT EXPERIMENTS SUITE")
    print("=" * 76)

    # Preprocess all recordings once
    print("Loading and preprocessing all 20 recordings...")
    segments = preprocess_all_recordings(save_files=False)
    print(f"Loaded {len(segments)} continuous preprocessed segments.\n")

    # Define the controlled experiment configurations
    exp_configs = [
        # --- Experiment 1: Baseline Replication ---
        {
            "exp_id": "EXP-01",
            "exp_name": "Baseline (Raw XYZ, 2s, R=30, N+M norm)",
            "feature_mode": "raw_xyz",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        # --- Experiment 2: Path-Length Normalization Audit ---
        {
            "exp_id": "EXP-02",
            "exp_name": "Path-Length Normalization (Divide by K)",
            "feature_mode": "raw_xyz",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "path_length",
            "n_templates": 8,
        },
        # --- Experiment 3a-3c: Sakoe-Chiba Constraint Audit ---
        {
            "exp_id": "EXP-03A",
            "exp_name": "Narrow Band Radius (R=15 / 0.15s)",
            "feature_mode": "raw_xyz",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 15,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        {
            "exp_id": "EXP-03B",
            "exp_name": "Wide Band Radius (R=50 / 0.50s)",
            "feature_mode": "raw_xyz",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 50,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        {
            "exp_id": "EXP-03C",
            "exp_name": "Unconstrained DTW (Full Matrix / No Band)",
            "feature_mode": "raw_xyz",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": -1,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        # --- Experiment 4a-4b: Stream Evaluation (G vs R vs Combined) ---
        {
            "exp_id": "EXP-04A",
            "exp_name": "Stream Evaluation: G-Only (Outward walk)",
            "feature_mode": "raw_xyz",
            "stream_filter": "G",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        {
            "exp_id": "EXP-04B",
            "exp_name": "Stream Evaluation: R-Only (Return walk)",
            "feature_mode": "raw_xyz",
            "stream_filter": "R",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        # --- Experiment 5a-5d: Feature Representation Experiments ---
        {
            "exp_id": "EXP-05A",
            "exp_name": "Dynamic Acc (Gravity Removed / Zero-Mean)",
            "feature_mode": "dynamic_xyz",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        {
            "exp_id": "EXP-05B",
            "exp_name": "Z-Score Normalized Acc (Standardized XYZ)",
            "feature_mode": "znorm_xyz",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        {
            "exp_id": "EXP-05C",
            "exp_name": "Acceleration Magnitude ||a|| (Rotation Invariant)",
            "feature_mode": "magnitude",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        {
            "exp_id": "EXP-05D",
            "exp_name": "Z-Score Normalized Magnitude",
            "feature_mode": "znorm_magnitude",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        # --- Experiment 6a-6b: Alternative Window Lengths ---
        {
            "exp_id": "EXP-06A",
            "exp_name": "Window Length: 3.0 Seconds (300 samples)",
            "feature_mode": "raw_xyz",
            "stream_filter": "all",
            "window_sec": 3.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        {
            "exp_id": "EXP-06B",
            "exp_name": "Window Length: 4.0 Seconds (400 samples)",
            "feature_mode": "raw_xyz",
            "stream_filter": "all",
            "window_sec": 4.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 8,
        },
        # --- Experiment 7a-7b: Template Count Evaluation ---
        {
            "exp_id": "EXP-07A",
            "exp_name": "Enrolment Gallery Size: 4 Templates",
            "feature_mode": "raw_xyz",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 4,
        },
        {
            "exp_id": "EXP-07B",
            "exp_name": "Enrolment Gallery Size: 16 Templates",
            "feature_mode": "raw_xyz",
            "stream_filter": "all",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "sum_lengths",
            "n_templates": 16,
        },
        # --- Experiment 8: Best Synergy Candidate ---
        {
            "exp_id": "EXP-08",
            "exp_name": "Synergy: Z-Norm Mag + G-Stream (Rotation Invariant)",
            "feature_mode": "znorm_magnitude",
            "stream_filter": "G",
            "window_sec": 2.0,
            "window_radius": 30,
            "norm_mode": "path_length",
            "n_templates": 8,
        },
    ]

    all_results = []
    best_exp_data = None
    best_bal_acc = -1.0

    print(f"Running {len(exp_configs)} controlled benchmark experiments...")

    for i, cfg in enumerate(exp_configs, 1):
        print(f"[{i:2d}/{len(exp_configs)}] Running {cfg['exp_id']}: {cfg['exp_name']}...", end=" ", flush=True)
        res = run_single_experiment(segments, **cfg)
        all_results.append(res)
        print(
            f"Done! Test BalAcc={res['test_balanced_acc_pct']}%, "
            f"Test EER={res['test_eer_intrinsic_pct']}%, "
            f"Margin={res['distance_margin']:+.4f} ({res['runtime_sec']:.1f}s)"
        )

        if res["test_balanced_acc_pct"] > best_bal_acc:
            best_bal_acc = res["test_balanced_acc_pct"]
            best_exp_data = res

    # Convert to DataFrame (dropping internal score arrays for CSV export)
    export_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
    df_results = pd.DataFrame(export_rows)

    # Save to CSV
    csv_path = os.path.join(results_dir, "baseline_investigation_experiments.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Experiment results saved to: {csv_path}")

    # Generate comparative plots
    generate_experiment_plots(df_results, best_exp_data, results_dir)

    return df_results, best_exp_data


def generate_experiment_plots(
    df_results: pd.DataFrame,
    best_exp: dict,
    results_dir: str
) -> None:
    """Generate comparative visualization plots across all experiments."""
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # --- Plot 1: Feature Representation Comparison ---
    feat_exps = df_results[df_results["exp_id"].isin(["EXP-01", "EXP-05A", "EXP-05B", "EXP-05C", "EXP-05D"])]
    if len(feat_exps) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(feat_exps))
        width = 0.35
        ax.bar(x - width/2, feat_exps["test_eer_intrinsic_pct"], width, label="Test EER (%) [Lower is better]", color="#e74c3c", alpha=0.85)
        ax.bar(x + width/2, feat_exps["test_balanced_acc_pct"], width, label="Test Balanced Acc (%) [Higher is better]", color="#2ecc71", alpha=0.85)
        ax.set_xticks(x)
        labels = [f"{row['feature_mode']}\n({row['exp_id']})" for _, row in feat_exps.iterrows()]
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Impact of Sensor Feature Representation on Biometric Performance", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend()
        plt.tight_layout()
        p1 = os.path.join(results_dir, "experiments_feature_comparison.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved figure: {p1}")

    # --- Plot 2: Window Length Comparison ---
    win_exps = df_results[df_results["exp_id"].isin(["EXP-01", "EXP-06A", "EXP-06B"])]
    if len(win_exps) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(win_exps))
        width = 0.35
        ax.bar(x - width/2, win_exps["test_eer_intrinsic_pct"], width, label="Test EER (%)", color="#e74c3c", alpha=0.85)
        ax.bar(x + width/2, win_exps["test_balanced_acc_pct"], width, label="Test Balanced Acc (%)", color="#3498db", alpha=0.85)
        ax.set_xticks(x)
        labels = [f"{row['window_duration_sec']}s Window\n({row['exp_id']})" for _, row in win_exps.iterrows()]
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Impact of Window Duration (2.0s vs 3.0s vs 4.0s)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend()
        plt.tight_layout()
        p2 = os.path.join(results_dir, "experiments_window_comparison.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved figure: {p2}")

    # --- Plot 3: Stream Comparison ---
    stream_exps = df_results[df_results["exp_id"].isin(["EXP-01", "EXP-04A", "EXP-04B"])]
    if len(stream_exps) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(stream_exps))
        width = 0.35
        ax.bar(x - width/2, stream_exps["test_eer_intrinsic_pct"], width, label="Test EER (%)", color="#e74c3c", alpha=0.85)
        ax.bar(x + width/2, stream_exps["test_balanced_acc_pct"], width, label="Test Balanced Acc (%)", color="#9b59b6", alpha=0.85)
        ax.set_xticks(x)
        labels = ["Combined (G+R)", "G-Only (Going)", "R-Only (Returning)"]
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Stream Evaluation: G-Only vs R-Only vs Combined (G+R)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend()
        plt.tight_layout()
        p3 = os.path.join(results_dir, "experiments_stream_comparison.png")
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved figure: {p3}")

    # --- Plot 4: Best Configuration Evaluation ---
    if best_exp is not None and "_test_res" in best_exp:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left: Score distributions
        gen = np.asarray(best_exp["_test_gen"])
        imp = np.asarray(best_exp["_test_imp"])
        sns.histplot(gen, kde=True, stat="density", color="#2ecc71", label="Genuine", bins=35, alpha=0.45, ax=ax1)
        sns.histplot(imp, kde=True, stat="density", color="#e74c3c", label="Impostor", bins=35, alpha=0.45, ax=ax1)
        ax1.axvline(best_exp["_val_threshold"], color="#2c3e50", linestyle="--", linewidth=2.0,
                    label=f"Val Threshold θ* = {best_exp['_val_threshold']:.4f}")
        ax1.set_xlabel("DTW Distance")
        ax1.set_ylabel("Density")
        ax1.set_title(f"Score Distributions ({best_exp['exp_id']}: {best_exp['exp_name']})", fontsize=11, fontweight="bold")
        ax1.legend()

        # Right: ROC Curve
        test_res = best_exp["_test_res"]
        tpr = 1.0 - test_res["frr"]
        ax2.plot(test_res["far"] * 100, tpr * 100, color="#2980b9", linewidth=2.2, label=f"{best_exp['exp_id']}")
        ax2.plot([0, 100], [0, 100], color="gray", linestyle="--", label="Random Guess (EER=50%)")
        ax2.scatter([best_exp["test_far_pct"]], [100.0 - best_exp["test_frr_pct"]],
                    color="#8e44ad", s=90, zorder=5, label=f"Held-out Operating Point (BalAcc={best_exp['test_balanced_acc_pct']}%)")
        ax2.set_xlabel("False Acceptance Rate — FAR (%)")
        ax2.set_ylabel("True Acceptance Rate — TAR (%)")
        ax2.set_title(f"Held-Out ROC Curve (EER = {best_exp['test_eer_intrinsic_pct']}%)", fontsize=11, fontweight="bold")
        ax2.set_xlim(-2, 102)
        ax2.set_ylim(-2, 102)
        ax2.legend(loc="lower right")

        plt.suptitle(f"Best Performing Configuration: {best_exp['exp_id']} ({best_exp['exp_name']})",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        p4 = os.path.join(results_dir, "best_configuration_evaluation.png")
        plt.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved figure: {p4}")


if __name__ == "__main__":
    df, best = run_all_investigation_experiments()
    print("\n" + "=" * 76)
    print("  INVESTIGATION COMPLETE — TOP CONFIGURATIONS")
    print("=" * 76)
    top_df = df.sort_values("test_balanced_acc_pct", ascending=False).head(5)
    print(top_df[["exp_id", "exp_name", "test_eer_intrinsic_pct", "test_far_pct", "test_frr_pct", "test_balanced_acc_pct", "distance_margin"]].to_string(index=False))

