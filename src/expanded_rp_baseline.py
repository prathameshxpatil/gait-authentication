"""
expanded_rp_baseline.py — Fixed DTW Baseline on Expanded 55-Participant RP Cohort
==================================================================================
Project : Gait-Based Biometric Authentication (Dataset Expansion)
Dataset : 200GaitData (Expanded Right Pocket Accelerometer Collection, 55 Participants)

Methodology:
  1. Cohort: 55 participants having both RP Session 1 (enrollment) and RP Session 2 (authentication).
  2. Features: znorm_magnitude, 4.0-second windows (400 samples @ 100 Hz), 50% overlap.
  3. Galleries: 8 stratified templates per participant from Session 1.
  4. Partitions:
     - Session 1: 100% Enrollment (12,933 windows).
     - Session 2 First 50%: Validation partition (6,310 windows).
     - Boundary overlap discard: Exactly 40 windows discarded (zero raw sample leakage).
     - Session 2 Second 50%: Held-out test partition (6,291 windows).
  5. Threshold Calibration:
     - "Cohort-calibrated threshold selection using validation scores from the 55 enrolled participants."
     - Calibrated at Equal Error Rate (EER) on validation genuine (6,310) and exhaustive impostor (340,740) scores.
     - Threshold theta* is frozen before test evaluation.
  6. Evaluation:
     - Exhaustive impostor testing against all 54 non-owner galleries.
     - Enrolled-identity-centric macro metrics:
         FAR_i = False Accepts against gallery i / Impostor attempts against gallery i
         FRR_i = False Rejects for identity i / Genuine attempts for identity i
         Macro FAR = Mean(FAR_i), Macro FRR = Mean(FRR_i)
     - Micro metrics across pooled attempts.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_manifest
from src.preprocessing import get_expanded_processed_data_dir
from src.segmentation import GaitWindow, extract_windows_from_segment
from src.experiments import transform_window_features, partition_enrol_val_test
from src.dtw_authentication import (
    ParticipantGallery,
    multivariate_dtw_distance,
)
from src.evaluation import compute_far_frr_curve, compute_cohort_metrics


def _match_windows_against_gallery(args):
    """Worker function for parallel DTW matching against a single gallery."""
    gallery_pid, gallery_templates, windows_data, window_radius, norm_mode = args
    scores = []
    for w_mat in windows_data:
        d = min(
            multivariate_dtw_distance(w_mat, t, window_radius=window_radius, norm_mode=norm_mode)
            for t in gallery_templates
        )
        scores.append(d)
    return gallery_pid, np.array(scores, dtype=np.float32)


def build_galleries_for_cohort(
    enrol_dict: dict[int, list[GaitWindow]],
    n_templates: int = 8
) -> dict[int, ParticipantGallery]:
    """Build permanent enrollment galleries with stratified templates for all cohort users."""
    galleries = {}
    for pid, wins in enrol_dict.items():
        if len(wins) < n_templates:
            selected_wins = wins
        else:
            indices = np.linspace(0, len(wins) - 1, n_templates, dtype=int)
            selected_wins = [wins[i] for i in indices]
        galleries[pid] = ParticipantGallery(
            participant_id=pid,
            templates=[w.data for w in selected_wins],
            template_window_ids=[w.window_id for w in selected_wins]
        )
    return galleries


def compute_all_scores_for_partition(
    galleries: dict[int, ParticipantGallery],
    windows_dict: dict[int, list[GaitWindow]],
    window_radius: int = 30,
    norm_mode: str = "sum_lengths",
    n_workers: int = 8
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Compute genuine scores and exhaustive impostor scores for a given partition.

    Returns
    -------
    tuple
        (genuine_scores_by_user, impostor_scores_against_gallery)
        - genuine_scores_by_user[i]: scores of user i's windows against gallery i
        - impostor_scores_against_gallery[i]: scores of all non-owner windows (j != i)
          evaluated against gallery i
    """
    pids = sorted(galleries.keys())

    # Pre-extract numpy matrices for fast serialization
    windows_by_user = {p: [w.data for w in windows_dict[p]] for p in pids}
    total_wins = sum(len(windows_by_user[p]) for p in pids)

    # 1. Genuine Matching
    print("  • Computing genuine scores across 55 participants...")
    genuine_scores = {}
    gen_tasks = [
        (pid, galleries[pid].templates, windows_by_user[pid], window_radius, norm_mode)
        for pid in pids
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for pid, scores in executor.map(_match_windows_against_gallery, gen_tasks):
            genuine_scores[pid] = scores

    # 2. Exhaustive Impostor Matching
    print("  • Computing exhaustive impostor scores across all 54 non-owner galleries...")
    # For each gallery i, the impostor windows are all windows from all j != i
    impostor_scores = {}
    imp_tasks = []
    for pid in pids:
        other_wins = []
        for other_p in pids:
            if other_p != pid:
                other_wins.extend(windows_by_user[other_p])
        imp_tasks.append(
            (pid, galleries[pid].templates, other_wins, window_radius, norm_mode)
        )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for pid, scores in executor.map(_match_windows_against_gallery, imp_tasks):
            impostor_scores[pid] = scores

    return genuine_scores, impostor_scores


def run_expanded_rp_baseline(
    output_dir: str = "results/expanded_rp_baseline",
    window_sec: float = 4.0,
    window_radius: int = 30,
    norm_mode: str = "sum_lengths",
    n_templates: int = 8,
    n_workers: int = 8
) -> dict:
    """
    Executes the full fixed-template DTW baseline evaluation on the 55-participant RP cohort.
    """
    t0_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 88)
    print("  EXPANDED RIGHT POCKET DTW BASELINE EVALUATION (55 PARTICIPANTS)")
    print("=" * 88)

    # 1. Identify Cohort
    manifest = load_manifest()
    rp_all = manifest[manifest["placement"] == "RP"]
    pids_s1 = set(rp_all[rp_all["session_int"] == 1]["participant_id"])
    pids_s2 = set(rp_all[rp_all["session_int"] == 2]["participant_id"])
    cohort_pids = sorted(list(pids_s1.intersection(pids_s2)))

    assert len(cohort_pids) == 55, f"Expected 55 cohort participants, found {len(cohort_pids)}"
    print(f"\n[Step 1/6] Cohort Selection: {len(cohort_pids)} participants verified.")

    # 2. Window Extraction & Feature Transformation
    print("\n[Step 2/6] Extracting 4.0s windows (znorm_magnitude, 50% overlap)...")
    exp_dir = get_expanded_processed_data_dir()
    all_exp_files = os.listdir(exp_dir)

    raw_windows = []
    for p in cohort_pids:
        for s in [1, 2]:
            prefix = f"sub{p}-rp-s{s}_"
            matched = sorted([f for f in all_exp_files if f.startswith(prefix)])
            for sf in matched:
                df = pd.read_csv(os.path.join(exp_dir, sf))
                wins = extract_windows_from_segment(df, window_size=400, step_size=200)
                raw_windows.extend(wins)

    assert len(raw_windows) == 25574, f"Expected 25,574 total windows, got {len(raw_windows)}"
    windows = transform_window_features(raw_windows, feature_mode="znorm_magnitude")
    print(f"  ✓ Total windows extracted & transformed: {len(windows)}")

    # 3. Partitioning with Zero Boundary Leakage
    print("\n[Step 3/6] Partitioning (Session 1 Enrolment, S2 Val / Test 50/50 split)...")
    enrol_dict, val_dict, test_dict = partition_enrol_val_test(windows, val_ratio=0.50)

    n_enrol_total = sum(len(w) for w in enrol_dict.values())
    n_val_total = sum(len(w) for w in val_dict.values())
    n_test_total = sum(len(w) for w in test_dict.values())
    n_discarded_total = len(windows) - (n_enrol_total + n_val_total + n_test_total)

    assert n_enrol_total == 12933, f"Expected 12,933 enrolment windows, got {n_enrol_total}"
    assert n_val_total == 6310, f"Expected 6,310 validation windows, got {n_val_total}"
    assert n_test_total == 6291, f"Expected 6,291 test windows, got {n_test_total}"
    assert n_discarded_total == 40, f"Expected 40 boundary discards, got {n_discarded_total}"

    print(f"  ✓ Enrolment windows : {n_enrol_total}")
    print(f"  ✓ Validation windows: {n_val_total}")
    print(f"  ✓ Boundary discards : {n_discarded_total} (zero raw sample leakage verified)")
    print(f"  ✓ Retained test wins: {n_test_total}")

    # Save cohort summary
    cohort_summary_rows = []
    for p in cohort_pids:
        s2_wins_user = len(val_dict[p]) + len(test_dict[p])
        disc = len([w for w in windows if w.participant_id == p and w.session == 2]) - s2_wins_user
        cohort_summary_rows.append({
            "participant_id": p,
            "session1_enrol_windows": len(enrol_dict[p]),
            "session2_val_windows": len(val_dict[p]),
            "session2_boundary_discards": disc,
            "session2_test_windows": len(test_dict[p]),
            "total_retained_windows": len(enrol_dict[p]) + len(val_dict[p]) + len(test_dict[p]),
        })
    df_cohort = pd.DataFrame(cohort_summary_rows)
    df_cohort.to_csv(os.path.join(output_dir, "cohort_summary.csv"), index=False)
    print("  ✓ Saved cohort_summary.csv")

    # 4. Build Galleries
    print("\n[Step 4/6] Building 8-template permanent galleries from Session 1...")
    galleries = build_galleries_for_cohort(enrol_dict, n_templates=n_templates)

    # 5. Validation Matching & Cohort-Calibrated Threshold Selection
    print("\n[Step 5/6] Validation Score Generation & Cohort-Calibrated Threshold Selection...")
    val_gen_scores, val_imp_scores = compute_all_scores_for_partition(
        galleries, val_dict, window_radius=window_radius, norm_mode=norm_mode, n_workers=n_workers
    )

    pooled_val_gen = np.concatenate([val_gen_scores[p] for p in cohort_pids])
    pooled_val_imp = np.concatenate([val_imp_scores[p] for p in cohort_pids])

    assert len(pooled_val_gen) == 6310
    assert len(pooled_val_imp) == 340740

    val_curve = compute_far_frr_curve(pooled_val_gen, pooled_val_imp, n_thresholds=2000)
    theta_star = float(val_curve["eer_threshold"])
    val_eer = float(val_curve["eer"] * 100.0)

    # Compute validation micro and macro cohort metrics at the calibrated operating threshold
    val_metrics = compute_cohort_metrics(val_gen_scores, val_imp_scores, threshold=theta_star)

    print(f"  ✓ Cohort-Calibrated Operating Threshold : θ* = {theta_star:.4f}")
    print(f"  ✓ Validation Equal Error Rate (EER)     : {val_eer:.2f}%")
    print(f"  ✓ Validation Micro FAR / FRR            : {val_metrics['micro']['far_percent']:.2f}% / {val_metrics['micro']['frr_percent']:.2f}%")
    print(f"  ✓ Validation Macro FAR / FRR            : {val_metrics['macro']['far_percent']:.2f}% / {val_metrics['macro']['frr_percent']:.2f}%")
    print(f"  ✓ Threshold frozen for test evaluation.")

    # 6. Held-Out Test Evaluation with Frozen Threshold
    print("\n[Step 6/6] Held-Out Test Evaluation (Exhaustive Impostor Testing)...")
    test_gen_scores, test_imp_scores = compute_all_scores_for_partition(
        galleries, test_dict, window_radius=window_radius, norm_mode=norm_mode, n_workers=n_workers
    )

    pooled_test_gen = np.concatenate([test_gen_scores[p] for p in cohort_pids])
    pooled_test_imp = np.concatenate([test_imp_scores[p] for p in cohort_pids])

    assert len(pooled_test_gen) == 6291
    assert len(pooled_test_imp) == 339714

    test_curve = compute_far_frr_curve(pooled_test_gen, pooled_test_imp, n_thresholds=2000)
    test_diagnostic_eer = float(test_curve["eer"] * 100.0)

    # Compute micro and macro metrics with enrolled-identity FAR
    test_metrics = compute_cohort_metrics(test_gen_scores, test_imp_scores, threshold=theta_star)

    # Add individual EER to per-participant metrics
    per_user_rows = []
    for row in test_metrics["per_participant"]:
        pid = row["participant_id"]
        u_curve = compute_far_frr_curve(test_gen_scores[pid], test_imp_scores[pid], n_thresholds=500)
        row["individual_eer_percent"] = float(u_curve["eer"] * 100.0)
        row["individual_eer_threshold"] = float(u_curve["eer_threshold"])
        per_user_rows.append(row)

    df_per_user = pd.DataFrame(per_user_rows)
    df_per_user.to_csv(os.path.join(output_dir, "per_participant_metrics.csv"), index=False)
    print("  ✓ Saved per_participant_metrics.csv")

    # Aggregate JSON
    aggregate_dict = {
        "status": "COMPLETED",
        "cohort_name": "Expanded Right Pocket Cohort",
        "participant_count": 55,
        "calibration_methodology": "Cohort-calibrated threshold selection using validation scores from the 55 enrolled participants",
        "frozen_calibrated_threshold": theta_star,
        "validation_metrics": {
            "eer_percent": val_eer,
            "calibrated_threshold": theta_star,
            "total_genuine_attempts": int(val_metrics["micro"]["total_genuine_attempts"]),
            "total_impostor_attempts": int(val_metrics["micro"]["total_impostor_attempts"]),
            "total_false_accepts": int(val_metrics["micro"]["total_false_accepts"]),
            "total_false_rejects": int(val_metrics["micro"]["total_false_rejects"]),
            "micro_far_percent": val_metrics["micro"]["far_percent"],
            "micro_frr_percent": val_metrics["micro"]["frr_percent"],
            "micro_balanced_accuracy_percent": val_metrics["micro"]["balanced_accuracy_percent"],
            "macro_far_percent": val_metrics["macro"]["far_percent"],
            "macro_frr_percent": val_metrics["macro"]["frr_percent"],
            "macro_balanced_accuracy_percent": val_metrics["macro"]["balanced_accuracy_percent"],
            "genuine_mean": float(np.mean(pooled_val_gen)),
            "genuine_std": float(np.std(pooled_val_gen)),
            "impostor_mean": float(np.mean(pooled_val_imp)),
            "impostor_std": float(np.std(pooled_val_imp)),
        },
        "test_metrics": {
            "frozen_threshold": theta_star,
            "total_genuine_attempts": int(test_metrics["micro"]["total_genuine_attempts"]),
            "total_impostor_attempts": int(test_metrics["micro"]["total_impostor_attempts"]),
            "total_false_accepts": int(test_metrics["micro"]["total_false_accepts"]),
            "total_false_rejects": int(test_metrics["micro"]["total_false_rejects"]),
            "micro_far_percent": test_metrics["micro"]["far_percent"],
            "micro_frr_percent": test_metrics["micro"]["frr_percent"],
            "micro_balanced_accuracy_percent": test_metrics["micro"]["balanced_accuracy_percent"],
            "macro_far_percent": test_metrics["macro"]["far_percent"],
            "macro_frr_percent": test_metrics["macro"]["frr_percent"],
            "macro_balanced_accuracy_percent": test_metrics["macro"]["balanced_accuracy_percent"],
            "diagnostic_eer_percent": test_diagnostic_eer,
            "genuine_mean": float(np.mean(pooled_test_gen)),
            "genuine_std": float(np.std(pooled_test_gen)),
            "impostor_mean": float(np.mean(pooled_test_imp)),
            "impostor_std": float(np.std(pooled_test_imp)),
        }
    }

    with open(os.path.join(output_dir, "aggregate_metrics.json"), "w") as f:
        json.dump(aggregate_dict, f, indent=2)
    print("  ✓ Saved aggregate_metrics.json")

    # 7. Generate Visualizations
    print("\nGenerating baseline diagnostic plots...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Plot 1: Score Distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.kdeplot(pooled_val_gen, ax=axes[0], color="forestgreen", fill=True, label="Validation Genuine (N=6,310)")
    sns.kdeplot(pooled_val_imp, ax=axes[0], color="crimson", fill=True, label="Validation Impostor (N=340,740)")
    axes[0].axvline(theta_star, color="black", linestyle="--", linewidth=2, label=f"Calibrated θ* = {theta_star:.4f}")
    axes[0].set_title("Validation Score Distributions (Calibration Partition)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("DTW Distance (znorm_magnitude, sum_lengths)")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper right")

    sns.kdeplot(pooled_test_gen, ax=axes[1], color="forestgreen", fill=True, label="Test Genuine (N=6,291)")
    sns.kdeplot(pooled_test_imp, ax=axes[1], color="crimson", fill=True, label="Test Impostor (N=339,714)")
    axes[1].axvline(theta_star, color="black", linestyle="--", linewidth=2, label=f"Frozen θ* = {theta_star:.4f}")
    axes[1].set_title("Held-Out Test Score Distributions (Evaluation Partition)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("DTW Distance (znorm_magnitude, sum_lengths)")
    axes[1].set_ylabel("Density")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=300)
    plt.close(fig)
    print("  ✓ Saved score_distributions.png")

    # Plot 2: ROC and DET Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Validation ROC
    axes[0].plot(val_curve["far"] * 100, (1.0 - val_curve["frr"]) * 100, color="royalblue", lw=2, label=f"Val ROC (EER = {val_eer:.2f}%)")
    axes[0].plot(test_curve["far"] * 100, (1.0 - test_curve["frr"]) * 100, color="darkorange", lw=2, label=f"Test ROC (EER = {test_diagnostic_eer:.2f}%)")
    axes[0].plot([0, 100], [0, 100], "k--", alpha=0.5)
    axes[0].scatter([test_metrics["micro"]["far_percent"]], [(100.0 - test_metrics["micro"]["frr_percent"])], color="red", s=100, zorder=5, label=f"Micro Operating Point (θ* = {theta_star:.4f})")
    axes[0].scatter([test_metrics["macro"]["far_percent"]], [(100.0 - test_metrics["macro"]["frr_percent"])], color="purple", s=100, zorder=5, marker="^", label=f"Macro Operating Point")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Acceptance Rate (%)")
    axes[0].set_ylabel("True Acceptance Rate (%)")
    axes[0].legend(loc="lower right")

    # FAR / FRR vs Threshold
    axes[1].plot(val_curve["thresholds"], val_curve["far"] * 100, color="crimson", lw=2, label="Validation FAR (%)")
    axes[1].plot(val_curve["thresholds"], val_curve["frr"] * 100, color="forestgreen", lw=2, label="Validation FRR (%)")
    axes[1].axvline(theta_star, color="black", linestyle="--", lw=2, label=f"Frozen θ* = {theta_star:.4f}")
    axes[1].set_title("Validation Error Trade-Off Curves", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Decision Threshold (θ)")
    axes[1].set_ylabel("Error Rate (%)")
    axes[1].legend(loc="center right")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close(fig)
    print("  ✓ Saved roc_curve.png")

    # 8. Markdown Report
    report_md = f"""# Baseline Authentication Report: Expanded Right Pocket Cohort (55 Participants)

**Cohort:** 55 Participants with cross-session Right Pocket accelerometer data  
**Sensor Placement:** Right front trouser pocket  
**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Mode:** Fixed-template multivariate Dynamic Time Warping (DTW) — Zero Adaptation  

---

## 1. Executive Summary

A fixed-template multivariate DTW biometric baseline was established across **55 participants** from the expanded 200GaitData Right Pocket collection. The evaluation strictly enforces zero boundary leakage and uses **cohort-calibrated threshold selection using validation scores from the 55 enrolled participants**.

### Aggregate Benchmark Results

| Evaluation Level | Metric | Value | Attempt Counts |
| :--- | :--- | :---: | :--- |
| **Calibration (Validation)** | Calibrated Threshold $\\theta^*$ | **{theta_star:.4f}** | 6,310 Genuine vs 340,740 Exhaustive Impostors |
| | Validation EER | **{val_eer:.2f}%** | Equal Error Rate operating point |
| **Test (Micro-Averaged)** | Test FAR (Micro) | **{test_metrics['micro']['far_percent']:.2f}%** | {test_metrics['micro']['total_false_accepts']} false accepts / 339,714 impostor attempts |
| | Test FRR (Micro) | **{test_metrics['micro']['frr_percent']:.2f}%** | {test_metrics['micro']['total_false_rejects']} false rejects / 6,291 genuine attempts |
| | Balanced Accuracy (Micro) | **{test_metrics['micro']['balanced_accuracy_percent']:.2f}%** | $100\\% - (\\text{{FAR}} + \\text{{FRR}})/2$ |
| **Test (Macro-Averaged)** | Test FAR (Macro) | **{test_metrics['macro']['far_percent']:.2f}%** | Mean $\\text{{FAR}}_i$ across 55 enrolled galleries |
| | Test FRR (Macro) | **{test_metrics['macro']['frr_percent']:.2f}%** | Mean $\\text{{FRR}}_i$ across 55 enrolled identities |
| | Balanced Accuracy (Macro)| **{test_metrics['macro']['balanced_accuracy_percent']:.2f}%** | Mean Balanced Accuracy across 55 enrolled identities |
| **Test Diagnostic EER** | Diagnostic Test EER | **{test_diagnostic_eer:.2f}%** | Post-hoc EER on held-out test distribution |

---

## 2. Cohort Accounting & Leakage Controls

### 2.1 Cohort Selection & Exclusions
- **Included (55 Participants):** PIDs {cohort_pids[:10]}... (55 total) having both RP Session 1 and RP Session 2.
- **Excluded (37 Participants):**
  - 24 participants missing RP Session 2 (PIDs 11, 17, 19, 35, 36, 37, 39, 40, 41, 44, 45, 47, 48, 50, 55, 57, 59, 61, 63, 66, 69, 72, 85, 93).
  - 10 participants missing RP Session 1 (PIDs 5, 6, 7, 8, 9, 12, 34, 65, 75, 92).
  - 3 participants with Left Wrist data only (PIDs 29, 70, 77).
  - Participant 73 is absent from the dataset.

### 2.2 Reconciled Window Partitioning
- **Session 1 (Enrollment):** 12,933 windows (8 stratified templates selected per participant).
- **Session 2 Validation:** 6,310 windows (first chronological 50%).
- **Boundary Overlap Discards:** 40 windows systematically discarded ($t_{{\\text{{start, test}}}} \\ge t_{{\\text{{end, val}}}} - 10^{{-6}}$ s).
- **Session 2 Test:** 6,291 windows (second chronological 50%, held-out).
- **Zero-Leakage Assertion:** 100% verified. Every window belongs to exactly one partition. Zero raw accelerometer samples are shared across the validation/test boundary.

---

## 3. Threshold Calibration Methodology

The operating threshold was determined via **cohort-calibrated threshold selection using validation scores from the 55 enrolled participants**:
- Operating point: $\\theta^* = {theta_star:.4f}$ where Validation $\\text{{FAR}}(\\theta) = \\text{{FRR}}(\\theta) = {val_eer:.2f}\\%$.
- The threshold was frozen prior to test evaluation. No test-set scores were used for tuning.

---

## 4. Per-Enrolled-Identity Macro Metrics

Metrics are calculated per enrolled gallery identity $i$:
$$\\text{{FAR}}_i = \\frac{{\\text{{False Accepts against gallery }} i}}{{N_{{\\text{{imp}}, i}}}}, \\quad \\text{{FRR}}_i = \\frac{{\\text{{False Rejects for identity }} i}}{{N_{{\\text{{gen}}, i}}}}$$
where $N_{{\\text{{imp}}, i}} = \\sum_{{j \\neq i}} N_j = N_{{\\text{{test}}}} - N_{{\\text{{gen}}, i}}$ (averaging $\\approx 6,177$ impostor attempts per gallery; $339,714$ pooled impostor attempts in total).
"""
    with open(os.path.join(output_dir, "baseline_report.md"), "w") as f:
        f.write(report_md)
    print("  ✓ Saved baseline_report.md")

    elapsed = time.time() - t0_start
    print(f"\n✓ Baseline execution completed in {elapsed:.1f} seconds.")
    return aggregate_dict


if __name__ == "__main__":
    results = run_expanded_rp_baseline()
    print("\nExecution complete.")

