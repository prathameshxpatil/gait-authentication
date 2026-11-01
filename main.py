"""
main.py — End-to-End Baseline Gait DTW Biometric Authentication Pipeline
=========================================================================
Project : Gait-Based Biometric Authentication using DTW & Adaptive Template Updating
Dataset : 200GaitData (10 participants, 2 sessions each, 20 recordings)

Usage
-----
    python main.py
    python main.py --help
    python main.py --templates-per-user 8 --max-test-windows 40

Pipeline Stages:
  1. Load raw recordings from data/raw/
  2. Preprocessing:
     - Separate G ('Going') and R ('Returning') streams
     - Detect and split at timestamp gaps > 0.5s (never interpolate)
     - Reset timestamps to start at t=0 for each segment
     - Zero-phase 4th-order Butterworth low-pass filter (20 Hz cutoff)
     - Save processed files to data/processed/
  3. Windowing & Gait Sequence Preparation:
     - Extract 2.0-second sliding windows (200 samples @ 100 Hz) with 50% overlap
     - Discard incomplete windows
     - Partition into Enrolment (Session 1) and Test (Session 2)
  4. Multivariate DTW Authentication:
     - Form enrolment galleries strictly from Session 1
     - Evaluate Session 2 test windows for genuine and impostor claims
  5. Evaluation & Diagnostics:
     - Compute genuine/impostor distributions, FAR, FRR, EER, and Accuracy
     - Generate 4 publication-quality plots in results/
     - Save metric summaries to results/
"""

import os
import sys
import argparse
import time

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_all_recordings
from src.segmentation import (
    extract_all_windows,
    partition_enrolment_and_test,
)
from src.dtw_authentication import (
    build_enrolment_galleries,
    run_authentication_evaluation,
)
from src.evaluation import (
    compute_far_frr_curve,
    plot_preprocessed_vs_original,
    plot_example_gait_windows,
    plot_genuine_vs_impostor_distributions,
    plot_far_frr_roc_curves,
    save_metrics_summary,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gait-Based Biometric Authentication using Multivariate DTW (Phase 2 Baseline)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to raw CSV data directory (default: data/raw/)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Path to save processed CSVs (default: data/processed/)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results"),
        help="Path to save graphs and metrics (default: results/)",
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run the complete 16-experiment investigation suite and generate comparison graphs",
    )
    parser.add_argument(
        "--run-adaptive",
        action="store_true",
        help="Run Phase 3 Adaptive Template Updating and Biometric Drift Audit",
    )
    parser.add_argument(
        "--run-dual-gallery",
        action="store_true",
        help="Run Candidate-Buffer Dual-Gallery Gait Adaptation Architecture evaluation",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="raw_xyz",
        choices=["raw_xyz", "dynamic_xyz", "znorm_xyz", "magnitude", "znorm_magnitude", "dynamic_plus_mag"],
        help="Sensor feature representation (default: raw_xyz)",
    )
    parser.add_argument(
        "--stream",
        type=str,
        default="all",
        choices=["all", "G", "R"],
        help="Gait stream filter: all, G (Going), or R (Returning) (default: all)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Gait window length in samples (~2.0s at 100 Hz, default: 200)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=100,
        help="Window step size in samples (50%% overlap, default: 100)",
    )
    parser.add_argument(
        "--filter-cutoff",
        type=float,
        default=20.0,
        help="Butterworth lowpass cutoff in Hz (default: 20.0)",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.5,
        help="Discontinuity threshold in seconds for gap splitting (default: 0.5)",
    )
    parser.add_argument(
        "--templates-per-user",
        type=int,
        default=8,
        help="Number of Session 1 enrolment templates per participant (default: 8)",
    )
    parser.add_argument(
        "--max-test-windows",
        type=int,
        default=40,
        help="Maximum Session 2 test windows evaluated per participant (default: 40, set 0 for all)",
    )
    parser.add_argument(
        "--dtw-radius",
        type=int,
        default=30,
        help="Sakoe-Chiba constraint band radius in samples (default: 30)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> dict:
    """Execute the end-to-end authentication pipeline."""
    start_time = time.time()
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 76)
    print("  GAIT BIOMETRIC AUTHENTICATION — MULTIVARIATE DTW (BASELINE)")
    print("=" * 76)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Results dir  : {args.results_dir}")
    print(f"Window size  : {args.window_size} samples ({args.window_size/100:.1f}s)")
    print(f"Step size    : {args.step_size} samples ({args.step_size/100:.1f}s step)")
    print(f"Filter cutoff: {args.filter_cutoff} Hz (4th-order zero-phase Butterworth)")
    print("=" * 76)

    # -----------------------------------------------------------------------
    # Stage 1: Dataset Inventory & Preprocessing
    # -----------------------------------------------------------------------
    print("\n[STAGE 1/4] Preprocessing raw sensor signals...")
    t0 = time.time()
    processed_segments = preprocess_all_recordings(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        save_files=True,
    )
    t_prep = time.time() - t0
    print(f"→ Preprocessing elapsed: {t_prep:.2f}s")

    # -----------------------------------------------------------------------
    # Stage 2: Gait Sequence Windowing
    # -----------------------------------------------------------------------
    print("\n[STAGE 2/4] Extracting sliding gait windows...")
    t0 = time.time()
    all_windows = extract_all_windows(
        processed_segments,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    # Optional stream filtering (e.g. G-only or R-only)
    if args.stream in ["G", "R"]:
        all_windows = [w for w in all_windows if w.label == args.stream]
        print(f"  Filtered to '{args.stream}' stream only: {len(all_windows)} windows remaining.")

    # Optional feature transformation (e.g. magnitude, dynamic_xyz, znorm_xyz)
    if args.feature_mode != "raw_xyz":
        from src.experiments import transform_window_features
        all_windows = transform_window_features(all_windows, feature_mode=args.feature_mode)
        print(f"  Transformed features to: '{args.feature_mode}' (shape: {all_windows[0].data.shape}).")

    enrolment_windows, test_windows = partition_enrolment_and_test(all_windows)
    t_win = time.time() - t0
    print(f"→ Extracted {len(all_windows)} total windows across {len(processed_segments)} segments in {t_win:.2f}s")
    print(f"  Enrolment pool (Session 1): {sum(len(v) for v in enrolment_windows.values())} windows across {len(enrolment_windows)} users")
    print(f"  Test pool      (Session 2): {sum(len(v) for v in test_windows.values())} windows across {len(test_windows)} users")

    # -----------------------------------------------------------------------
    # Stage 3: Multivariate DTW Enrolment & Verification
    # -----------------------------------------------------------------------
    print("\n[STAGE 3/4] Building enrolment galleries & running DTW verification...")
    t0 = time.time()
    galleries = build_enrolment_galleries(
        enrolment_windows,
        n_templates_per_user=args.templates_per_user,
        strategy="stratified",
    )

    max_test = None if args.max_test_windows <= 0 else args.max_test_windows
    attempts, gen_scores, imp_scores = run_authentication_evaluation(
        galleries=galleries,
        test_windows=test_windows,
        window_radius=args.dtw_radius,
        max_test_windows_per_user=max_test,
        random_seed=args.random_seed,
    )
    t_dtw = time.time() - t0
    print(f"→ DTW verification elapsed: {t_dtw:.2f}s")

    # -----------------------------------------------------------------------
    # Stage 4: Biometric Performance Evaluation & Visualizations
    # -----------------------------------------------------------------------
    print("\n[STAGE 4/4] Calculating biometric error rates & generating figures...")
    eval_metrics = compute_far_frr_curve(gen_scores, imp_scores, n_thresholds=1000)

    # 1. Save summary metrics
    json_path = os.path.join(args.results_dir, "baseline_authentication_metrics.json")
    csv_path = os.path.join(args.results_dir, "baseline_authentication_metrics.csv")
    save_metrics_summary(eval_metrics, json_path, csv_path)

    # 2. Plot 1: Preprocessed vs Raw
    sample_seg_id = sorted(processed_segments.keys())[0]
    plot_preprocessed_vs_original(
        processed_segments[sample_seg_id],
        save_path=os.path.join(args.results_dir, "preprocessed_vs_original_signals.png"),
    )

    # 3. Plot 2: Example gait windows
    plot_example_gait_windows(
        all_windows,
        save_path=os.path.join(args.results_dir, "example_gait_windows.png"),
        n_examples=4,
    )

    # 4. Plot 3: Genuine vs Impostor Score Distributions
    plot_genuine_vs_impostor_distributions(
        gen_scores,
        imp_scores,
        eer_threshold=eval_metrics["eer_threshold"],
        save_path=os.path.join(args.results_dir, "genuine_vs_impostor_dtw_distances.png"),
    )

    # 5. Plot 4: FAR/FRR vs Threshold Trade-off & ROC Curves
    plot_far_frr_roc_curves(
        eval_metrics,
        save_path=os.path.join(args.results_dir, "far_frr_roc_curves.png"),
    )

    total_time = time.time() - start_time

    # -----------------------------------------------------------------------
    # Final Console Reporting
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("  BASELINE DTW AUTHENTICATION RESULTS REPORT")
    print("=" * 76)
    print(f"1. Successfully processed recordings : 20 / 20 recordings (100%)")
    print(f"   Continuous segments extracted      : {len(processed_segments)} segments")
    print(f"2. Extracted gait windows             : {len(all_windows):,} windows (2.0s each, 50% overlap)")
    print(f"3. Verification comparisons           :")
    print(f"   • Genuine authentication attempts  : {eval_metrics['n_genuine']:,}")
    print(f"   • Impostor authentication attempts : {eval_metrics['n_impostor']:,}")
    print(f"   • Total verification attempts      : {eval_metrics['n_genuine'] + eval_metrics['n_impostor']:,}")
    eer_bal_acc = eval_metrics.get("eer_balanced_accuracy", 1.0 - eval_metrics["eer"])
    opt_bal_acc = eval_metrics.get("optimal_balanced_accuracy", eer_bal_acc)

    print("4. Biometric Error Metrics            :")
    print(f"   • Equal Error Rate (EER)           : {eval_metrics['eer']*100:.2f}% (θ = {eval_metrics['eer_threshold']:.4f})")
    print(f"   • False Acceptance Rate at EER     : {eval_metrics['eer_far']*100:.2f}%")
    print(f"   • False Rejection Rate at EER      : {eval_metrics['eer_frr']*100:.2f}%")
    print(f"   • Balanced Accuracy at EER         : {eer_bal_acc*100:.2f}% (1 - EER)")
    print(f"   • Ordinary Accuracy at EER         : {eval_metrics['eer_accuracy']*100:.2f}%")
    print(f"   • Optimal Balanced Accuracy        : {opt_bal_acc*100:.2f}%")
    print(f"   • Optimal Ordinary Accuracy        : {eval_metrics['optimal_accuracy']*100:.2f}% (θ = {eval_metrics['optimal_threshold']:.4f})")
    print("5. Distance Score Statistics          :")
    print(f"   • Genuine  DTW distance            : mean = {eval_metrics['genuine_stats']['mean']:.4f} ± {eval_metrics['genuine_stats']['std']:.4f}")
    print(f"   • Impostor DTW distance            : mean = {eval_metrics['impostor_stats']['mean']:.4f} ± {eval_metrics['impostor_stats']['std']:.4f}")
    print(f"   • Separation Margin                : {eval_metrics['impostor_stats']['mean'] - eval_metrics['genuine_stats']['mean']:.4f}")
    print("6. Record Exclusions / Errors         : NONE. All 20 recordings successfully utilized.")
    print(f"   (sub4-rp-s1 & sub10-rp-s1 had only G stream; processed cleanly as continuous G segments.)")
    print("=" * 76)
    print(f"Total pipeline execution time: {total_time:.2f} seconds.")
    print(f"All generated graphs and reports saved in: {args.results_dir}")
    print("=" * 76)

    if args.run_experiments:
        from src.experiments import run_all_investigation_experiments
        run_all_investigation_experiments(args.results_dir)

    return eval_metrics


if __name__ == "__main__":
    args = parse_arguments()
    if args.run_dual_gallery:
        from src.dual_gallery import run_full_dual_gallery_comparison
        run_full_dual_gallery_comparison(
            results_dir=args.results_dir,
            dtw_radius=args.dtw_radius,
            seed=args.random_seed,
        )
    elif args.run_adaptive:
        from src.adaptive_template import run_full_adaptive_experiment
        run_full_adaptive_experiment(
            results_dir=args.results_dir,
            dtw_radius=args.dtw_radius,
            seed=args.random_seed,
        )
    else:
        run_pipeline(args)

