"""
evaluation.py — Biometric Evaluation Metrics & Visualizations
==============================================================
Project : Gait-Based Biometric Authentication (Phase 2)
Dataset : 200GaitData

This module evaluates the authentication performance:
  1. Score distribution analysis: Genuine vs Impostor statistics.
  2. Operating characteristic curves: FAR and FRR across thresholds.
  3. Equal Error Rate (EER) and optimal operating threshold determination.
  4. Authentication accuracy computation.
  5. Publication-quality diagnostic plots saved to results/:
     - Preprocessed vs original raw signals (filtering comparison).
     - Example 2-second gait windows from different subjects.
     - Genuine vs Impostor DTW distance distribution histograms/KDE.
     - FAR vs FRR threshold trade-off curve & ROC curve.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.segmentation import GaitWindow


def compute_far_frr_curve(
    genuine_scores: list[float] | np.ndarray,
    impostor_scores: list[float] | np.ndarray,
    n_thresholds: int = 1000
) -> dict:
    """
    Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    across a continuous sweep of authentication thresholds.

    In distance-based biometric verification (lower is closer/better):
      - A claim is ACCEPTED as genuine if:  score <= threshold
      - A claim is REJECTED as impostor if: score >  threshold

    Therefore:
      - FAR(θ) = P(impostor_score <= θ) = # {imp <= θ} / N_imp
      - FRR(θ) = P(genuine_score > θ)   = # {gen > θ}  / N_gen

    Parameters
    ----------
    genuine_scores : array-like
        DTW matching scores for authentic claimants.
    impostor_scores : array-like
        DTW matching scores for impostor claimants.
    n_thresholds : int
        Number of threshold evaluation points (default: 1000).

    Returns
    -------
    dict
        Dictionary containing thresholds, FAR, FRR, accuracies, EER, and optimal threshold.
    """
    gen = np.asarray(genuine_scores, dtype=np.float64)
    imp = np.asarray(impostor_scores, dtype=np.float64)

    n_gen = len(gen)
    n_imp = len(imp)

    if n_gen == 0 or n_imp == 0:
        raise ValueError("Genuine and impostor score lists must both be non-empty.")

    # Sweep thresholds across the entire dynamic range of observed distances
    min_score = min(float(gen.min()), float(imp.min()))
    max_score = max(float(gen.max()), float(imp.max()))
    padding = 0.05 * (max_score - min_score + 1e-9)

    thresholds = np.linspace(min_score - padding, max_score + padding, n_thresholds)

    far_list = np.zeros(n_thresholds)
    frr_list = np.zeros(n_thresholds)
    acc_list = np.zeros(n_thresholds)
    bal_acc_list = np.zeros(n_thresholds)

    # Pre-sort for fast empirical CDF evaluation
    gen_sorted = np.sort(gen)
    imp_sorted = np.sort(imp)

    for i, theta in enumerate(thresholds):
        # Impostors accepted: impostor score <= theta
        accepted_impostors = np.searchsorted(imp_sorted, theta, side="right")
        far = accepted_impostors / n_imp

        # Genuines rejected: genuine score > theta
        accepted_genuines = np.searchsorted(gen_sorted, theta, side="right")
        rejected_genuines = n_gen - accepted_genuines
        frr = rejected_genuines / n_gen

        # Classification accuracy
        correct = accepted_genuines + (n_imp - accepted_impostors)
        acc = correct / (n_gen + n_imp)
        # Balanced accuracy = (True Accept Rate + True Reject Rate) / 2
        # = ((1 - FRR) + (1 - FAR)) / 2 = 1 - (FAR + FRR) / 2
        bal_acc = 1.0 - (far + frr) / 2.0

        far_list[i] = far
        frr_list[i] = frr
        acc_list[i] = acc
        bal_acc_list[i] = bal_acc

    # Equal Error Rate (EER): index where |FAR - FRR| is minimal
    # If there is a plateau of zero error (perfect separation), pick the midpoint of the plateau
    diff = np.abs(far_list - frr_list)
    min_diff = np.min(diff)
    candidate_indices = np.where(diff <= min_diff + 1e-9)[0]
    eer_idx = int(candidate_indices[len(candidate_indices) // 2])

    eer_threshold = float(thresholds[eer_idx])
    eer_far = float(far_list[eer_idx])
    eer_frr = float(frr_list[eer_idx])
    eer_val = float((eer_far + eer_frr) / 2.0)
    eer_acc = float(acc_list[eer_idx])
    eer_bal_acc = float(bal_acc_list[eer_idx])

    # Maximum accuracy operating point
    best_acc_idx = int(np.argmax(acc_list))
    best_acc_threshold = float(thresholds[best_acc_idx])
    best_acc = float(acc_list[best_acc_idx])

    # Maximum balanced accuracy operating point
    best_bal_acc_idx = int(np.argmax(bal_acc_list))
    best_bal_acc_threshold = float(thresholds[best_bal_acc_idx])
    best_bal_acc = float(bal_acc_list[best_bal_acc_idx])

    return {
        "thresholds": thresholds,
        "far": far_list,
        "frr": frr_list,
        "accuracies": acc_list,
        "balanced_accuracies": bal_acc_list,
        "eer": eer_val,
        "eer_threshold": eer_threshold,
        "eer_far": eer_far,
        "eer_frr": eer_frr,
        "eer_accuracy": eer_acc,
        "eer_balanced_accuracy": eer_bal_acc,
        "optimal_accuracy": best_acc,
        "optimal_threshold": best_acc_threshold,
        "optimal_balanced_accuracy": best_bal_acc,
        "optimal_balanced_threshold": best_bal_acc_threshold,
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "genuine_stats": {
            "mean": float(np.mean(gen)),
            "std": float(np.std(gen)),
            "median": float(np.median(gen)),
            "min": float(np.min(gen)),
            "max": float(np.max(gen)),
        },
        "impostor_stats": {
            "mean": float(np.mean(imp)),
            "std": float(np.std(imp)),
            "median": float(np.median(imp)),
            "min": float(np.min(imp)),
            "max": float(np.max(imp)),
        },
    }


def evaluate_fixed_threshold(
    genuine_scores: list[float] | np.ndarray,
    impostor_scores: list[float] | np.ndarray,
    threshold: float
) -> dict:
    """
    Evaluate biometric performance on held-out test data using a fixed decision threshold
    previously tuned on an independent validation set (zero leakage / no snooping).

    Parameters
    ----------
    genuine_scores : array-like
        Test genuine matching distances.
    impostor_scores : array-like
        Test impostor matching distances.
    threshold : float
        Decision threshold selected from validation data.

    Returns
    -------
    dict
        Evaluation results containing FAR, FRR, accuracy, and balanced accuracy.
    """
    gen = np.asarray(genuine_scores, dtype=np.float64)
    imp = np.asarray(impostor_scores, dtype=np.float64)

    n_gen = len(gen)
    n_imp = len(imp)

    if n_gen == 0 or n_imp == 0:
        raise ValueError("Score lists must both be non-empty.")

    false_accepts = int(np.sum(imp <= threshold))
    far = false_accepts / n_imp
    false_rejects = int(np.sum(gen > threshold))
    frr = false_rejects / n_gen

    true_accepts = n_gen - false_rejects
    true_rejects = n_imp - false_accepts

    accuracy = (true_accepts + true_rejects) / (n_gen + n_imp)
    balanced_accuracy = 1.0 - (far + frr) / 2.0

    return {
        "threshold": float(threshold),
        "far": float(far),
        "frr": float(frr),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "true_accepts": true_accepts,
        "false_rejects": false_rejects,
        "true_rejects": true_rejects,
        "false_accepts": false_accepts,
    }


def compute_cohort_metrics(
    genuine_scores_by_user: dict[int, list[float] | np.ndarray],
    impostor_scores_against_gallery: dict[int, list[float] | np.ndarray],
    threshold: float
) -> dict:
    """
    Compute micro- and macro-averaged biometric authentication metrics across a cohort,
    where FAR is strictly evaluated per enrolled identity (gallery).

    For each enrolled participant i:
      - Genuine attempts: Session 2 windows belonging to i, evaluated against gallery i.
      - False rejects: Genuine attempts where score > threshold.
      - Impostor attempts: All windows belonging to the other participants (j != i),
        evaluated against gallery i.
      - False accepts: Impostor attempts where score <= threshold.

    Calculates:
      FAR_i = False Accepts against gallery i / Total impostor attempts against gallery i
      FRR_i = False Rejects for identity i / Total genuine attempts for identity i
      Macro FAR = Mean(FAR_i) across identities
      Macro FRR = Mean(FRR_i) across identities
      Macro Balanced Accuracy = 100% - (Macro FAR + Macro FRR) / 2
      Micro FAR = Sum(False Accepts) / Sum(Impostor Attempts)
      Micro FRR = Sum(False Rejects) / Sum(Genuine Attempts)
      Micro Balanced Accuracy = 100% - (Micro FAR + Micro FRR) / 2

    Parameters
    ----------
    genuine_scores_by_user : dict[int, array-like]
        Mapping from participant_id -> genuine verification distances.
    impostor_scores_against_gallery : dict[int, array-like]
        Mapping from participant_id -> impostor verification distances against gallery i.
    threshold : float
        Decision threshold.

    Returns
    -------
    dict
        Structured results with keys:
          - 'per_participant': list of dicts with individual metrics
          - 'macro': dict with macro_far, macro_frr, macro_balanced_accuracy
          - 'micro': dict with micro_far, micro_frr, micro_balanced_accuracy, totals
    """
    pids = sorted(genuine_scores_by_user.keys())
    per_user = []

    total_fa = 0
    total_fr = 0
    total_gen = 0
    total_imp = 0

    far_list = []
    frr_list = []
    bal_acc_list = []

    for pid in pids:
        gen = np.asarray(genuine_scores_by_user[pid], dtype=np.float64)
        imp = np.asarray(impostor_scores_against_gallery[pid], dtype=np.float64)

        n_gen_i = len(gen)
        n_imp_i = len(imp)

        if n_gen_i == 0:
            raise ValueError(f"Participant {pid} has 0 genuine attempts.")
        if n_imp_i == 0:
            raise ValueError(f"Gallery {pid} has 0 impostor attempts.")

        fa_i = int(np.sum(imp <= threshold))
        fr_i = int(np.sum(gen > threshold))

        far_i = fa_i / n_imp_i
        frr_i = fr_i / n_gen_i
        bal_acc_i = 1.0 - (far_i + frr_i) / 2.0

        far_list.append(far_i)
        frr_list.append(frr_i)
        bal_acc_list.append(bal_acc_i)

        total_fa += fa_i
        total_fr += fr_i
        total_gen += n_gen_i
        total_imp += n_imp_i

        per_user.append({
            "participant_id": pid,
            "n_genuine": n_gen_i,
            "n_impostor": n_imp_i,
            "false_accepts": fa_i,
            "false_rejects": fr_i,
            "far": float(far_i),
            "frr": float(frr_i),
            "far_percent": float(far_i * 100.0),
            "frr_percent": float(frr_i * 100.0),
            "balanced_accuracy_percent": float(bal_acc_i * 100.0),
        })

    macro_far = float(np.mean(far_list))
    macro_frr = float(np.mean(frr_list))
    macro_bal_acc = float(np.mean(bal_acc_list))

    micro_far = total_fa / total_imp if total_imp > 0 else 0.0
    micro_frr = total_fr / total_gen if total_gen > 0 else 0.0
    micro_bal_acc = 1.0 - (micro_far + micro_frr) / 2.0

    return {
        "threshold": float(threshold),
        "n_participants": len(pids),
        "macro": {
            "far": macro_far,
            "frr": macro_frr,
            "far_percent": macro_far * 100.0,
            "frr_percent": macro_frr * 100.0,
            "balanced_accuracy_percent": macro_bal_acc * 100.0,
        },
        "micro": {
            "total_genuine_attempts": total_gen,
            "total_impostor_attempts": total_imp,
            "total_false_accepts": total_fa,
            "total_false_rejects": total_fr,
            "far": float(micro_far),
            "frr": float(micro_frr),
            "far_percent": float(micro_far * 100.0),
            "frr_percent": float(micro_frr * 100.0),
            "balanced_accuracy_percent": float(micro_bal_acc * 100.0),
        },
        "per_participant": per_user,
    }


# ---------------------------------------------------------------------------
# 4 Visualizations
# ---------------------------------------------------------------------------

def plot_preprocessed_vs_original(
    processed_df: pd.DataFrame,
    save_path: str,
    n_samples: int = 1000
) -> None:
    """
    Plot 1: Side-by-side comparison of raw noisy signal vs 20 Hz filtered signal.
    Shows all three axes (X, Y, Z) and acceleration magnitude.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    subset = processed_df.head(n_samples)
    t = subset["time_rel"].to_numpy() if "time_rel" in subset.columns else np.arange(len(subset)) * 0.01

    axis_info = [
        ("X", "#e74c3c", "X-axis (Lateral acceleration)"),
        ("Y", "#2ecc71", "Y-axis (Vertical acceleration / Gravity)"),
        ("Z", "#3498db", "Z-axis (Forward progression)"),
        ("a_mag", "#9b59b6", "Acceleration Magnitude ||a||"),
    ]

    for ax, (col, color, title) in zip(axes, axis_info):
        raw_col = f"{col}_raw"
        if raw_col in subset.columns:
            ax.plot(t, subset[raw_col], color="gray", alpha=0.55, linewidth=0.9, label="Raw signal")
        ax.plot(t, subset[col], color=color, linewidth=1.4, label="Butterworth 20Hz filtered")
        ax.set_ylabel(f"{col} (g)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9)

    axes[-1].set_xlabel("Time (seconds from segment start)", fontsize=11)
    rec_id = processed_df.get("stream_segment_id", pd.Series(["Recording"])).iloc[0]
    plt.suptitle(
        f"Signal Preprocessing Verification — Raw vs 4th-order Butterworth Filtered ({rec_id})",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved plot: {save_path}")


def plot_example_gait_windows(
    windows: list[GaitWindow],
    save_path: str,
    n_examples: int = 4
) -> None:
    """
    Plot 2: Visual gallery of representative 2.0-second gait windows from
    different participants, showing rhythmic stride cycles and individuality.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    # Pick windows from different participants
    seen_pids = set()
    sampled = []
    for w in windows:
        if w.participant_id not in seen_pids:
            sampled.append(w)
            seen_pids.add(w.participant_id)
            if len(sampled) >= n_examples:
                break

    fig, axes = plt.subplots(len(sampled), 1, figsize=(14, 2.8 * len(sampled)), sharex=True)
    if len(sampled) == 1:
        axes = [axes]

    colors = {"X": "#e74c3c", "Y": "#2ecc71", "Z": "#3498db"}

    for ax, w in zip(axes, sampled):
        t = np.linspace(0, 2.0, w.data.shape[0])
        if w.data.ndim == 1 or w.data.shape[1] == 1:
            val_col = w.data[:, 0] if w.data.ndim > 1 else w.data
            ax.plot(t, val_col, label="Feature Signal", color="#9b59b6", linewidth=1.4)
        else:
            for idx, axis_name in enumerate(["X", "Y", "Z"][:w.data.shape[1]]):
                ax.plot(t, w.data[:, idx], label=axis_name, color=colors.get(axis_name, "black"), linewidth=1.3)
            if w.data.shape[1] == 3:
                ax.plot(t, w.magnitude, label="||a|| Mag", color="#34495e", linestyle="--", linewidth=1.1, alpha=0.7)
        ax.set_title(
            f"Participant {w.participant_id} | Session {w.session} ({w.label}) | Window ID: {w.window_id}",
            fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("Accel (g)")
        ax.legend(loc="upper right", ncol=4, framealpha=0.9)

    axes[-1].set_xlabel("Window Elapsed Time (seconds)", fontsize=11)
    plt.suptitle(
        "Extracted 2.0-Second Gait Windows (200 samples @ 100 Hz, 50% Overlap)",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved plot: {save_path}")


def plot_genuine_vs_impostor_distributions(
    genuine_scores: list[float] | np.ndarray,
    impostor_scores: list[float] | np.ndarray,
    eer_threshold: float,
    save_path: str
) -> None:
    """
    Plot 3: Overlaid probability distributions (histogram + KDE) of DTW distances
    for Genuine and Impostor authentication attempts, marking the decision boundary.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(12, 6))

    gen = np.asarray(genuine_scores)
    imp = np.asarray(impostor_scores)

    # Histogram & KDE
    sns.histplot(
        gen, kde=True, stat="density", color="#2ecc71", label="Genuine Attempts (Same User)",
        bins=40, alpha=0.45, ax=ax
    )
    sns.histplot(
        imp, kde=True, stat="density", color="#e74c3c", label="Impostor Attempts (Different Users)",
        bins=40, alpha=0.45, ax=ax
    )

    # Mark the EER decision threshold
    ax.axvline(
        eer_threshold, color="#2c3e50", linestyle="--", linewidth=2.0,
        label=f"EER Threshold (θ = {eer_threshold:.4f})"
    )

    ax.set_xlabel("Normalized Multivariate DTW Distance", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title(
        "Authentication Score Distributions — Genuine vs Impostor DTW Distances",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)

    # Add text annotation with mean distances
    info_text = (
        f"Genuine:  mean={gen.mean():.3f} (σ={gen.std():.3f})\n"
        f"Impostor: mean={imp.mean():.3f} (σ={imp.std():.3f})\n"
        f"Separation margin: {imp.mean() - gen.mean():.3f}"
    )
    ax.text(
        0.03, 0.95, info_text, transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
        fontsize=10, family="monospace"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved plot: {save_path}")


def plot_far_frr_roc_curves(
    eval_dict: dict,
    save_path: str
) -> None:
    """
    Plot 4: Two-panel diagnostic of biometric error tradeoffs:
      - Left: FAR and FRR curves vs Decision Threshold θ (EER intersection).
      - Right: Receiver Operating Characteristic (ROC) curve: Detection Rate (1-FRR) vs FAR.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    thresholds = eval_dict["thresholds"]
    far = eval_dict["far"]
    frr = eval_dict["frr"]
    eer_thresh = eval_dict["eer_threshold"]
    eer_val = eval_dict["eer"]

    # --- Left: FAR vs FRR across threshold ---
    ax1.plot(thresholds, far * 100, color="#e74c3c", linewidth=2.0, label="FAR (False Acceptance Rate %)")
    ax1.plot(thresholds, frr * 100, color="#3498db", linewidth=2.0, label="FRR (False Rejection Rate %)")
    ax1.plot(thresholds, eval_dict["accuracies"] * 100, color="#2ecc71", linestyle=":", linewidth=1.5, label="Accuracy %")

    # Mark EER
    ax1.scatter([eer_thresh], [eer_val * 100], color="#8e44ad", s=80, zorder=5)
    ax1.axvline(eer_thresh, color="gray", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Authentication Distance Threshold (θ)", fontsize=11)
    ax1.set_ylabel("Error Rate (%)", fontsize=11)
    ax1.set_title(f"FAR & FRR vs Threshold (EER = {eer_val*100:.2f}%)", fontsize=12, fontweight="bold")
    ax1.set_ylim(-2, 102)
    ax1.legend(loc="center right", framealpha=0.9)

    # --- Right: ROC Curve (Detection Rate vs False Acceptance Rate) ---
    tpr = 1.0 - frr
    ax2.plot(far * 100, tpr * 100, color="#2980b9", linewidth=2.2, label="DTW Gait Authenticator")
    ax2.plot([0, 100], [0, 100], color="#7f8c8d", linestyle="--", linewidth=1.2, label="Random Guess (EER=50%)")

    # Mark EER point on ROC
    ax2.scatter([eval_dict["eer_far"] * 100], [(1.0 - eval_dict["eer_frr"]) * 100],
                color="#8e44ad", s=80, zorder=5, label=f"EER Operating Point ({eer_val*100:.1f}%)")

    ax2.set_xlabel("False Acceptance Rate — FAR (%)", fontsize=11)
    ax2.set_ylabel("True Acceptance Rate — TAR (1 - FRR %)", fontsize=11)
    ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax2.set_xlim(-2, 102)
    ax2.set_ylim(-2, 102)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.suptitle("Baseline DTW Gait Biometric Verification Performance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved plot: {save_path}")


def save_metrics_summary(
    eval_dict: dict,
    json_path: str,
    csv_path: str
) -> None:
    """Save scalar evaluation metrics to JSON and CSV files."""
    summary = {
        "n_genuine_attempts": int(eval_dict["n_genuine"]),
        "n_impostor_attempts": int(eval_dict["n_impostor"]),
        "eer": round(float(eval_dict["eer"]), 4),
        "eer_percent": round(float(eval_dict["eer"] * 100), 2),
        "eer_threshold": round(float(eval_dict["eer_threshold"]), 4),
        "eer_far": round(float(eval_dict["eer_far"]), 4),
        "eer_frr": round(float(eval_dict["eer_frr"]), 4),
        "eer_accuracy_percent": round(float(eval_dict["eer_accuracy"] * 100), 2),
        "eer_balanced_accuracy_percent": round(float(eval_dict.get("eer_balanced_accuracy", 1.0 - eval_dict["eer"]) * 100), 2),
        "optimal_threshold": round(float(eval_dict["optimal_threshold"]), 4),
        "optimal_accuracy_percent": round(float(eval_dict["optimal_accuracy"] * 100), 2),
        "optimal_balanced_accuracy_percent": round(float(eval_dict.get("optimal_balanced_accuracy", 1.0 - eval_dict["eer"]) * 100), 2),
        "genuine_mean_distance": round(float(eval_dict["genuine_stats"]["mean"]), 4),
        "genuine_std_distance": round(float(eval_dict["genuine_stats"]["std"]), 4),
        "impostor_mean_distance": round(float(eval_dict["impostor_stats"]["mean"]), 4),
        "impostor_std_distance": round(float(eval_dict["impostor_stats"]["std"]), 4),
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    print(f"  ✓ Metrics saved to: {json_path} and {csv_path}")
