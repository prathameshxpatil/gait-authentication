"""
fair_adaptive_comparison.py — Exhaustive Stream Evaluation: Fixed Baseline vs Dual-Gallery
==========================================================================================
Project : Gait-Based Biometric Authentication (Fair Comparative Evaluation)
Dataset : 200GaitData (Expanded Right Pocket Accelerometer, 55 Participants)

Methodology:
  1. Cohort: 55 participants having both RP Session 1 (enrollment) and RP Session 2 (authentication).
  2. Data: Session 1 for permanent enrollment (8 stratified templates per user).
           Session 2 Validation Partition ONLY (6,310 genuine windows).
           Zero exposure to Held-Out Test partition.
  3. Exhaustive Evaluation:
     Every validation window is evaluated against ALL 55 enrolled galleries:
       - Genuine attempt against owner's gallery (6,310 trials pooled).
       - Impostor attempt against all 54 non-owner galleries (340,740 trials pooled).
       - Total evaluated comparisons: exactly 347,050 trials per run.
  4. Stateful Chronological Processing:
     For each enrolled gallery i:
       - Genuine attempts arrive in physical chronological sequence.
       - Impostor attempts (from all other 54 users) are interleaved across the timeline
         using deterministic random seeds (42, 100, 2024).
       - Galleries maintain continuous state across all 6,310 incoming stream attempts.
  5. Strategies Compared at Frozen Threshold theta* = 0.13362185565710222:
       (a) Fixed-Template DTW Baseline (immutable 8 templates, zero updates).
       (b) Dual-Gallery Standard (w_perm=0.70, M=4, K=3, alpha=0.70, gamma=0.85, beta=1.00).
       (c) Dual-Gallery Conservative (w_perm=0.80, M=4, K=3, alpha=0.65, gamma=0.80, beta=0.85).
  6. Outputs:
       - Micro (pooled 347,050 attempts) and Macro (per-gallery mean) metrics.
       - Master CSVs, JSON summaries, audit logs, and diagnostic plots in results/fair_adaptive_comparison/.
"""

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
from src.dual_gallery import (
    CandidateCluster,
    CandidateBuffer,
    DualGallery,
)
from src.adaptive_template import are_windows_overlapping


# ---------------------------------------------------------------------------
# 1. Precomputing Permanent Matching Matrix
# ---------------------------------------------------------------------------

def _worker_precompute_gallery_perm(args):
    """Computes DTW distance of all validation windows against a gallery's permanent templates."""
    gallery_pid, permanent_templates, windows_data, window_radius, norm_mode = args
    scores = np.empty(len(windows_data), dtype=np.float32)
    for idx, w_mat in enumerate(windows_data):
        d = min(
            multivariate_dtw_distance(w_mat, t, window_radius=window_radius, norm_mode=norm_mode)
            for t in permanent_templates
        )
        scores[idx] = d
    return gallery_pid, scores


def precompute_all_permanent_distances(
    permanent_templates_dict: dict[int, list[np.ndarray]],
    all_val_windows: list[GaitWindow],
    window_radius: int = 30,
    norm_mode: str = "sum_lengths",
    n_workers: int = 8,
) -> dict[int, np.ndarray]:
    """
    Precomputes DTW matching distance of every validation window against the permanent
    templates of each of the 55 galleries.
    """
    pids = sorted(permanent_templates_dict.keys())
    windows_data = [w.data for w in all_val_windows]

    tasks = [
        (pid, permanent_templates_dict[pid], windows_data, window_radius, norm_mode)
        for pid in pids
    ]

    print("  • Precomputing permanent gallery DTW distances (347,050 evaluations)...")
    t0 = time.time()
    perm_dist_matrix = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for pid, scores in executor.map(_worker_precompute_gallery_perm, tasks):
            perm_dist_matrix[pid] = scores
    elapsed = time.time() - t0
    print(f"  ✓ Precomputed permanent distance matrix in {elapsed:.1f}s.")
    return perm_dist_matrix


# ---------------------------------------------------------------------------
# 2. Stateful Stream Evaluation per Gallery
# ---------------------------------------------------------------------------

def _worker_evaluate_gallery_stream(args):
    """
    Evaluates a single gallery over its complete chronological stream of 6,310 attempts.
    """
    (
        gallery_pid,
        strategy_type,  # "fixed", "dual_standard", "dual_conservative"
        permanent_templates,
        stream_meta,    # list of (win_idx, claimant_pid, is_genuine)
        val_windows,    # list of all 6310 GaitWindow objects
        perm_scores,    # precomputed float32 array of length 6310
        threshold,
        strategy_params,
    ) = args

    # Initialize gallery state
    if strategy_type == "fixed":
        dg = None
    else:
        dg = DualGallery(
            participant_id=gallery_pid,
            permanent_templates=permanent_templates,
            max_adaptive_capacity=strategy_params.get("max_adaptive_capacity", 4),
            w_perm=strategy_params.get("w_perm", 0.70),
            candidate_target_size=strategy_params.get("candidate_target_size", 3),
            dtw_radius=30,
            norm_mode="sum_lengths",
            replacement_strategy="redundancy",
            max_candidate_age_steps=strategy_params.get("max_candidate_age_steps", 20),
            max_buffer_capacity=strategy_params.get("max_buffer_capacity", 5),
        )

    alpha = strategy_params.get("alpha_high_confidence", 0.70)
    beta = strategy_params.get("beta_anchor_guard", 1.00)
    gamma = strategy_params.get("gamma_cluster_consistency", 0.85)

    n_gen = 0
    n_imp = 0
    true_accepts = 0
    false_accepts = 0
    false_rejects = 0
    total_updates = 0
    genuine_updates = 0
    poisoned_updates = 0
    drift_history = [0.0]

    for step, (win_idx, claimant_pid, is_genuine) in enumerate(stream_meta, start=1):
        d_perm = float(perm_scores[win_idx])

        if strategy_type == "fixed":
            score_comb = d_perm
            accepted = bool(score_comb <= threshold)
            promoted = False
            is_poisoned = False
            current_drift = 0.0
        else:
            w_obj = val_windows[win_idx]
            w_mat = w_obj.data
            # If adaptive gallery has templates, compute DTW to adaptive templates
            if not dg.adaptive_templates:
                d_adapt = d_perm
            else:
                d_adapt = min(
                    multivariate_dtw_distance(w_mat, at, window_radius=30, norm_mode="sum_lengths")
                    for at in dg.adaptive_templates
                )
            score_comb = dg.w_perm * d_perm + (1.0 - dg.w_perm) * d_adapt
            accepted = bool(score_comb <= threshold)
            is_high_conf = bool(score_comb <= alpha * threshold)

            promoted = False
            is_poisoned = False

            if not is_high_conf:
                # Quarantined candidates flushed on degraded attempt
                if len(dg.candidate_buffer) > 0:
                    dg.candidate_buffer.clear()
            else:
                appended, _ = dg.candidate_buffer.add_candidate(
                    window=w_obj,
                    score=score_comb,
                    step=step,
                    is_genuine=is_genuine,
                )

                if appended and dg.candidate_buffer.is_ready():
                    cluster = dg.candidate_buffer.get_cluster()
                    _, max_pair_d = cluster.cluster_consistency(dtw_radius=30, norm_mode="sum_lengths")

                    if max_pair_d > gamma * threshold:
                        dg.candidate_buffer.clear()
                    else:
                        _, medoid_w, medoid_is_gen, _ = cluster.compute_medoid(dtw_radius=30, norm_mode="sum_lengths")
                        # Anchor guard
                        anchor_dists = sorted([
                            multivariate_dtw_distance(medoid_w.data, pt, window_radius=30, norm_mode="sum_lengths")
                            for pt in dg.permanent_templates
                        ])
                        top2_mean = (anchor_dists[0] + anchor_dists[1]) / 2.0 if len(anchor_dists) >= 2 else anchor_dists[0]

                        if top2_mean > beta * threshold:
                            dg.candidate_buffer.clear()
                        else:
                            promoted = True
                            is_poisoned = not bool(medoid_is_gen)
                            total_updates += 1
                            if is_poisoned:
                                poisoned_updates += 1
                            else:
                                genuine_updates += 1

                            if len(dg._adaptive_templates) < dg.max_adaptive_capacity:
                                dg._adaptive_templates.append(np.copy(medoid_w.data))
                            else:
                                # Redundancy replacement
                                dists = [
                                    multivariate_dtw_distance(medoid_w.data, at, window_radius=30, norm_mode="sum_lengths")
                                    for at in dg._adaptive_templates
                                ]
                                r_idx = int(np.argmin(dists))
                                dg._adaptive_templates[r_idx] = np.copy(medoid_w.data)

                            dg.candidate_buffer.clear()

            current_drift = dg.compute_current_drift()
            drift_history.append(current_drift)

        # Metric tracking
        if is_genuine:
            n_gen += 1
            if accepted:
                true_accepts += 1
            else:
                false_rejects += 1
        else:
            n_imp += 1
            if accepted:
                false_accepts += 1

    far_pct = (false_accepts / n_imp * 100.0) if n_imp > 0 else 0.0
    frr_pct = (false_rejects / n_gen * 100.0) if n_gen > 0 else 0.0
    bal_acc_pct = 100.0 - (far_pct + frr_pct) / 2.0
    final_drift = float(drift_history[-1])

    result = {
        "gallery_pid": gallery_pid,
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "true_accepts": true_accepts,
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
        "far_percent": far_pct,
        "frr_percent": frr_pct,
        "balanced_accuracy_percent": bal_acc_pct,
        "total_updates": total_updates,
        "genuine_updates": genuine_updates,
        "poisoned_updates": poisoned_updates,
        "final_drift": final_drift,
        "drift_history": drift_history,
    }
    return gallery_pid, result


# ---------------------------------------------------------------------------
# 3. Master Runner for Fair Adaptive Comparison
# ---------------------------------------------------------------------------

def run_fair_adaptive_comparison(
    output_dir: str = "results/fair_adaptive_comparison",
    seeds: list[int] = [42, 100, 2024],
    threshold: float = 0.13362185565710222,
    n_workers: int = 8,
):
    """
    Executes the fair comparison evaluating every validation window against all 55 galleries.
    """
    t0_all = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 88)
    print("  FAIR EXHAUSTIVE ADAPTIVE COMPARISON (55 PARTICIPANTS)")
    print("=" * 88)
    print(f"Target Output Directory : {output_dir}")
    print(f"Baseline Threshold (θ*) : {threshold:.4f}")
    print(f"Random Seeds Evaluated  : {seeds}")
    print(f"Evaluation Mode         : Exhaustive (Every Validation Window against All 55 Galleries)")
    print(f"Total Trials per Seed   : 6,310 Genuine + 340,740 Impostor = 347,050 Trials")

    # 1. Ingest Data & Extract Windows
    print("\n[Step 1/5] Ingesting Cohort & Extracting Windows...")
    manifest = load_manifest()
    rp_all = manifest[manifest["placement"] == "RP"]
    pids_s1 = set(rp_all[rp_all["session_int"] == 1]["participant_id"])
    pids_s2 = set(rp_all[rp_all["session_int"] == 2]["participant_id"])
    cohort_pids = sorted(list(pids_s1.intersection(pids_s2)))
    assert len(cohort_pids) == 55

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

    windows = transform_window_features(raw_windows, feature_mode="znorm_magnitude")
    enrol_dict, val_dict, test_dict = partition_enrol_val_test(windows, val_ratio=0.50)
    del test_dict  # Strict test-set isolation

    # Flatten all validation windows in fixed deterministic order
    all_val_windows = []
    for pid in cohort_pids:
        all_val_windows.extend(val_dict[pid])
    assert len(all_val_windows) == 6310

    # Map each window to its index, claimant_pid, and timestamp
    val_win_data = [w.data for w in all_val_windows]
    val_win_pids = [w.participant_id for w in all_val_windows]

    print(f"  ✓ Enrolled cohort: 55 participants ({sum(len(w) for w in enrol_dict.values())} S1 windows)")
    print(f"  ✓ Validation set: {len(all_val_windows)} windows across 55 participants")

    # 2. Build Permanent Enrollment Galleries (8 stratified templates)
    print("\n[Step 2/5] Building Permanent Enrollment Galleries (8 stratified templates)...")
    permanent_templates_dict = {}
    for pid in cohort_pids:
        s1_wins = enrol_dict[pid]
        if len(s1_wins) < 8:
            selected = s1_wins
        else:
            idx = np.linspace(0, len(s1_wins) - 1, 8, dtype=int)
            selected = [s1_wins[i] for i in idx]
        permanent_templates_dict[pid] = [w.data for w in selected]

    # 3. Precompute Permanent Distance Matrix
    print("\n[Step 3/5] Computing Permanent Distance Matrix...")
    perm_dist_matrix = precompute_all_permanent_distances(
        permanent_templates_dict=permanent_templates_dict,
        all_val_windows=all_val_windows,
        window_radius=30,
        norm_mode="sum_lengths",
        n_workers=n_workers,
    )

    # 4. Define the 3 Target Strategies
    strategies = [
        {
            "id": "fixed",
            "name": "Fixed Baseline",
            "params": {},
        },
        {
            "id": "dual_standard",
            "name": "Dual-Gallery Standard",
            "params": {
                "w_perm": 0.70,
                "max_adaptive_capacity": 4,
                "candidate_target_size": 3,
                "alpha_high_confidence": 0.70,
                "gamma_cluster_consistency": 0.85,
                "beta_anchor_guard": 1.00,
            },
        },
        {
            "id": "dual_conservative",
            "name": "Dual-Gallery Conservative",
            "params": {
                "w_perm": 0.80,
                "max_adaptive_capacity": 4,
                "candidate_target_size": 3,
                "alpha_high_confidence": 0.65,
                "gamma_cluster_consistency": 0.80,
                "beta_anchor_guard": 0.85,
            },
        },
    ]

    # 5. Execute Multi-Seed Exhaustive Streams
    print(f"\n[Step 4/5] Executing {len(strategies)} Strategies across {len(seeds)} Seeds ({len(strategies)*len(seeds)} runs)...")
    all_summary_rows = []
    per_user_rows = []
    drift_traces = {}

    for strat in strategies:
        strat_id = strat["id"]
        strat_name = strat["name"]
        print(f"\n--- Strategy: {strat_name} ---")

        for seed in seeds:
            t_seed0 = time.time()
            print(f"  • Running seed {seed}...", end="", flush=True)

            # Build streams for each gallery under this seed
            tasks = []
            for pid in cohort_pids:
                # Find indices of genuine windows for this gallery
                gen_indices = [idx for idx, p in enumerate(val_win_pids) if p == pid]
                imp_indices = [idx for idx, p in enumerate(val_win_pids) if p != pid]

                n_gen_pid = len(gen_indices)
                assert n_gen_pid == len(val_dict[pid])
                assert len(imp_indices) == 6310 - n_gen_pid

                # Interleave genuine windows with all impostors
                rng = np.random.default_rng(seed + pid)
                gen_items = [(float(i), idx, pid, True) for i, idx in enumerate(gen_indices)]
                imp_times = rng.uniform(0.0, float(n_gen_pid), size=len(imp_indices))
                imp_items = [(float(t), idx, val_win_pids[idx], False) for t, idx in zip(imp_times, imp_indices)]

                combined = sorted(gen_items + imp_items, key=lambda x: x[0])
                stream_meta = [(idx, cl, is_g) for (_, idx, cl, is_g) in combined]

                tasks.append((
                    pid,
                    strat_id,
                    permanent_templates_dict[pid],
                    stream_meta,
                    all_val_windows,
                    perm_dist_matrix[pid],
                    threshold,
                    strat["params"],
                ))

            # Run 55 galleries in parallel
            gallery_results = {}
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for pid, res in executor.map(_worker_evaluate_gallery_stream, tasks):
                    gallery_results[pid] = res

            # Aggregate cohort statistics
            tot_gen = sum(res["n_genuine"] for res in gallery_results.values())
            tot_imp = sum(res["n_impostor"] for res in gallery_results.values())
            tot_ta = sum(res["true_accepts"] for res in gallery_results.values())
            tot_fr = sum(res["false_rejects"] for res in gallery_results.values())
            tot_fa = sum(res["false_accepts"] for res in gallery_results.values())
            tot_upd = sum(res["total_updates"] for res in gallery_results.values())
            tot_poi = sum(res["poisoned_updates"] for res in gallery_results.values())
            tot_gup = sum(res["genuine_updates"] for res in gallery_results.values())

            assert tot_gen == 6310, f"Expected 6,310 genuine attempts, got {tot_gen}"
            assert tot_imp == 340740, f"Expected 340,740 impostor attempts, got {tot_imp}"

            micro_far = (tot_fa / tot_imp) * 100.0
            micro_frr = (tot_fr / tot_gen) * 100.0
            micro_bal_acc = 100.0 - (micro_far + micro_frr) / 2.0

            macro_far = float(np.mean([res["far_percent"] for res in gallery_results.values()]))
            macro_frr = float(np.mean([res["frr_percent"] for res in gallery_results.values()]))
            macro_bal_acc = float(np.mean([res["balanced_accuracy_percent"] for res in gallery_results.values()]))

            drifts = [res["final_drift"] for res in gallery_results.values()]
            mean_drift = float(np.mean(drifts))
            max_drift = float(np.max(drifts))
            poi_rate = (tot_poi / tot_upd * 100.0) if tot_upd > 0 else 0.0

            elapsed_seed = time.time() - t_seed0
            print(f" Done ({elapsed_seed:.1f}s) | Micro BalAcc: {micro_bal_acc:.2f}% | FAR: {micro_far:.2f}% | FRR: {micro_frr:.2f}% | Updates: {tot_upd} (Poisoned: {tot_poi})")

            all_summary_rows.append({
                "strategy": strat_name,
                "seed": seed,
                "runtime_sec": round(elapsed_seed, 2),
                "total_genuine": tot_gen,
                "total_impostor": tot_imp,
                "true_accepts": tot_ta,
                "false_rejects": tot_fr,
                "false_accepts": tot_fa,
                "micro_far_percent": round(micro_far, 2),
                "micro_frr_percent": round(micro_frr, 2),
                "micro_balanced_accuracy_percent": round(micro_bal_acc, 2),
                "macro_far_percent": round(macro_far, 2),
                "macro_frr_percent": round(macro_frr, 2),
                "macro_balanced_accuracy_percent": round(macro_bal_acc, 2),
                "total_updates": tot_upd,
                "genuine_updates": tot_gup,
                "poisoned_updates": tot_poi,
                "poisoning_rate_percent": round(poi_rate, 2),
                "mean_drift": round(mean_drift, 5),
                "max_drift": round(max_drift, 5),
            })

            # Record per-user rows
            for pid, res in gallery_results.items():
                per_user_rows.append({
                    "strategy": strat_name,
                    "seed": seed,
                    "participant_id": pid,
                    "n_genuine": res["n_genuine"],
                    "n_impostor": res["n_impostor"],
                    "true_accepts": res["true_accepts"],
                    "false_rejects": res["false_rejects"],
                    "false_accepts": res["false_accepts"],
                    "far_percent": round(res["far_percent"], 4),
                    "frr_percent": round(res["frr_percent"], 4),
                    "balanced_accuracy_percent": round(res["balanced_accuracy_percent"], 4),
                    "total_updates": res["total_updates"],
                    "genuine_updates": res["genuine_updates"],
                    "poisoned_updates": res["poisoned_updates"],
                    "final_drift": round(res["final_drift"], 5),
                })

            if seed == 42:
                # Capture drift trajectory for sample participants
                for sample_p in [1, 2, 10, 15, 20, 80]:
                    drift_traces[f"{strat_name}_pid{sample_p}"] = gallery_results[sample_p]["drift_history"]

    # 6. Save Tables & Compute Aggregates
    print("\n[Step 5/5] Compiling Results & Generating Reports...")
    df_summary = pd.DataFrame(all_summary_rows)
    df_per_user = pd.DataFrame(per_user_rows)

    df_summary.to_csv(os.path.join(output_dir, "fair_comparison_summary.csv"), index=False)
    df_per_user.to_csv(os.path.join(output_dir, "per_participant_fair_comparison.csv"), index=False)
    print("  ✓ Saved fair_comparison_summary.csv")
    print("  ✓ Saved per_participant_fair_comparison.csv")

    # Aggregate JSON (Mean ± Std)
    agg_json = {}
    for strat_name, grp in df_summary.groupby("strategy"):
        agg_json[strat_name] = {
            "micro_bal_acc_mean": float(grp["micro_balanced_accuracy_percent"].mean()),
            "micro_bal_acc_std": float(grp["micro_balanced_accuracy_percent"].std()),
            "micro_far_mean": float(grp["micro_far_percent"].mean()),
            "micro_far_std": float(grp["micro_far_percent"].std()),
            "micro_frr_mean": float(grp["micro_frr_percent"].mean()),
            "micro_frr_std": float(grp["micro_frr_percent"].std()),
            "macro_bal_acc_mean": float(grp["macro_balanced_accuracy_percent"].mean()),
            "macro_bal_acc_std": float(grp["macro_balanced_accuracy_percent"].std()),
            "macro_far_mean": float(grp["macro_far_percent"].mean()),
            "macro_far_std": float(grp["macro_far_percent"].std()),
            "macro_frr_mean": float(grp["macro_frr_percent"].mean()),
            "macro_frr_std": float(grp["macro_frr_percent"].std()),
            "true_accepts_mean": float(grp["true_accepts"].mean()),
            "false_rejects_mean": float(grp["false_rejects"].mean()),
            "false_accepts_mean": float(grp["false_accepts"].mean()),
            "total_updates_mean": float(grp["total_updates"].mean()),
            "genuine_updates_mean": float(grp["genuine_updates"].mean()),
            "poisoned_updates_mean": float(grp["poisoned_updates"].mean()),
            "poisoning_rate_mean": float(grp["poisoning_rate_percent"].mean()),
            "mean_drift_mean": float(grp["mean_drift"].mean()),
            "max_drift_mean": float(grp["max_drift"].mean()),
        }

    with open(os.path.join(output_dir, "strategy_metrics_summary.json"), "w") as f:
        json.dump(agg_json, f, indent=2)
    print("  ✓ Saved strategy_metrics_summary.json")

    # 7. Generate Visualizations
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    strat_order = ["Fixed Baseline", "Dual-Gallery Standard", "Dual-Gallery Conservative"]

    # Balanced Accuracy
    sns.barplot(data=df_summary, x="strategy", y="micro_balanced_accuracy_percent", order=strat_order, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Micro Balanced Accuracy (%)", fontweight="bold")
    axes[0].set_ylabel("Balanced Accuracy (%)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=15)

    # FAR vs FRR
    df_melt = df_summary.melt(id_vars=["strategy", "seed"], value_vars=["micro_far_percent", "micro_frr_percent"], var_name="Metric", value_name="Error Rate (%)")
    df_melt["Metric"] = df_melt["Metric"].replace({"micro_far_percent": "FAR (%)", "micro_frr_percent": "FRR (%)"})
    sns.barplot(data=df_melt, x="strategy", y="Error Rate (%)", hue="Metric", order=strat_order, ax=axes[1], palette=["crimson", "forestgreen"])
    axes[1].set_title("FAR vs FRR Comparison", fontweight="bold")
    axes[1].set_ylabel("Error Rate (%)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=15)

    # Updates and Poisoning
    df_upd = df_summary.melt(id_vars=["strategy", "seed"], value_vars=["genuine_updates", "poisoned_updates"], var_name="Type", value_name="Updates")
    df_upd["Type"] = df_upd["Type"].replace({"genuine_updates": "Genuine Updates", "poisoned_updates": "Poisoned Updates"})
    sns.barplot(data=df_upd, x="strategy", y="Updates", hue="Type", order=strat_order, ax=axes[2], palette=["royalblue", "firebrick"])
    axes[2].set_title("Exhaustive Stream Updates & Poisoning", fontweight="bold")
    axes[2].set_ylabel("Count across 55 Galleries")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparative_bars.png"), dpi=300)
    plt.close(fig)
    print("  ✓ Saved comparative_bars.png")

    # Plot Drift Trajectories
    fig, ax = plt.subplots(figsize=(12, 6))
    for key, trace in drift_traces.items():
        if "pid80" in key or "pid1" in key:
            ax.plot(trace, label=key, lw=2)
    ax.set_title("Gallery Drift Across Exhaustive 6,310-Attempt Streams", fontweight="bold")
    ax.set_xlabel("Stream Step (Authentication Attempts)")
    ax.set_ylabel("Gallery Drift (DTW Distance)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "drift_over_time.png"), dpi=300)
    plt.close(fig)
    print("  ✓ Saved drift_over_time.png")

    # 8. Markdown Report
    report_md = f"""# Fair Exhaustive Comparison: Fixed Baseline vs Dual-Gallery Adaptation

**Cohort:** 55 Enrolled Participants (Right Pocket Collection)  
**Evaluation Partition:** Session 2 Validation Stream ONLY (Held-Out Test Partition 100% Isolated)  
**Evaluation Protocol:** Exhaustive Matching (Every validation window tested against all 55 galleries: 6,310 genuine + 340,740 impostor = 347,050 trials per run)  
**Frozen Operating Threshold:** $\\theta^* = {threshold:.4f}$  
**Evaluated Seeds:** {seeds}  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Executive Summary & Aggregate Benchmark

| Evaluation Metric | Fixed Baseline | Dual-Gallery Standard | Dual-Gallery Conservative |
| :--- | :---: | :---: | :---: |
| **Micro Balanced Accuracy (%)** | **{agg_json['Fixed Baseline']['micro_bal_acc_mean']:.2f} ± {agg_json['Fixed Baseline']['micro_bal_acc_std']:.2f}** | **{agg_json['Dual-Gallery Standard']['micro_bal_acc_mean']:.2f} ± {agg_json['Dual-Gallery Standard']['micro_bal_acc_std']:.2f}** | **{agg_json['Dual-Gallery Conservative']['micro_bal_acc_mean']:.2f} ± {agg_json['Dual-Gallery Conservative']['micro_bal_acc_std']:.2f}** |
| **Micro FAR (%)** | {agg_json['Fixed Baseline']['micro_far_mean']:.2f} ± {agg_json['Fixed Baseline']['micro_far_std']:.2f} | {agg_json['Dual-Gallery Standard']['micro_far_mean']:.2f} ± {agg_json['Dual-Gallery Standard']['micro_far_std']:.2f} | {agg_json['Dual-Gallery Conservative']['micro_far_mean']:.2f} ± {agg_json['Dual-Gallery Conservative']['micro_far_std']:.2f} |
| **Micro FRR (%)** | {agg_json['Fixed Baseline']['micro_frr_mean']:.2f} ± {agg_json['Fixed Baseline']['micro_frr_std']:.2f} | {agg_json['Dual-Gallery Standard']['micro_frr_mean']:.2f} ± {agg_json['Dual-Gallery Standard']['micro_frr_std']:.2f} | {agg_json['Dual-Gallery Conservative']['micro_frr_mean']:.2f} ± {agg_json['Dual-Gallery Conservative']['micro_frr_std']:.2f} |
| **Macro Balanced Accuracy (%)** | {agg_json['Fixed Baseline']['macro_bal_acc_mean']:.2f} ± {agg_json['Fixed Baseline']['macro_bal_acc_std']:.2f} | {agg_json['Dual-Gallery Standard']['macro_bal_acc_mean']:.2f} ± {agg_json['Dual-Gallery Standard']['macro_bal_acc_std']:.2f} | {agg_json['Dual-Gallery Conservative']['macro_bal_acc_mean']:.2f} ± {agg_json['Dual-Gallery Conservative']['macro_bal_acc_std']:.2f} |
| **Macro FAR (%)** | {agg_json['Fixed Baseline']['macro_far_mean']:.2f} ± {agg_json['Fixed Baseline']['macro_far_std']:.2f} | {agg_json['Dual-Gallery Standard']['macro_far_mean']:.2f} ± {agg_json['Dual-Gallery Standard']['macro_far_std']:.2f} | {agg_json['Dual-Gallery Conservative']['macro_far_mean']:.2f} ± {agg_json['Dual-Gallery Conservative']['macro_far_std']:.2f} |
| **Macro FRR (%)** | {agg_json['Fixed Baseline']['macro_frr_mean']:.2f} ± {agg_json['Fixed Baseline']['macro_frr_std']:.2f} | {agg_json['Dual-Gallery Standard']['macro_frr_mean']:.2f} ± {agg_json['Dual-Gallery Standard']['macro_frr_std']:.2f} | {agg_json['Dual-Gallery Conservative']['macro_frr_mean']:.2f} ± {agg_json['Dual-Gallery Conservative']['macro_frr_std']:.2f} |
| **Genuine Accepts ($N_{{gen}}=6,310$)** | {int(agg_json['Fixed Baseline']['true_accepts_mean'])} | {int(agg_json['Dual-Gallery Standard']['true_accepts_mean'])} | {int(agg_json['Dual-Gallery Conservative']['true_accepts_mean'])} |
| **False Accepts ($N_{{imp}}=340,740$)**| {int(agg_json['Fixed Baseline']['false_accepts_mean'])} | {int(agg_json['Dual-Gallery Standard']['false_accepts_mean'])} | {int(agg_json['Dual-Gallery Conservative']['false_accepts_mean'])} |
| **Total Accepted Updates** | 0.0 | {agg_json['Dual-Gallery Standard']['total_updates_mean']:.1f} | {agg_json['Dual-Gallery Conservative']['total_updates_mean']:.1f} |
| **Poisoned Updates** | **0.0 (0.0%)** | **{agg_json['Dual-Gallery Standard']['poisoned_updates_mean']:.1f} ({agg_json['Dual-Gallery Standard']['poisoning_rate_mean']:.1f}%)** | **{agg_json['Dual-Gallery Conservative']['poisoned_updates_mean']:.1f} ({agg_json['Dual-Gallery Conservative']['poisoning_rate_mean']:.1f}%)** |
| **Mean Gallery Drift** | 0.0000 | {agg_json['Dual-Gallery Standard']['mean_drift_mean']:.4f} | {agg_json['Dual-Gallery Conservative']['mean_drift_mean']:.4f} |

---

## 2. Methodological Differences: Exhaustive Evaluation vs 1:1 Representative Experiment

| Protocol Aspect | Earlier 1:1 Representative Experiment | Fair Exhaustive Evaluation |
| :--- | :--- | :--- |
| **Impostor Sampling** | 1:1 balanced sampling ($N_{{imp}} = N_{{gen}} \\approx 115$ per user). | **Exhaustive Matching:** Every non-owner window ($N_{{imp}} = 6,310 - N_i \\approx 6,195$ per user). |
| **Total Trials per Seed** | 12,620 authentication trials ($6,310 \\text{{ gen}} + 6,310 \\text{{ imp}}$). | **347,050 authentication trials** ($6,310 \\text{{ gen}} + 340,740 \\text{{ imp}}$). |
| **Impostor Prevalence** | 50.0% ($1$ impostor attempt per $1$ genuine attempt). | **98.2%** ($54$ impostor attempts per $1$ genuine attempt). |
| **Adversarial Stress** | Low-density intrusion; long periods of pure genuine walking. | **High-density adversarial bombardment** (54× higher attack frequency). |
| **Buffer Disruption Pressure** | Infrequent buffer quarantine events. | Degraded impostor attempts arrive rapidly, continuously testing quarantine resilience. |
| **Baseline Comparability** | Baseline evaluated on same 1:1 stream (FAR = 26.18%). | Baseline evaluates all 340,740 non-owner trials, **matching the exact validation baseline denominator** (FAR = 28.87%). |
"""
    with open(os.path.join(output_dir, "fair_comparison_report.md"), "w") as f:
        f.write(report_md)
    print("  ✓ Saved fair_comparison_report.md")

    total_time = time.time() - t0_all
    print(f"\n✓ Fair adaptive comparison completed in {total_time:.1f} seconds.")
    return agg_json


if __name__ == "__main__":
    run_fair_adaptive_comparison()
