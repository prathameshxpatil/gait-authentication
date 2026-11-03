"""
Corrected Adaptive Authentication Stream Validation.

Separates exhaustive verification scoring from operational adaptive candidate updates:
  1. Exhaustive verification scoring: Evaluates all 6,310 genuine and 340,740 impostor
     comparisons across all 55 participants for authoritative FAR, FRR, and balanced accuracy.
  2. Operational adaptive stream: Simulates a realistic sequential authentication stream
     for each enrolled identity with genuine walking bouts and scheduled impostor attacks
     claiming that identity, preserving candidate buffer and adaptive gallery states.

Evaluated Strategies:
  - Fixed-Template Baseline
  - Dual-Gallery Standard (w_perm=0.70, M=4, K=3, alpha=0.70, gamma=0.85, beta=1.00)
  - Dual-Gallery Conservative (w_perm=0.80, M=4, K=3, alpha=0.65, gamma=0.80, beta=0.85)

Cohort: Expanded 55-Participant Right Pocket (RP) Cohort.
Partition: Session 2 Validation Stream ONLY (Held-out test partition 100% isolated).
Seeds: [42, 100, 2024].
Operating Threshold: Frozen baseline threshold theta* = 0.13362185565710222 (~0.1336).
"""

import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_manifest
from src.preprocessing import get_expanded_processed_data_dir
from src.segmentation import (
    extract_windows_from_segment,
    GaitWindow,
)
from src.experiments import transform_window_features, partition_enrol_val_test
from src.dtw_authentication import multivariate_dtw_distance
from src.dual_gallery import DualGallery


# ---------------------------------------------------------------------------
# 1. Operational Stream Builder
# ---------------------------------------------------------------------------

def build_operational_user_stream(
    uid: int,
    all_pids: list[int],
    val_windows_dict: dict[int, list[GaitWindow]],
    seed: int,
) -> list[tuple[GaitWindow, int, bool, float]]:
    """
    Constructs a separate chronological authentication stream for enrolled user uid.

    - Genuine windows: All N_i Session 2 validation windows of uid in chronological order.
    - Impostor windows: Exactly N_i windows sampled evenly across all 54 other participants.
    - Attack Scheduling: Impostor arrival times t_imp ~ Uniform(0, N_i) parameterized
      by rng(seed + uid), interleaved and sorted with genuine timestamps t_gen = 0, 1, ..., N_i-1.

    Returns:
      List of (window, claimant_pid, is_genuine, timestamp) tuples.
    """
    gen_wins = val_windows_dict[uid]
    n_gen = len(gen_wins)

    other_pids = [p for p in all_pids if p != uid]
    rng = np.random.default_rng(seed + uid)

    # Sample exactly n_gen impostor attempts evenly across other participants
    samples_per_other = max(1, int(np.ceil(n_gen / len(other_pids))))
    imp_pool = []
    for op in other_pids:
        op_wins = val_windows_dict[op]
        # Take representative windows deterministically to maintain cohort coverage
        take_cnt = min(samples_per_other, len(op_wins))
        imp_pool.extend([(w, op) for w in op_wins[:take_cnt]])

    # Shuffle pool with seed and trim to exactly n_gen
    rng.shuffle(imp_pool)
    imp_wins = imp_pool[:n_gen]

    # Assign chronological timestamps: genuine at 0, 1, ..., n_gen - 1
    gen_items = [(float(i), w, uid, True) for i, w in enumerate(gen_wins)]
    imp_times = rng.uniform(0.0, float(n_gen), size=len(imp_wins))
    imp_items = [(float(t), w, op, False) for t, (w, op) in zip(imp_times, imp_wins)]

    # Combine and sort strictly by arrival timestamp
    combined = sorted(gen_items + imp_items, key=lambda x: x[0])
    stream = [(w, claimant, is_g, t) for (t, w, claimant, is_g) in combined]
    return stream


# ---------------------------------------------------------------------------
# 2. Precomputation of Permanent Distance Matrix
# ---------------------------------------------------------------------------

def _worker_precompute_row(args):
    pid, perm_templates, all_val_data, window_radius, norm_mode = args
    n_wins = len(all_val_data)
    dists = np.zeros(n_wins, dtype=np.float32)
    for j in range(n_wins):
        w = all_val_data[j]
        min_d = min(
            multivariate_dtw_distance(w, t, window_radius=window_radius, norm_mode=norm_mode)
            for t in perm_templates
        )
        dists[j] = float(min_d)
    return pid, dists


def precompute_all_permanent_distances(
    permanent_templates_dict: dict[int, list[np.ndarray]],
    all_val_windows: list[GaitWindow],
    window_radius: int = 30,
    norm_mode: str = "sum_lengths",
    n_workers: int = 8,
) -> dict[int, np.ndarray]:
    all_val_data = [w.data for w in all_val_windows]
    tasks = [
        (pid, permanent_templates_dict[pid], all_val_data, window_radius, norm_mode)
        for pid in sorted(permanent_templates_dict.keys())
    ]
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for pid, dist_arr in executor.map(_worker_precompute_row, tasks):
            results[pid] = dist_arr
    elapsed = time.time() - t0
    print(f"  ✓ Precomputed permanent distance matrix for {len(results)} galleries in {elapsed:.1f}s.")
    return results


# ---------------------------------------------------------------------------
# 3. Stateful Stream Evaluation Worker (per Enrolled Identity)
# ---------------------------------------------------------------------------

def _worker_evaluate_user_stream_and_exhaustive(args):
    """
    Evaluates:
      1. Operational stream adaptation for user gallery_pid.
      2. Exhaustive verification scoring against all 6,310 validation windows using
         the adapted gallery state.
    """
    (
        gallery_pid,
        strategy_type,
        permanent_templates,
        stream_tuples,      # (window, claimant_pid, is_genuine, timestamp)
        all_val_windows,    # list of all 6310 GaitWindow objects
        perm_scores,        # float32 array length 6310
        val_win_pids,       # list of length 6310
        threshold,
        strategy_params,
    ) = args

    # Initialize DualGallery or Fixed Gallery
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

    # -----------------------------------------------------------------------
    # Step A: Process Operational Authentication Stream (State Updates)
    # -----------------------------------------------------------------------
    stream_n_gen = 0
    stream_n_imp = 0
    stream_true_accepts = 0
    stream_false_rejects = 0
    stream_false_accepts = 0
    total_updates = 0
    genuine_updates = 0
    poisoned_updates = 0
    drift_history = [0.0]

    for step, (w_obj, claimant_pid, is_genuine, ts) in enumerate(stream_tuples, start=1):
        if strategy_type == "fixed":
            # Match directly to permanent templates
            d_perm = min(
                multivariate_dtw_distance(w_obj.data, pt, window_radius=30, norm_mode="sum_lengths")
                for pt in permanent_templates
            )
            score_comb = d_perm
            accepted = bool(score_comb <= threshold)
        else:
            # Dual-Gallery stateful evaluation
            (
                accepted,
                d_perm,
                d_adapt,
                score_comb,
                promoted,
                event_meta,
            ) = dg.evaluate_and_update(
                window=w_obj,
                threshold=threshold,
                alpha_high_confidence=alpha,
                beta_anchor_guard=beta,
                gamma_cluster_consistency=gamma,
                is_genuine_groundtruth=is_genuine,
                step=step,
                claimant_id=claimant_pid,
            )

            if promoted and event_meta and event_meta.get("event_type") == "candidate_cluster_promoted":
                total_updates += 1
                if event_meta.get("is_poisoned", False):
                    poisoned_updates += 1
                else:
                    genuine_updates += 1

            drift_history.append(dg.compute_current_drift())

        # Stream metric tracking
        if is_genuine:
            stream_n_gen += 1
            if accepted:
                stream_true_accepts += 1
            else:
                stream_false_rejects += 1
        else:
            stream_n_imp += 1
            if accepted:
                stream_false_accepts += 1

    final_drift = float(drift_history[-1])
    stream_far_pct = (stream_false_accepts / stream_n_imp * 100.0) if stream_n_imp > 0 else 0.0
    stream_frr_pct = (stream_false_rejects / stream_n_gen * 100.0) if stream_n_gen > 0 else 0.0
    stream_bal_acc = 100.0 - (stream_far_pct + stream_frr_pct) / 2.0

    # -----------------------------------------------------------------------
    # Step B: Exhaustive Verification Scoring against All 6,310 Validation Windows
    # -----------------------------------------------------------------------
    # Probe comparisons do NOT mutate the candidate buffer or gallery templates.
    exh_n_gen = 0
    exh_n_imp = 0
    exh_true_accepts = 0
    exh_false_rejects = 0
    exh_false_accepts = 0

    active_adaptive_tmpls = dg.adaptive_templates if dg is not None else []
    w_perm = dg.w_perm if dg is not None else 1.0

    n_total_val = len(all_val_windows)
    for w_idx in range(n_total_val):
        is_gen_probe = bool(val_win_pids[w_idx] == gallery_pid)
        d_p = float(perm_scores[w_idx])

        if not active_adaptive_tmpls:
            d_comb_probe = d_p
        else:
            w_data = all_val_windows[w_idx].data
            d_a = min(
                multivariate_dtw_distance(w_data, at, window_radius=30, norm_mode="sum_lengths")
                for at in active_adaptive_tmpls
            )
            d_comb_probe = w_perm * d_p + (1.0 - w_perm) * d_a

        acc_probe = bool(d_comb_probe <= threshold)

        if is_gen_probe:
            exh_n_gen += 1
            if acc_probe:
                exh_true_accepts += 1
            else:
                exh_false_rejects += 1
        else:
            exh_n_imp += 1
            if acc_probe:
                exh_false_accepts += 1

    exh_far_pct = (exh_false_accepts / exh_n_imp * 100.0) if exh_n_imp > 0 else 0.0
    exh_frr_pct = (exh_false_rejects / exh_n_gen * 100.0) if exh_n_gen > 0 else 0.0
    exh_bal_acc = 100.0 - (exh_far_pct + exh_frr_pct) / 2.0

    res = {
        "gallery_pid": gallery_pid,
        # Operational stream metrics
        "stream_n_gen": stream_n_gen,
        "stream_n_imp": stream_n_imp,
        "stream_true_accepts": stream_true_accepts,
        "stream_false_rejects": stream_false_rejects,
        "stream_false_accepts": stream_false_accepts,
        "stream_far_percent": stream_far_pct,
        "stream_frr_percent": stream_frr_pct,
        "stream_balanced_accuracy": stream_bal_acc,
        "total_updates": total_updates,
        "genuine_updates": genuine_updates,
        "poisoned_updates": poisoned_updates,
        "final_drift": final_drift,
        "drift_history": drift_history,
        # Exhaustive verification metrics
        "exh_n_gen": exh_n_gen,
        "exh_n_imp": exh_n_imp,
        "exh_true_accepts": exh_true_accepts,
        "exh_false_rejects": exh_false_rejects,
        "exh_false_accepts": exh_false_accepts,
        "exh_far_percent": exh_far_pct,
        "exh_frr_percent": exh_frr_pct,
        "exh_balanced_accuracy": exh_bal_acc,
    }
    return gallery_pid, res


# ---------------------------------------------------------------------------
# 4. Master Orchestrator
# ---------------------------------------------------------------------------

def run_adaptive_stream_validation(
    output_dir: str = "results/adaptive_stream_validation",
    seeds: list[int] = [42, 100, 2024],
    threshold: float = 0.13362185565710222,
    n_workers: int = 8,
):
    """
    Executes the corrected adaptive authentication stream validation.
    """
    t0_all = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 88)
    print("  CORRECTED ADAPTIVE AUTHENTICATION STREAM VALIDATION (55 PARTICIPANTS)")
    print("=" * 88)
    print(f"Output Directory       : {output_dir}")
    print(f"Baseline Threshold (θ*): {threshold:.6f}")
    print(f"Random Seeds Evaluated : {seeds}")
    print(f"Protocol Design        : Separated Operational Stream & Exhaustive Verification Scoring")

    # 1. Ingest Data & Extract Windows
    print("\n[Step 1/5] Ingesting Cohort & Partitioning Data...")
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
    val_win_pids = [w.participant_id for w in all_val_windows]

    print(f"  ✓ Cohort: 55 participants ({sum(len(w) for w in enrol_dict.values())} S1 enrollment windows)")
    print(f"  ✓ Validation set: {len(all_val_windows)} windows ({min(len(v) for v in val_dict.values())} to {max(len(v) for v in val_dict.values())} per participant)")

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
    print("\n[Step 3/5] Precomputing Permanent Distance Matrix (55 x 6,310)...")
    perm_dist_matrix = precompute_all_permanent_distances(
        permanent_templates_dict=permanent_templates_dict,
        all_val_windows=all_val_windows,
        window_radius=30,
        norm_mode="sum_lengths",
        n_workers=n_workers,
    )

    # 4. Strategy Definitions
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

    # 5. Execute Evaluation
    print(f"\n[Step 4/5] Executing {len(strategies)} Strategies across {len(seeds)} Seeds...")
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

            # Build operational streams for each participant
            tasks = []
            for pid in cohort_pids:
                stream_tuples = build_operational_user_stream(
                    uid=pid,
                    all_pids=cohort_pids,
                    val_windows_dict=val_dict,
                    seed=seed,
                )
                tasks.append((
                    pid,
                    strat_id,
                    permanent_templates_dict[pid],
                    stream_tuples,
                    all_val_windows,
                    perm_dist_matrix[pid],
                    val_win_pids,
                    threshold,
                    strat["params"],
                ))

            # Run in parallel across workers
            gallery_results = {}
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for pid, res in executor.map(_worker_evaluate_user_stream_and_exhaustive, tasks):
                    gallery_results[pid] = res

            t_seed_elapsed = time.time() - t_seed0
            print(f" done in {t_seed_elapsed:.1f}s.")

            # Aggregate Stream Metrics
            str_tot_gen = sum(r["stream_n_gen"] for r in gallery_results.values())
            str_tot_imp = sum(r["stream_n_imp"] for r in gallery_results.values())
            str_ta = sum(r["stream_true_accepts"] for r in gallery_results.values())
            str_fr = sum(r["stream_false_rejects"] for r in gallery_results.values())
            str_fa = sum(r["stream_false_accepts"] for r in gallery_results.values())
            str_tot_upd = sum(r["total_updates"] for r in gallery_results.values())
            str_gen_upd = sum(r["genuine_updates"] for r in gallery_results.values())
            str_psn_upd = sum(r["poisoned_updates"] for r in gallery_results.values())
            str_psn_rate = (str_psn_upd / str_tot_upd * 100.0) if str_tot_upd > 0 else 0.0

            str_micro_far = (str_fa / str_tot_imp * 100.0) if str_tot_imp > 0 else 0.0
            str_micro_frr = (str_fr / str_tot_gen * 100.0) if str_tot_gen > 0 else 0.0
            str_micro_bal = 100.0 - (str_micro_far + str_micro_frr) / 2.0

            str_macro_far = float(np.mean([r["stream_far_percent"] for r in gallery_results.values()]))
            str_macro_frr = float(np.mean([r["stream_frr_percent"] for r in gallery_results.values()]))
            str_macro_bal = float(np.mean([r["stream_balanced_accuracy"] for r in gallery_results.values()]))

            mean_drift = float(np.mean([r["final_drift"] for r in gallery_results.values()]))
            max_drift = float(np.max([r["final_drift"] for r in gallery_results.values()]))

            # Aggregate Exhaustive Verification Metrics
            exh_tot_gen = sum(r["exh_n_gen"] for r in gallery_results.values())
            exh_tot_imp = sum(r["exh_n_imp"] for r in gallery_results.values())
            exh_ta = sum(r["exh_true_accepts"] for r in gallery_results.values())
            exh_fr = sum(r["exh_false_rejects"] for r in gallery_results.values())
            exh_fa = sum(r["exh_false_accepts"] for r in gallery_results.values())

            exh_micro_far = (exh_fa / exh_tot_imp * 100.0) if exh_tot_imp > 0 else 0.0
            exh_micro_frr = (exh_fr / exh_tot_gen * 100.0) if exh_tot_gen > 0 else 0.0
            exh_micro_bal = 100.0 - (exh_micro_far + exh_micro_frr) / 2.0

            exh_macro_far = float(np.mean([r["exh_far_percent"] for r in gallery_results.values()]))
            exh_macro_frr = float(np.mean([r["exh_frr_percent"] for r in gallery_results.values()]))
            exh_macro_bal = float(np.mean([r["exh_balanced_accuracy"] for r in gallery_results.values()]))

            # Record Summary Row
            all_summary_rows.append({
                "strategy": strat_name,
                "seed": seed,
                "runtime_sec": round(t_seed_elapsed, 2),
                # Stream metrics
                "stream_total_genuine": str_tot_gen,
                "stream_total_impostor": str_tot_imp,
                "stream_true_accepts": str_ta,
                "stream_false_rejects": str_fr,
                "stream_false_accepts": str_fa,
                "stream_micro_far_percent": round(str_micro_far, 4),
                "stream_micro_frr_percent": round(str_micro_frr, 4),
                "stream_micro_balanced_accuracy": round(str_micro_bal, 4),
                "stream_macro_far_percent": round(str_macro_far, 4),
                "stream_macro_frr_percent": round(str_macro_frr, 4),
                "stream_macro_balanced_accuracy": round(str_macro_bal, 4),
                "total_updates": str_tot_upd,
                "genuine_updates": str_gen_upd,
                "poisoned_updates": str_psn_upd,
                "poisoning_rate_percent": round(str_psn_rate, 2),
                "mean_drift": round(mean_drift, 5),
                "max_drift": round(max_drift, 5),
                # Exhaustive verification metrics
                "exh_total_genuine": exh_tot_gen,
                "exh_total_impostor": exh_tot_imp,
                "exh_true_accepts": exh_ta,
                "exh_false_rejects": exh_fr,
                "exh_false_accepts": exh_fa,
                "exh_micro_far_percent": round(exh_micro_far, 4),
                "exh_micro_frr_percent": round(exh_micro_frr, 4),
                "exh_micro_balanced_accuracy": round(exh_micro_bal, 4),
                "exh_macro_far_percent": round(exh_macro_far, 4),
                "exh_macro_frr_percent": round(exh_macro_frr, 4),
                "exh_macro_balanced_accuracy": round(exh_macro_bal, 4),
            })

            # Record per-participant rows
            for pid, r in gallery_results.items():
                per_user_rows.append({
                    "strategy": strat_name,
                    "seed": seed,
                    "participant_id": pid,
                    "stream_genuine": r["stream_n_gen"],
                    "stream_impostor": r["stream_n_imp"],
                    "stream_far_percent": round(r["stream_far_percent"], 4),
                    "stream_frr_percent": round(r["stream_frr_percent"], 4),
                    "stream_balanced_accuracy": round(r["stream_balanced_accuracy"], 4),
                    "total_updates": r["total_updates"],
                    "genuine_updates": r["genuine_updates"],
                    "poisoned_updates": r["poisoned_updates"],
                    "final_drift": round(r["final_drift"], 5),
                    "exh_genuine": r["exh_n_gen"],
                    "exh_impostor": r["exh_n_imp"],
                    "exh_far_percent": round(r["exh_far_percent"], 4),
                    "exh_frr_percent": round(r["exh_frr_percent"], 4),
                    "exh_balanced_accuracy": round(r["exh_balanced_accuracy"], 4),
                })

            # Save drift traces for seed 42
            if seed == 42:
                all_traces = [r["drift_history"] for r in gallery_results.values()]
                max_len = max(len(tr) for tr in all_traces)
                padded = np.zeros((len(all_traces), max_len))
                for i, tr in enumerate(all_traces):
                    padded[i, :len(tr)] = tr
                    if len(tr) < max_len:
                        padded[i, len(tr):] = tr[-1]
                drift_traces[strat_name] = np.mean(padded, axis=0)

    # -----------------------------------------------------------------------
    # Step 5: Save Deliverables, Aggregates & Figures
    # -----------------------------------------------------------------------
    print("\n[Step 5/5] Generating Summaries, Visualizations and Reports...")
    df_summary = pd.DataFrame(all_summary_rows)
    df_summary.to_csv(os.path.join(output_dir, "stream_validation_summary.csv"), index=False)

    df_user = pd.DataFrame(per_user_rows)
    df_user.to_csv(os.path.join(output_dir, "per_participant_validation.csv"), index=False)

    # Compute Aggregate Stats (Mean +/- Std across seeds)
    agg_dict = {}
    for strat in ["Fixed Baseline", "Dual-Gallery Standard", "Dual-Gallery Conservative"]:
        sub = df_summary[df_summary["strategy"] == strat]
        agg_dict[strat] = {
            # Operational Stream
            "stream_micro_bal_mean": float(sub["stream_micro_balanced_accuracy"].mean()),
            "stream_micro_bal_std": float(sub["stream_micro_balanced_accuracy"].std()),
            "stream_micro_far_mean": float(sub["stream_micro_far_percent"].mean()),
            "stream_micro_far_std": float(sub["stream_micro_far_percent"].std()),
            "stream_micro_frr_mean": float(sub["stream_micro_frr_percent"].mean()),
            "stream_micro_frr_std": float(sub["stream_micro_frr_percent"].std()),
            "stream_macro_bal_mean": float(sub["stream_macro_balanced_accuracy"].mean()),
            "stream_macro_bal_std": float(sub["stream_macro_balanced_accuracy"].std()),
            "stream_macro_far_mean": float(sub["stream_macro_far_percent"].mean()),
            "stream_macro_far_std": float(sub["stream_macro_far_percent"].std()),
            "stream_macro_frr_mean": float(sub["stream_macro_frr_percent"].mean()),
            "stream_macro_frr_std": float(sub["stream_macro_frr_percent"].std()),
            "total_updates_mean": float(sub["total_updates"].mean()),
            "total_updates_std": float(sub["total_updates"].std()),
            "genuine_updates_mean": float(sub["genuine_updates"].mean()),
            "genuine_updates_std": float(sub["genuine_updates"].std()),
            "poisoned_updates_mean": float(sub["poisoned_updates"].mean()),
            "poisoned_updates_std": float(sub["poisoned_updates"].std()),
            "poisoning_rate_mean": float(sub["poisoning_rate_percent"].mean()),
            "mean_drift_mean": float(sub["mean_drift"].mean()),
            "max_drift_mean": float(sub["max_drift"].mean()),
            # Exhaustive Verification
            "exh_micro_bal_mean": float(sub["exh_micro_balanced_accuracy"].mean()),
            "exh_micro_bal_std": float(sub["exh_micro_balanced_accuracy"].std()),
            "exh_micro_far_mean": float(sub["exh_micro_far_percent"].mean()),
            "exh_micro_far_std": float(sub["exh_micro_far_percent"].std()),
            "exh_micro_frr_mean": float(sub["exh_micro_frr_percent"].mean()),
            "exh_micro_frr_std": float(sub["exh_micro_frr_percent"].std()),
            "exh_macro_bal_mean": float(sub["exh_macro_balanced_accuracy"].mean()),
            "exh_macro_bal_std": float(sub["exh_macro_balanced_accuracy"].std()),
            "exh_macro_far_mean": float(sub["exh_macro_far_percent"].mean()),
            "exh_macro_far_std": float(sub["exh_macro_far_percent"].std()),
            "exh_macro_frr_mean": float(sub["exh_macro_frr_percent"].mean()),
            "exh_macro_frr_std": float(sub["exh_macro_frr_percent"].std()),
        }

    with open(os.path.join(output_dir, "stream_validation_metrics.json"), "w") as f:
        json.dump(agg_dict, f, indent=2)

    # Visualization 1: Comparative Grouped Bar Charts
    strat_labels = ["Fixed Baseline", "Dual-Gallery Standard", "Dual-Gallery Conservative"]
    x = np.arange(len(strat_labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Plot A: Operational Stream Metrics
    ax = axes[0]
    far_vals = [agg_dict[s]["stream_micro_far_mean"] for s in strat_labels]
    frr_vals = [agg_dict[s]["stream_micro_frr_mean"] for s in strat_labels]
    bal_vals = [agg_dict[s]["stream_micro_bal_mean"] for s in strat_labels]

    ax.bar(x - width, far_vals, width, label="Stream FAR (%)", color="#e74c3c", alpha=0.9)
    ax.bar(x, frr_vals, width, label="Stream FRR (%)", color="#3498db", alpha=0.9)
    ax.bar(x + width, bal_vals, width, label="Balanced Acc (%)", color="#2ecc71", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Fixed\nBaseline", "Dual-Gallery\nStandard", "Dual-Gallery\nConservative"], fontsize=10)
    ax.set_ylabel("Error Rate / Accuracy (%)", fontsize=11)
    ax.set_title("Operational Stream Performance (1:1 Ratio)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_ylim(0, 100)

    # Plot B: Exhaustive Verification Metrics (347,050 trials)
    ax = axes[1]
    exh_far_vals = [agg_dict[s]["exh_micro_far_mean"] for s in strat_labels]
    exh_frr_vals = [agg_dict[s]["exh_micro_frr_mean"] for s in strat_labels]
    exh_bal_vals = [agg_dict[s]["exh_micro_bal_mean"] for s in strat_labels]

    ax.bar(x - width, exh_far_vals, width, label="Exhaustive FAR (%)", color="#e74c3c", alpha=0.9)
    ax.bar(x, exh_frr_vals, width, label="Exhaustive FRR (%)", color="#3498db", alpha=0.9)
    ax.bar(x + width, exh_bal_vals, width, label="Balanced Acc (%)", color="#2ecc71", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Fixed\nBaseline", "Dual-Gallery\nStandard", "Dual-Gallery\nConservative"], fontsize=10)
    ax.set_ylabel("Error Rate / Accuracy (%)", fontsize=11)
    ax.set_title("Exhaustive Verification Performance (347,050 Trials)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "stream_vs_exhaustive_bars.png"))
    plt.close(fig)

    # Visualization 2: Drift Traces
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    for strat_name, trace in drift_traces.items():
        if "Fixed" in strat_name:
            continue
        ax.plot(trace, label=strat_name, linewidth=2.0)
    ax.set_xlabel("Operational Authentication Step", fontsize=11)
    ax.set_ylabel("Mean Template Drift (DTW distance)", fontsize=11)
    ax.set_title("Template Drift Across Operational Stream (Seed 42)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "adaptation_drift_traces.png"))
    plt.close(fig)

    # Generate Markdown Summary Report
    report_lines = [
        "# Adaptive Stream Validation Report: Separated Operational & Exhaustive Evaluation\n",
        f"**Cohort:** 55 Enrolled Participants (RP Session 1 Enrollment)  ",
        f"**Validation Stream:** Session 2 Validation Partition ONLY (6,310 genuine windows)  ",
        f"**Held-Out Test Partition:** 100% Isolated (6,291 windows untouched)  ",
        f"**Frozen Baseline Operating Threshold:** $\\theta^* = {threshold:.6f}$  ",
        f"**Evaluated Random Seeds:** {seeds}  \n",
        "---\n",
        "## 1. Operational Stream Performance (1:1 Ratio, Multi-Confirmation & Anchor Guards)\n",
        "| Strategy | Stream Bal. Acc. (%) | Stream FAR (%) | Stream FRR (%) | Genuine Updates | Poisoned Updates | Poisoning Rate (%) | Mean Drift |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in strat_labels:
        st = agg_dict[s]
        report_lines.append(
            f"| **{s}** | {st['stream_micro_bal_mean']:.2f} ± {st['stream_micro_bal_std']:.2f} "
            f"| {st['stream_micro_far_mean']:.2f} ± {st['stream_micro_far_std']:.2f} "
            f"| {st['stream_micro_frr_mean']:.2f} ± {st['stream_micro_frr_std']:.2f} "
            f"| {st['genuine_updates_mean']:.1f} ± {st['genuine_updates_std']:.1f} "
            f"| {st['poisoned_updates_mean']:.1f} ± {st['poisoned_updates_std']:.1f} "
            f"| {st['poisoning_rate_mean']:.1f}% "
            f"| {st['mean_drift_mean']:.4f} |"
        )

    report_lines.extend([
        "\n---\n",
        "## 2. Exhaustive Verification Performance (All 347,050 Genuine & Impostor Pairs)\n",
        "| Strategy | Exhaustive Bal. Acc. (%) | Exhaustive FAR (%) | Exhaustive FRR (%) | Macro Bal. Acc. (%) | Macro FAR (%) | Macro FRR (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for s in strat_labels:
        st = agg_dict[s]
        report_lines.append(
            f"| **{s}** | {st['exh_micro_bal_mean']:.2f} ± {st['exh_micro_bal_std']:.2f} "
            f"| {st['exh_micro_far_mean']:.2f} ± {st['exh_micro_far_std']:.2f} "
            f"| {st['exh_micro_frr_mean']:.2f} ± {st['exh_micro_frr_std']:.2f} "
            f"| {st['exh_macro_bal_mean']:.2f} ± {st['exh_macro_bal_std']:.2f} "
            f"| {st['exh_macro_far_mean']:.2f} ± {st['exh_macro_far_std']:.2f} "
            f"| {st['exh_macro_frr_mean']:.2f} ± {st['exh_macro_frr_std']:.2f} |"
        )

    with open(os.path.join(output_dir, "adaptive_validation_report.md"), "w") as f:
        f.write("\n".join(report_lines) + "\n")

    elapsed_all = time.time() - t0_all
    print(f"\n✓ Completed Corrected Stream Validation across all 3 strategies and 3 seeds in {elapsed_all:.1f}s.")
    print(f"✓ All outputs saved to: {output_dir}")
    return df_summary, agg_dict


if __name__ == "__main__":
    run_adaptive_stream_validation()
