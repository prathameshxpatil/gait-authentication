"""
evaluate_factors.py — Test individual factors across all 10 participants
========================================================================
Audits:
1. G-only vs R-only vs Combined G+R
2. Raw XYZ vs Dynamic Acc (zero-mean) vs Z-score normalized vs Acceleration Magnitude
3. Window lengths: 2s (200), 3s (300), 4s (400)
4. Sakoe-Chiba radius: 20, 30, 50, full
5. Template count: 4, 8, 16
"""

import sys, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_all_recordings
from src.segmentation import extract_all_windows, partition_enrolment_and_test
from src.dtw_authentication import (
    multivariate_dtw_distance,
    build_enrolment_galleries,
    compute_verification_score,
)
from src.evaluation import compute_far_frr_curve

segments = preprocess_all_recordings(save_files=False)

def run_experiment(
    window_sec=2,
    stream_filter="all",  # 'all', 'G', 'R'
    feat_mode="raw",      # 'raw', 'zero_mean', 'znorm', 'mag', 'mag_znorm'
    n_templates=8,
    radius=30,
    n_test_per_user=30,
    seed=42
):
    rng = np.random.default_rng(seed)
    w_size = int(window_sec * 100)
    s_size = int(w_size / 2)

    windows = extract_all_windows(segments, window_size=w_size, step_size=s_size)

    # Filter streams if requested
    if stream_filter in ["G", "R"]:
        windows = [w for w in windows if w.label == stream_filter]

    # Apply feature transform to windows
    transformed_windows = []
    for w in windows:
        if feat_mode == "raw":
            data = w.data
        elif feat_mode == "zero_mean":
            data = w.data - w.data.mean(axis=0)
        elif feat_mode == "znorm":
            std = w.data.std(axis=0)
            std[std == 0] = 1.0
            data = (w.data - w.data.mean(axis=0)) / std
        elif feat_mode == "mag":
            data = w.magnitude[:, None]
        elif feat_mode == "mag_znorm":
            std = w.magnitude.std()
            if std == 0: std = 1.0
            data = ((w.magnitude - w.magnitude.mean()) / std)[:, None]
        else:
            raise ValueError(f"Unknown feat_mode: {feat_mode}")

        w_copy = type(w)(
            window_id=w.window_id,
            participant_id=w.participant_id,
            session=w.session,
            label=w.label,
            segment_id=w.segment_id,
            window_index=w.window_index,
            start_time_rel=w.start_time_rel,
            end_time_rel=w.end_time_rel,
            data=data,
            magnitude=w.magnitude
        )
        transformed_windows.append(w_copy)

    enrol, test = partition_enrolment_and_test(transformed_windows)
    pids = sorted(enrol.keys())

    # Build galleries
    galleries = build_enrolment_galleries(enrol, n_templates_per_user=n_templates)

    gen_scores = []
    imp_scores = []

    for claimant_id in pids:
        test_wins = test.get(claimant_id, [])
        if len(test_wins) == 0:
            continue
        if len(test_wins) > n_test_per_user:
            idx = rng.choice(len(test_wins), size=n_test_per_user, replace=False)
            sampled = [test_wins[i] for i in idx]
        else:
            sampled = test_wins

        # Genuine
        gal_own = galleries[claimant_id]
        for w in sampled:
            score = compute_verification_score(w, gal_own, window_radius=radius)
            gen_scores.append(score)

        # Impostor
        for claimed_id in pids:
            if claimed_id == claimant_id:
                continue
            gal_other = galleries[claimed_id]
            imp_sample = sampled[:min(10, len(sampled))]
            for w in imp_sample:
                score = compute_verification_score(w, gal_other, window_radius=radius)
                imp_scores.append(score)

    eval_res = compute_far_frr_curve(gen_scores, imp_scores, n_thresholds=500)
    eer = eval_res["eer"]
    eer_far = eval_res["eer_far"]
    eer_frr = eval_res["eer_frr"]
    acc = eval_res["eer_accuracy"]
    balanced_acc = 1.0 - eer
    gen_m = np.mean(gen_scores)
    imp_m = np.mean(imp_scores)

    return {
        "window_sec": window_sec,
        "stream": stream_filter,
        "feature": feat_mode,
        "n_templates": n_templates,
        "radius": radius,
        "n_gen": len(gen_scores),
        "n_imp": len(imp_scores),
        "gen_mean": gen_m,
        "imp_mean": imp_m,
        "margin": imp_m - gen_m,
        "eer": eer,
        "far_at_eer": eer_far,
        "frr_at_eer": eer_frr,
        "acc_at_eer": acc,
        "balanced_acc": balanced_acc,
    }

print("Running comparative factor experiments...")
experiments = [
    # Baseline
    {"window_sec": 2, "stream_filter": "all", "feat_mode": "raw", "n_templates": 8, "radius": 30},
    # Radii comparison
    {"window_sec": 2, "stream_filter": "all", "feat_mode": "raw", "n_templates": 8, "radius": 15},
    {"window_sec": 2, "stream_filter": "all", "feat_mode": "raw", "n_templates": 8, "radius": 50},
    {"window_sec": 2, "stream_filter": "all", "feat_mode": "raw", "n_templates": 8, "radius": -1},
    # Features comparison
    {"window_sec": 2, "stream_filter": "all", "feat_mode": "zero_mean", "n_templates": 8, "radius": 30},
    {"window_sec": 2, "stream_filter": "all", "feat_mode": "znorm", "n_templates": 8, "radius": 30},
    {"window_sec": 2, "stream_filter": "all", "feat_mode": "mag", "n_templates": 8, "radius": 30},
    {"window_sec": 2, "stream_filter": "all", "feat_mode": "mag_znorm", "n_templates": 8, "radius": 30},
    # Stream comparison
    {"window_sec": 2, "stream_filter": "G", "feat_mode": "raw", "n_templates": 8, "radius": 30},
    {"window_sec": 2, "stream_filter": "R", "feat_mode": "raw", "n_templates": 8, "radius": 30},
    {"window_sec": 2, "stream_filter": "G", "feat_mode": "mag", "n_templates": 8, "radius": 30},
    {"window_sec": 2, "stream_filter": "G", "feat_mode": "znorm", "n_templates": 8, "radius": 30},
    # Window length comparison
    {"window_sec": 3, "stream_filter": "all", "feat_mode": "raw", "n_templates": 8, "radius": 30},
    {"window_sec": 4, "stream_filter": "all", "feat_mode": "raw", "n_templates": 8, "radius": 30},
    {"window_sec": 3, "stream_filter": "G", "feat_mode": "raw", "n_templates": 8, "radius": 30},
    {"window_sec": 4, "stream_filter": "G", "feat_mode": "raw", "n_templates": 8, "radius": 30},
    # Template count comparison
    {"window_sec": 2, "stream_filter": "G", "feat_mode": "raw", "n_templates": 4, "radius": 30},
    {"window_sec": 2, "stream_filter": "G", "feat_mode": "raw", "n_templates": 16, "radius": 30},
]

results = []
for exp in experiments:
    res = run_experiment(**exp)
    results.append(res)
    print(f"[{res['stream']:3s}|{res['feature']:9s}|{res['window_sec']}s|R={res['radius']:2d}|T={res['n_templates']:2d}] -> EER={res['eer']*100:.2f}%, BalAcc={res['balanced_acc']*100:.2f}%, Margin={res['margin']:+.4f} (Gen={res['gen_mean']:.3f}, Imp={res['imp_mean']:.3f})")

df_res = pd.DataFrame(results)
print("\nTop 5 configurations by Balanced Accuracy (lowest EER):")
print(df_res.sort_values("eer").head(5)[["stream", "feature", "window_sec", "radius", "n_templates", "eer", "balanced_acc", "margin"]])
