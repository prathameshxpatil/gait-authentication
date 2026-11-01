"""
prototype_adaptation.py — Explore Adaptive Template Dynamics
"""

import sys, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_all_recordings
from src.segmentation import extract_all_windows
from src.experiments import transform_window_features, partition_enrol_val_test
from src.dtw_authentication import build_enrolment_galleries, compute_verification_score, multivariate_dtw_distance
from src.evaluation import compute_far_frr_curve

segments = preprocess_all_recordings(save_files=False)
windows = extract_all_windows(segments, window_size=200, step_size=100)
windows = transform_window_features(windows, feature_mode="magnitude")

enrol, val, test = partition_enrol_val_test(windows, val_ratio=0.5)
galleries = build_enrolment_galleries(enrol, n_templates_per_user=8, strategy="stratified")

# Tune threshold on validation set
val_gen = []
val_imp = []
for p in sorted(galleries.keys()):
    for w in val[p][:30]:
        val_gen.append(compute_verification_score(w, galleries[p], window_radius=30))
    for other_p in sorted(galleries.keys()):
        if other_p == p: continue
        for w in val[other_p][:5]:
            val_imp.append(compute_verification_score(w, galleries[p], window_radius=30))

val_res = compute_far_frr_curve(val_gen, val_imp, n_thresholds=500)
theta_star = val_res["eer_threshold"]
print(f"Validation Threshold θ*: {theta_star:.4f}")

# Test adaptation on User 1
# Gallery for User 1:
initial_templates = list(galleries[1].templates) # 8 templates
fixed_templates = list(initial_templates)
adaptive_templates = list(initial_templates) # 4 permanent anchors + 4 dynamic slots

n_permanent = 4
permanent_anchors = initial_templates[:n_permanent]
dynamic_slots = list(initial_templates[n_permanent:])

print(f"User 1: {len(permanent_anchors)} permanent anchors, {len(dynamic_slots)} dynamic slots")

# Run chronological stream on User 1's test windows
test_wins = test[1][:100]

fixed_scores = []
adapt_scores = []
updates = 0

for i, w in enumerate(test_wins):
    # Fixed score:
    d_fixed = min(multivariate_dtw_distance(w.data, t, window_radius=30) for t in fixed_templates)
    fixed_scores.append(d_fixed)
    
    # Adaptive score:
    current_gallery = permanent_anchors + dynamic_slots
    d_adapt = min(multivariate_dtw_distance(w.data, t, window_radius=30) for t in current_gallery)
    adapt_scores.append(d_adapt)
    
    # Check update condition:
    # High confidence genuine accept: score <= 0.85 * theta_star
    # Anchor drift guard: distance to permanent anchors <= 1.25 * theta_star
    min_anchor_dist = min(multivariate_dtw_distance(w.data, a, window_radius=30) for a in permanent_anchors)
    if d_adapt <= 0.85 * theta_star and min_anchor_dist <= 1.25 * theta_star:
        # Update oldest dynamic slot (FIFO)
        dynamic_slots.pop(0)
        dynamic_slots.append(w.data)
        updates += 1

print(f"User 1 Chronological Test: {len(test_wins)} windows")
print(f"  Fixed scores mean: {np.mean(fixed_scores):.4f}")
print(f"  Adapt scores mean: {np.mean(adapt_scores):.4f}")
print(f"  Updates triggered: {updates}")
print(f"  First 20 Fixed vs Adapt diff: {np.mean(fixed_scores[:20]) - np.mean(adapt_scores[:20]):+.4f}")
print(f"  Last 20 Fixed vs Adapt diff:  {np.mean(fixed_scores[-20:]) - np.mean(adapt_scores[-20:]):+.4f}")

