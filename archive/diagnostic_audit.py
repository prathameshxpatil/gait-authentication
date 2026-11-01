"""
diagnostic_audit.py — Deep Diagnostic Audit of the Baseline DTW Pipeline
========================================================================
This script audits:
  1. Sakoe-Chiba constraint effect (radius: 15, 30, 50, 80, None)
  2. Normalization factor effect (N+M vs path length K vs symmetric weights)
  3. Feature representations:
     - Raw filtered (X, Y, Z)
     - Channel-wise Z-normalized (X, Y, Z) (removes static orientation bias)
     - Dynamic acceleration (gravity vector subtracted)
     - Acceleration Magnitude alone (a_mag)
     - Magnitude + Z-normalized XYZ
  4. Stream evaluation: G-only vs R-only vs Combined G+R
  5. Window lengths: 2s (200), 3s (300), 4s (400)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_all_recordings
from src.segmentation import extract_all_windows, partition_enrolment_and_test
from src.dtw_authentication import multivariate_dtw_distance, _numba_multivariate_dtw

print("Loading preprocessed segments...")
segments = preprocess_all_recordings(save_files=False)
print(f"Loaded {len(segments)} segments.")

# Let's inspect sub1 Session 1 vs Session 2 windows
# to directly measure the effect of orientation tilt and Sakoe-Chiba radius
windows_2s = extract_all_windows(segments, window_size=200, step_size=100)
enrol_2s, test_2s = partition_enrolment_and_test(windows_2s)

sub1_enrol = enrol_2s[1]
sub1_test = test_2s[1]
sub2_test = test_2s[2]

w_enrol = sub1_enrol[10]  # Reference window from sub1 S1
w_gen = sub1_test[10]     # Genuine window from sub1 S2
w_imp = sub2_test[10]     # Impostor window from sub2 S2

print("\n--- Diagnostic 1: Feature Statistics of Windows ---")
print(f"Sub1 S1 Window (Enrol): X mean={w_enrol.data[:,0].mean():.3f}, Y mean={w_enrol.data[:,1].mean():.3f}, Z mean={w_enrol.data[:,2].mean():.3f}")
print(f"Sub1 S2 Window (Gen)  : X mean={w_gen.data[:,0].mean():.3f}, Y mean={w_gen.data[:,1].mean():.3f}, Z mean={w_gen.data[:,2].mean():.3f}")
print(f"Sub2 S2 Window (Imp)  : X mean={w_imp.data[:,0].mean():.3f}, Y mean={w_imp.data[:,1].mean():.3f}, Z mean={w_imp.data[:,2].mean():.3f}")

# Notice the difference in mean between Sub1 S1 and Sub1 S2!
diff_s1_s2 = np.abs(w_enrol.data.mean(axis=0) - w_gen.data.mean(axis=0))
print(f"Sub1 S1 vs S2 DC mean shift (gravity tilt across sessions): {diff_s1_s2}")

print("\n--- Diagnostic 2: Sakoe-Chiba Radius Comparison on Raw XYZ ---")
for r in [10, 20, 30, 50, 80, -1]:
    r_label = "Full (no band)" if r == -1 else f"Radius={r}"
    d_gen = multivariate_dtw_distance(w_enrol.data, w_gen.data, window_radius=r, normalize=True)
    d_imp = multivariate_dtw_distance(w_enrol.data, w_imp.data, window_radius=r, normalize=True)
    margin = d_imp - d_gen
    print(f"  {r_label:16s}: Genuine={d_gen:.4f}, Impostor={d_imp:.4f}, Margin={margin:+.4f}")

print("\n--- Diagnostic 3: Feature Representation Comparison ---")
# 1. Raw XYZ
# 2. Window Z-score normalized XYZ (zero-mean, unit variance per channel)
# 3. Dynamic acceleration (subtract window mean)
# 4. Magnitude ||a|| (shape 200, 1)
# 5. Z-score normalized Magnitude

def z_norm(arr):
    std = arr.std(axis=0)
    std[std == 0] = 1.0
    return (arr - arr.mean(axis=0)) / std

def zero_mean(arr):
    return arr - arr.mean(axis=0)

features = {
    "Raw XYZ": (w_enrol.data, w_gen.data, w_imp.data),
    "Zero-Mean XYZ (Dynamic Acc)": (zero_mean(w_enrol.data), zero_mean(w_gen.data), zero_mean(w_imp.data)),
    "Z-Norm XYZ (Standardized)": (z_norm(w_enrol.data), z_norm(w_gen.data), z_norm(w_imp.data)),
    "Magnitude ||a||": (w_enrol.magnitude[:, None], w_gen.magnitude[:, None], w_imp.magnitude[:, None]),
    "Z-Norm Magnitude": (z_norm(w_enrol.magnitude[:, None]), z_norm(w_gen.magnitude[:, None]), z_norm(w_imp.magnitude[:, None])),
}

for feat_name, (e_sig, g_sig, i_sig) in features.items():
    print(f"\nFeature: {feat_name}")
    for r in [30, 50, -1]:
        r_label = "Full" if r == -1 else f"R={r}"
        dg = multivariate_dtw_distance(e_sig, g_sig, window_radius=r, normalize=True)
        di = multivariate_dtw_distance(e_sig, i_sig, window_radius=r, normalize=True)
        print(f"  {r_label:6s}: Genuine={dg:.4f}, Impostor={di:.4f}, Ratio (Imp/Gen)={di/(dg+1e-9):.2f}")

