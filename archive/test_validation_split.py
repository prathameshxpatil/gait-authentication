"""
test_validation_split.py — Verify the Validation / Held-out Test protocol
"""

import sys, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_all_recordings
from src.segmentation import extract_all_windows, partition_enrolment_and_test
from src.dtw_authentication import build_enrolment_galleries, compute_verification_score
from src.evaluation import compute_far_frr_curve

segments = preprocess_all_recordings(save_files=False)
windows = extract_all_windows(segments, window_size=200, step_size=100)
enrol, test = partition_enrolment_and_test(windows)

# Now, split Session 2 test windows for each user into 50% Validation, 50% Test
val_set = {}
test_heldout = {}

for pid, win_list in test.items():
    n = len(win_list)
    mid = n // 2
    val_set[pid] = win_list[:mid]        # First half of Session 2
    test_heldout[pid] = win_list[mid:]   # Second half of Session 2

print("Validation and Test partition sizes:")
for pid in sorted(val_set.keys()):
    print(f"  Sub{pid:2d}: Enrol(S1)={len(enrol[pid]):3d}, Val(S2_1st)={len(val_set[pid]):3d}, Test(S2_2nd)={len(test_heldout[pid]):3d}")

