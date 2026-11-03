"""
test_cohort_metrics.py — Unit Tests for Enrolled-Identity Macro & Micro Biometric Metrics
========================================================================================
Validates Correction 1:
  • FAR is calculated per enrolled identity (gallery), not per attacker identity.
  • Genuine attempts for identity i evaluated against gallery i.
  • Impostor attempts against gallery i are all windows from the other (N-1) users.
  • Macro FAR = Mean(FAR_i), Macro FRR = Mean(FRR_i).
  • Micro FAR and Micro FRR use pooled attempt counts.
  • Exact denominator integrity for uniform and non-uniform user window counts.
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation import compute_cohort_metrics


def test_cohort_metrics_uniform_sizes():
    """Verify macro and micro calculations with known synthetic score distributions across 3 users."""
    threshold = 0.50

    # User 1: 10 genuines (all <= 0.50 -> 0 false rejects -> FRR_1 = 0/10 = 0.0)
    # Impostors against Gallery 1: 20 attempts (5 <= 0.50 -> 5 false accepts -> FAR_1 = 5/20 = 0.25)
    gen_1 = [0.20] * 10
    imp_1 = [0.30] * 5 + [0.80] * 15

    # User 2: 10 genuines (2 > 0.50 -> 2 false rejects -> FRR_2 = 2/10 = 0.20)
    # Impostors against Gallery 2: 20 attempts (2 <= 0.50 -> 2 false accepts -> FAR_2 = 2/20 = 0.10)
    gen_2 = [0.40] * 8 + [0.60] * 2
    imp_2 = [0.45] * 2 + [0.75] * 18

    # User 3: 10 genuines (4 > 0.50 -> 4 false rejects -> FRR_3 = 4/10 = 0.40)
    # Impostors against Gallery 3: 20 attempts (8 <= 0.50 -> 8 false accepts -> FAR_3 = 8/20 = 0.40)
    gen_3 = [0.35] * 6 + [0.70] * 4
    imp_3 = [0.40] * 8 + [0.90] * 12

    gen_scores = {1: gen_1, 2: gen_2, 3: gen_3}
    imp_scores = {1: imp_1, 2: imp_2, 3: imp_3}

    res = compute_cohort_metrics(gen_scores, imp_scores, threshold=threshold)

    # 1. Verify individual identity metrics
    p_map = {row["participant_id"]: row for row in res["per_participant"]}

    assert p_map[1]["n_genuine"] == 10
    assert p_map[1]["n_impostor"] == 20
    assert p_map[1]["false_accepts"] == 5
    assert p_map[1]["false_rejects"] == 0
    assert p_map[1]["far"] == pytest.approx(0.25)
    assert p_map[1]["frr"] == pytest.approx(0.00)
    assert p_map[1]["balanced_accuracy_percent"] == pytest.approx((1.0 - (0.25 + 0.0) / 2) * 100)

    assert p_map[2]["far"] == pytest.approx(0.10)
    assert p_map[2]["frr"] == pytest.approx(0.20)

    assert p_map[3]["far"] == pytest.approx(0.40)
    assert p_map[3]["frr"] == pytest.approx(0.40)

    # 2. Verify Macro Metrics across enrolled identities
    expected_macro_far = (0.25 + 0.10 + 0.40) / 3.0  # 0.2500
    expected_macro_frr = (0.00 + 0.20 + 0.40) / 3.0  # 0.2000
    expected_macro_bal = (1.0 - (expected_macro_far + expected_macro_frr) / 2.0) * 100

    assert res["macro"]["far"] == pytest.approx(expected_macro_far)
    assert res["macro"]["frr"] == pytest.approx(expected_macro_frr)
    assert res["macro"]["balanced_accuracy_percent"] == pytest.approx(expected_macro_bal)

    # 3. Verify Micro Metrics (pooled counts)
    tot_fa = 5 + 2 + 8  # 15
    tot_imp = 20 + 20 + 20  # 60
    tot_fr = 0 + 2 + 4  # 6
    tot_gen = 10 + 10 + 10  # 30

    expected_micro_far = tot_fa / tot_imp  # 15 / 60 = 0.25
    expected_micro_frr = tot_fr / tot_gen  # 6 / 30 = 0.20

    assert res["micro"]["total_false_accepts"] == tot_fa
    assert res["micro"]["total_impostor_attempts"] == tot_imp
    assert res["micro"]["total_false_rejects"] == tot_fr
    assert res["micro"]["total_genuine_attempts"] == tot_gen
    assert res["micro"]["far"] == pytest.approx(expected_micro_far)
    assert res["micro"]["frr"] == pytest.approx(expected_micro_frr)


def test_cohort_metrics_non_uniform_window_counts():
    """Verify denominator integrity when participants contribute different numbers of windows."""
    threshold = 0.50

    # User 1: 10 windows (n_gen = 10). Impostor attempts from Users 2 (20) and 3 (30) = 50
    # Impostors against Gallery 1: 50 attempts (10 <= 0.50 -> FA = 10)
    gen_1 = [0.3] * 9 + [0.7] * 1  # FR = 1
    imp_1 = [0.4] * 10 + [0.8] * 40  # FA = 10

    # User 2: 20 windows (n_gen = 20). Impostor attempts from Users 1 (10) and 3 (30) = 40
    # Impostors against Gallery 2: 40 attempts (4 <= 0.50 -> FA = 4)
    gen_2 = [0.3] * 18 + [0.7] * 2  # FR = 2
    imp_2 = [0.4] * 4 + [0.8] * 36  # FA = 4

    # User 3: 30 windows (n_gen = 30). Impostor attempts from Users 1 (10) and 2 (20) = 30
    # Impostors against Gallery 3: 30 attempts (9 <= 0.50 -> FA = 9)
    gen_3 = [0.3] * 24 + [0.7] * 6  # FR = 6
    imp_3 = [0.4] * 9 + [0.8] * 21  # FA = 9

    gen_scores = {1: gen_1, 2: gen_2, 3: gen_3}
    imp_scores = {1: imp_1, 2: imp_2, 3: imp_3}

    res = compute_cohort_metrics(gen_scores, imp_scores, threshold=threshold)

    p_map = {row["participant_id"]: row for row in res["per_participant"]}

    # Check that individual impostor denominators match exactly (W_total - W_i)
    assert p_map[1]["n_impostor"] == 50
    assert p_map[2]["n_impostor"] == 40
    assert p_map[3]["n_impostor"] == 30

    # FAR per enrolled gallery
    assert p_map[1]["far"] == pytest.approx(10 / 50)  # 0.20
    assert p_map[2]["far"] == pytest.approx(4 / 40)   # 0.10
    assert p_map[3]["far"] == pytest.approx(9 / 30)   # 0.30

    # Macro FAR = (0.20 + 0.10 + 0.30) / 3 = 0.20
    assert res["macro"]["far"] == pytest.approx(0.20)

    # Micro Impostor Denominator = 50 + 40 + 30 = 120 = (3-1) * (10+20+30)
    assert res["micro"]["total_impostor_attempts"] == 120
    assert res["micro"]["total_false_accepts"] == 10 + 4 + 9  # 23
    assert res["micro"]["far"] == pytest.approx(23 / 120)

    # Micro Genuine Denominator = 10 + 20 + 30 = 60
    assert res["micro"]["total_genuine_attempts"] == 60
    assert res["micro"]["total_false_rejects"] == 1 + 2 + 6  # 9
    assert res["micro"]["frr"] == pytest.approx(9 / 60)
