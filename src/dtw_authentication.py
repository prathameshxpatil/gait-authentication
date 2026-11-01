"""
dtw_authentication.py — Multivariate Dynamic Time Warping Biometric Authentication
===================================================================================
Project : Gait-Based Biometric Authentication (Phase 2)
Dataset : 200GaitData

This module implements:
  1. Multivariate DTW distance between 3D accelerometer sequences (X, Y, Z).
  2. Numba-accelerated dynamic programming with Sakoe-Chiba band constraint
     and pure NumPy fallback.
  3. Enrolment gallery construction strictly from Session 1 (zero data leakage).
  4. Authentication verification for Session 2 claims (Genuine vs Impostor).
  5. Distance-based decision rule with thresholding.

Design principles:
  - Multivariate Euclidean distance across all 3 spatial axes:
      d(p_i, q_j) = sqrt((x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2)
  - Normalized DTW score: DTW(P, Q) / (N + M) for scale-invariant distance.
  - Zero test-to-train leakage: Templates are built exclusively from Session 1.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.segmentation import GaitWindow

# Try importing numba for C-speed JIT execution
try:
    import numba
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# 1. Fast Numba / NumPy Core DTW Kernels
# ---------------------------------------------------------------------------

if HAVE_NUMBA:
    @numba.njit(fastmath=True)
    def _numba_multivariate_dtw(
        seq1: np.ndarray,
        seq2: np.ndarray,
        window_radius: int = -1
    ) -> float:
        """
        Multivariate DTW dynamic programming implemented in Numba.

        Parameters
        ----------
        seq1 : np.ndarray, shape (N, D)
        seq2 : np.ndarray, shape (M, D)
        window_radius : int, Sakoe-Chiba band half-width (-1 for full matrix).

        Returns
        -------
        float
            Accumulated minimal warp path cost.
        """
        n = seq1.shape[0]
        m = seq2.shape[0]
        d = seq1.shape[1]

        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dp[0, 0] = 0.0

        use_band = window_radius > 0

        for i in range(1, n + 1):
            if use_band:
                j_start = max(1, i - window_radius)
                j_end = min(m + 1, i + window_radius + 1)
            else:
                j_start = 1
                j_end = m + 1

            for j in range(j_start, j_end):
                dist_sq = 0.0
                for k in range(d):
                    diff = seq1[i - 1, k] - seq2[j - 1, k]
                    dist_sq += diff * diff
                local_dist = math.sqrt(dist_sq)

                min_prev = dp[i - 1, j]
                if dp[i, j - 1] < min_prev:
                    min_prev = dp[i, j - 1]
                if dp[i - 1, j - 1] < min_prev:
                    min_prev = dp[i - 1, j - 1]

                dp[i, j] = local_dist + min_prev

        return dp[n, m]

    @numba.njit(fastmath=True)
    def _numba_dtw_with_path_length(
        seq1: np.ndarray,
        seq2: np.ndarray,
        window_radius: int = -1
    ):
        """Multivariate DTW returning both minimal path cost and exact warping path length K."""
        n = seq1.shape[0]
        m = seq2.shape[0]
        d = seq1.shape[1]

        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dp[0, 0] = 0.0
        trace = np.zeros((n + 1, m + 1), dtype=np.int32)

        use_band = window_radius > 0

        for i in range(1, n + 1):
            if use_band:
                j_start = max(1, i - window_radius)
                j_end = min(m + 1, i + window_radius + 1)
            else:
                j_start = 1
                j_end = m + 1

            for j in range(j_start, j_end):
                dist_sq = 0.0
                for k in range(d):
                    diff = seq1[i - 1, k] - seq2[j - 1, k]
                    dist_sq += diff * diff
                local_dist = math.sqrt(dist_sq)

                m_diag = dp[i - 1, j - 1]
                m_up = dp[i - 1, j]
                m_left = dp[i, j - 1]

                min_prev = m_diag
                direction = 1
                if m_up < min_prev:
                    min_prev = m_up
                    direction = 2
                if m_left < min_prev:
                    min_prev = m_left
                    direction = 3

                dp[i, j] = local_dist + min_prev
                trace[i, j] = direction

        i_curr, j_curr = n, m
        path_len = 0
        while i_curr > 0 and j_curr > 0:
            path_len += 1
            step = trace[i_curr, j_curr]
            if step == 1:
                i_curr -= 1
                j_curr -= 1
            elif step == 2:
                i_curr -= 1
            elif step == 3:
                j_curr -= 1
            else:
                break
        while i_curr > 0:
            path_len += 1
            i_curr -= 1
        while j_curr > 0:
            path_len += 1
            j_curr -= 1

        return dp[n, m], path_len

else:
    def _numba_multivariate_dtw(
        seq1: np.ndarray,
        seq2: np.ndarray,
        window_radius: int = -1
    ) -> float:
        """Fallback NumPy DTW implementation when Numba is not installed."""
        return _numpy_multivariate_dtw(seq1, seq2, window_radius)

    def _numba_dtw_with_path_length(
        seq1: np.ndarray,
        seq2: np.ndarray,
        window_radius: int = -1
    ):
        """Fallback path-length calculation."""
        cost = _numpy_multivariate_dtw(seq1, seq2, window_radius)
        return cost, seq1.shape[0] + seq2.shape[0]


def _numpy_multivariate_dtw(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window_radius: int = -1
) -> float:
    """Pure NumPy multivariate DTW with optional Sakoe-Chiba constraint."""
    n = seq1.shape[0]
    m = seq2.shape[0]

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    use_band = window_radius > 0

    for i in range(1, n + 1):
        if use_band:
            j_start = max(1, i - window_radius)
            j_end = min(m + 1, i + window_radius + 1)
        else:
            j_start = 1
            j_end = m + 1

        p = seq1[i - 1]
        for j in range(j_start, j_end):
            q = seq2[j - 1]
            local_dist = float(np.linalg.norm(p - q))
            min_prev = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            dp[i, j] = local_dist + min_prev

    return float(dp[n, m])


def multivariate_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window_radius: int = -1,
    normalize: bool = True,
    norm_mode: str = "sum_lengths"
) -> float:
    """
    Calculate the multivariate DTW distance between two 3D gait sequences.

    Parameters
    ----------
    seq1 : np.ndarray, shape (N, D)
        First gait window.
    seq2 : np.ndarray, shape (M, D)
        Second gait window.
    window_radius : int
        Sakoe-Chiba band half-width. If -1, full warping is permitted.
        Default: 30 samples (~0.3s temporal tolerance at 100 Hz).
    normalize : bool
        If True, normalize accumulated path cost.
    norm_mode : str
        'sum_lengths' (divide by N + M) or 'path_length' (divide by exact warping path length K).

    Returns
    -------
    float
        Normalized multivariate DTW distance.
    """
    seq1_c = np.ascontiguousarray(seq1, dtype=np.float64)
    seq2_c = np.ascontiguousarray(seq2, dtype=np.float64)

    # If 1D array passed, reshape to (N, 1)
    if seq1_c.ndim == 1:
        seq1_c = seq1_c[:, None]
    if seq2_c.ndim == 1:
        seq2_c = seq2_c[:, None]

    if norm_mode == "path_length" and normalize and HAVE_NUMBA:
        total_cost, path_len = _numba_dtw_with_path_length(seq1_c, seq2_c, window_radius=window_radius)
        return total_cost / path_len if path_len > 0 else total_cost

    total_cost = _numba_multivariate_dtw(seq1_c, seq2_c, window_radius=window_radius)

    if normalize:
        norm_factor = seq1_c.shape[0] + seq2_c.shape[0]
        return total_cost / norm_factor if norm_factor > 0 else total_cost
    return total_cost


# ---------------------------------------------------------------------------
# 2. Enrolment Gallery & Template Selection
# ---------------------------------------------------------------------------

@dataclass
class ParticipantGallery:
    """
    Enrolment profile for a registered user built strictly from Session 1.

    Attributes
    ----------
    participant_id : int
        User identifier.
    templates : list[np.ndarray]
        List of representative enrolment templates (each shape (200, 3)).
    template_window_ids : list[str]
        IDs of the source windows used as templates.
    """
    participant_id: int
    templates: list[np.ndarray]
    template_window_ids: list[str]


def select_representative_templates(
    windows: list[GaitWindow],
    n_templates: int = 8,
    selection_strategy: str = "stratified",
    radius: int = 30
) -> list[GaitWindow]:
    """
    Select representative enrolment templates from a participant's Session 1 windows.

    Strategies:
      - 'stratified' (default): Evenly samples windows across the walking session,
        capturing cadence variations across the beginning, middle, and end of the walk.
      - 'medoids': Computes pairwise DTW across a candidate subset and picks the medoids
        (windows that minimize distance to all other windows in Session 1).

    Parameters
    ----------
    windows : list[GaitWindow]
        Session 1 windows for one participant.
    n_templates : int
        Number of templates to select (default: 8).
    selection_strategy : str
        'stratified' or 'medoids'.
    radius : int
        DTW radius for medoids computation.

    Returns
    -------
    list[GaitWindow]
        Selected representative template windows.
    """
    if len(windows) <= n_templates:
        return windows

    if selection_strategy == "stratified":
        # Evenly spaced indices across the entire walking session
        indices = np.linspace(0, len(windows) - 1, n_templates, dtype=int)
        return [windows[i] for i in indices]

    elif selection_strategy == "medoids":
        # Candidate pool: take 40 candidate windows evenly spaced
        pool_size = min(40, len(windows))
        candidate_indices = np.linspace(0, len(windows) - 1, pool_size, dtype=int)
        candidates = [windows[i] for i in candidate_indices]

        # Compute pairwise distance matrix
        dists = np.zeros((pool_size, pool_size))
        for i in range(pool_size):
            for j in range(i + 1, pool_size):
                d = multivariate_dtw_distance(candidates[i].data, candidates[j].data, window_radius=radius)
                dists[i, j] = d
                dists[j, i] = d

        # Sort candidates by minimum sum of distances (centrality)
        centrality = dists.sum(axis=1)
        best_indices = np.argsort(centrality)[:n_templates]
        return [candidates[i] for i in best_indices]

    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")


def build_enrolment_galleries(
    enrolment_windows: dict[int, list[GaitWindow]],
    n_templates_per_user: int = 8,
    strategy: str = "stratified"
) -> dict[int, ParticipantGallery]:
    """
    Construct enrolment galleries for all participants using Session 1 data.

    Parameters
    ----------
    enrolment_windows : dict[int, list[GaitWindow]]
        Mapping: participant_id -> list of Session 1 windows.
    n_templates_per_user : int
        Number of templates per user (default: 8).
    strategy : str
        Template selection strategy ('stratified' or 'medoids').

    Returns
    -------
    dict[int, ParticipantGallery]
        Enrolled user profiles ready for authentication verification.
    """
    galleries = {}
    for pid, win_list in sorted(enrolment_windows.items()):
        selected = select_representative_templates(
            win_list, n_templates=n_templates_per_user, selection_strategy=strategy
        )
        templates = [w.data for w in selected]
        template_ids = [w.window_id for w in selected]

        galleries[pid] = ParticipantGallery(
            participant_id=pid,
            templates=templates,
            template_window_ids=template_ids,
        )

    return galleries


# ---------------------------------------------------------------------------
# 3. Authentication Verification & Matching Score
# ---------------------------------------------------------------------------

def compute_verification_score(
    test_window: GaitWindow,
    gallery: ParticipantGallery,
    window_radius: int = 30,
    norm_mode: str = "sum_lengths"
) -> float:
    """
    Calculate the authentication score for a test window claiming an identity.

    The matching score is the minimum normalized multivariate DTW distance
    to any of the participant's enrolled templates:
        score = min_{T in Templates} DTW_norm(test_window, T)

    A lower score indicates higher similarity (better match).

    Parameters
    ----------
    test_window : GaitWindow
        Claimant window from Session 2.
    gallery : ParticipantGallery
        Enrolment gallery of the claimed participant.
    window_radius : int
        Sakoe-Chiba constraint radius.
    norm_mode : str
        'sum_lengths' (default) or 'path_length'.

    Returns
    -------
    float
        Matching distance score (lower = more similar).
    """
    test_data = test_window.data
    min_dist = float("inf")

    for tmpl in gallery.templates:
        d = multivariate_dtw_distance(
            test_data, tmpl, window_radius=window_radius, norm_mode=norm_mode
        )
        if d < min_dist:
            min_dist = d

    return min_dist


@dataclass
class AuthenticationAttempt:
    """Record of an authentication claim and outcome."""
    claimant_participant_id: int
    claimed_participant_id: int
    is_genuine: bool
    test_window_id: str
    matching_score: float
    session: int
    decision: Optional[bool] = None  # True = Accept, False = Reject


def run_authentication_evaluation(
    galleries: dict[int, ParticipantGallery],
    test_windows: dict[int, list[GaitWindow]],
    window_radius: int = 30,
    max_test_windows_per_user: Optional[int] = 50,
    random_seed: int = 42
) -> tuple[list[AuthenticationAttempt], list[float], list[float]]:
    """
    Conduct complete zero-leakage authentication verification:
      - Genuine attempts: Session 2 test windows claiming their own identity.
      - Impostor attempts: Session 2 test windows claiming every other enrolled identity.

    Parameters
    ----------
    galleries : dict[int, ParticipantGallery]
        Enrolment profiles from Session 1.
    test_windows : dict[int, list[GaitWindow]]
        Test windows from Session 2.
    window_radius : int
        DTW Sakoe-Chiba band radius.
    max_test_windows_per_user : int, optional
        Sample up to N test windows per user to balance speed and statistical power.
        If None, evaluates every available test window.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[list[AuthenticationAttempt], list[float], list[float]]
        (all_attempts, genuine_scores, impostor_scores)
    """
    rng = np.random.default_rng(random_seed)
    participant_ids = sorted(galleries.keys())

    all_attempts: list[AuthenticationAttempt] = []
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    print(
        f"Running DTW Authentication on {len(participant_ids)} enrolled participants..."
    )

    for claimant_id in participant_ids:
        user_test_windows = test_windows.get(claimant_id, [])
        if len(user_test_windows) == 0:
            continue

        # Subsample test windows if requested
        if (
            max_test_windows_per_user is not None
            and len(user_test_windows) > max_test_windows_per_user
        ):
            sub_indices = rng.choice(
                len(user_test_windows), size=max_test_windows_per_user, replace=False
            )
            sampled_test_wins = [user_test_windows[i] for i in sub_indices]
        else:
            sampled_test_wins = user_test_windows

        # 1. Genuine attempts: Claimant claims their OWN identity
        gallery_own = galleries[claimant_id]
        for w in sampled_test_wins:
            score = compute_verification_score(
                w, gallery_own, window_radius=window_radius
            )
            genuine_scores.append(score)
            all_attempts.append(
                AuthenticationAttempt(
                    claimant_participant_id=claimant_id,
                    claimed_participant_id=claimant_id,
                    is_genuine=True,
                    test_window_id=w.window_id,
                    matching_score=score,
                    session=w.session,
                )
            )

        # 2. Impostor attempts: Claimant claims EVERY OTHER participant's identity
        for claimed_id in participant_ids:
            if claimed_id == claimant_id:
                continue

            gallery_other = galleries[claimed_id]
            # Test a subset of windows against the impostor claim
            n_imp_samples = min(15, len(sampled_test_wins))
            imp_sample = sampled_test_wins[:n_imp_samples]

            for w in imp_sample:
                score = compute_verification_score(
                    w, gallery_other, window_radius=window_radius
                )
                impostor_scores.append(score)
                all_attempts.append(
                    AuthenticationAttempt(
                        claimant_participant_id=claimant_id,
                        claimed_participant_id=claimed_id,
                        is_genuine=False,
                        test_window_id=w.window_id,
                        matching_score=score,
                        session=w.session,
                    )
                )

    print(
        f"✓ Completed authentication: {len(genuine_scores)} genuine attempts, "
        f"{len(impostor_scores)} impostor attempts."
    )
    return all_attempts, genuine_scores, impostor_scores


if __name__ == "__main__":
    from src.preprocessing import preprocess_all_recordings
    from src.segmentation import extract_all_windows, partition_enrolment_and_test

    segs = preprocess_all_recordings()
    wins = extract_all_windows(segs)
    enrol, test = partition_enrolment_and_test(wins)
    galleries = build_enrolment_galleries(enrol, n_templates_per_user=6)
    attempts, gen_scores, imp_scores = run_authentication_evaluation(
        galleries, test, max_test_windows_per_user=20
    )
    print(f"Genuine scores:  mean={np.mean(gen_scores):.4f}, std={np.std(gen_scores):.4f}")
    print(f"Impostor scores: mean={np.mean(imp_scores):.4f}, std={np.std(imp_scores):.4f}")

