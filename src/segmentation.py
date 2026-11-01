"""
segmentation.py — Gait Sequence Windowing & Preparation
========================================================
Project : Gait-Based Biometric Authentication (Phase 2)
Dataset : 200GaitData

This module divides continuous preprocessed gait signals into fixed-length,
overlapping temporal windows:
  - Window length: 2.0 seconds (~200 samples at 100 Hz).
  - Overlap: 50% (step size = 100 samples ~ 1.0 second).
  - Discards incomplete trailing windows (length < 200).
  - Preserves all metadata: participant_id, session, label (G/R), segment_id.
  - Partitions windows into Enrolment (Session 1) and Test (Session 2) sets
    with zero data leakage.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class GaitWindow:
    """
    Data container for an individual window of tri-axial gait acceleration.

    Attributes
    ----------
    window_id : str
        Globally unique window identifier.
    participant_id : int
        Participant ID (e.g. 1, 2, 3, ...).
    session : int
        Session number (1 = Enrolment, 2 = Test).
    label : str
        'G' (Going) or 'R' (Returning).
    segment_id : int
        Continuous segment index within the stream.
    window_index : int
        Window index within the continuous segment.
    start_time_rel : float
        Relative start timestamp in seconds.
    end_time_rel : float
        Relative end timestamp in seconds.
    data : np.ndarray
        Array of shape (window_size, 3) containing filtered [X, Y, Z] acceleration.
    magnitude : np.ndarray
        Array of shape (window_size,) containing vector acceleration magnitude.
    """
    window_id: str
    participant_id: int
    session: int
    label: str
    segment_id: int
    window_index: int
    start_time_rel: float
    end_time_rel: float
    data: np.ndarray
    magnitude: np.ndarray

    def as_dict(self) -> dict:
        """Return metadata summary without large array fields."""
        return {
            "window_id": self.window_id,
            "participant_id": self.participant_id,
            "session": self.session,
            "label": self.label,
            "segment_id": self.segment_id,
            "window_index": self.window_index,
            "start_time_rel": self.start_time_rel,
            "end_time_rel": self.end_time_rel,
            "n_samples": self.data.shape[0],
            "n_channels": self.data.shape[1],
        }


def extract_windows_from_segment(
    segment_df: pd.DataFrame,
    window_size: int = 200,
    step_size: int = 100,
    axes: tuple[str, ...] = ("X", "Y", "Z")
) -> list[GaitWindow]:
    """
    Extract sliding overlapping windows from a single continuous segment.

    Parameters
    ----------
    segment_df : pd.DataFrame
        Continuous preprocessed walking segment.
    window_size : int
        Number of samples per window (default: 200 samples = 2.0s at 100 Hz).
    step_size : int
        Step size between successive windows (default: 100 samples = 50% overlap).
    axes : tuple[str, ...]
        Accelerometer axes to include (default: ('X', 'Y', 'Z')).

    Returns
    -------
    list[GaitWindow]
        List of complete gait windows. Incomplete windows are discarded.
    """
    n_samples = len(segment_df)
    if n_samples < window_size:
        return []

    # Extract metadata from the segment DataFrame
    participant_id = int(segment_df["participant_id"].iloc[0])
    session = int(segment_df["session"].iloc[0])
    label = str(segment_df["label"].iloc[0])
    segment_id = int(segment_df["segment_id"].iloc[0])
    stream_segment_id = segment_df.get(
        "stream_segment_id",
        pd.Series([f"sub{participant_id}_s{session}_{label}_seg{segment_id}"] * n_samples)
    ).iloc[0]

    # Pre-extract numpy matrices for efficiency
    sensor_matrix = segment_df[list(axes)].to_numpy(dtype=np.float64)
    if "a_mag" in segment_df.columns:
        magnitude_vector = segment_df["a_mag"].to_numpy(dtype=np.float64)
    else:
        magnitude_vector = np.linalg.norm(sensor_matrix, axis=1)

    has_time = "time_rel" in segment_df.columns
    time_vector = segment_df["time_rel"].to_numpy(dtype=np.float64) if has_time else None

    windows = []
    window_idx = 0
    start = 0

    while start + window_size <= n_samples:
        end = start + window_size
        win_data = sensor_matrix[start:end, :]
        win_mag = magnitude_vector[start:end]

        t_start = float(time_vector[start]) if time_vector is not None else float(start * 0.010)
        t_end = float(time_vector[end - 1]) if time_vector is not None else float((end - 1) * 0.010)

        win_id = f"{stream_segment_id}_w{window_idx}"

        gait_win = GaitWindow(
            window_id=win_id,
            participant_id=participant_id,
            session=session,
            label=label,
            segment_id=segment_id,
            window_index=window_idx,
            start_time_rel=t_start,
            end_time_rel=t_end,
            data=win_data,
            magnitude=win_mag,
        )
        windows.append(gait_win)

        window_idx += 1
        start += step_size

    return windows


def extract_all_windows(
    processed_segments: dict[str, pd.DataFrame] | list[pd.DataFrame],
    window_size: int = 200,
    step_size: int = 100,
    axes: tuple[str, ...] = ("X", "Y", "Z")
) -> list[GaitWindow]:
    """
    Extract sliding windows across all processed continuous segments.

    Parameters
    ----------
    processed_segments : dict or list of pd.DataFrame
        Container of preprocessed segments.
    window_size : int
        Window length in samples (200 = 2.0s at 100 Hz).
    step_size : int
        Step between windows (100 = 50% overlap).
    axes : tuple[str, ...]
        Sensor axes.

    Returns
    -------
    list[GaitWindow]
        All extracted windows across the dataset.
    """
    segment_list = (
        list(processed_segments.values())
        if isinstance(processed_segments, dict)
        else processed_segments
    )

    all_windows = []
    for seg in segment_list:
        win_list = extract_windows_from_segment(
            seg, window_size=window_size, step_size=step_size, axes=axes
        )
        all_windows.extend(win_list)

    print(
        f"✓ Extracted {len(all_windows)} gait windows "
        f"({window_size} samples / {window_size/100:.1f}s each, {int((1-step_size/window_size)*100)}% overlap)."
    )
    return all_windows


def partition_enrolment_and_test(
    windows: list[GaitWindow]
) -> tuple[dict[int, list[GaitWindow]], dict[int, list[GaitWindow]]]:
    """
    Split windows into Enrolment (Session 1) and Test (Session 2) sets,
    grouped by participant ID.

    Guarantees strict separation:
      - Enrolment pool: strictly Session 1.
      - Test pool: strictly Session 2.
      - Zero cross-session data leakage.

    Parameters
    ----------
    windows : list[GaitWindow]
        All gait windows.

    Returns
    -------
    tuple[dict[int, list[GaitWindow]], dict[int, list[GaitWindow]]]
        (enrolment_dict, test_dict) where keys are participant IDs (e.g. 1, 2, ...).
    """
    enrolment: dict[int, list[GaitWindow]] = {}
    test: dict[int, list[GaitWindow]] = {}

    for w in windows:
        pid = w.participant_id
        if w.session == 1:
            enrolment.setdefault(pid, []).append(w)
        elif w.session == 2:
            test.setdefault(pid, []).append(w)

    return enrolment, test


def get_windows_dataframe(windows: list[GaitWindow]) -> pd.DataFrame:
    """Return a summary metadata DataFrame for a list of windows."""
    records = [w.as_dict() for w in windows]
    return pd.DataFrame(records)


if __name__ == "__main__":
    from src.preprocessing import preprocess_all_recordings
    segments = preprocess_all_recordings()
    windows = extract_all_windows(segments)
    enrol, test = partition_enrolment_and_test(windows)
    print(f"Participants in Enrolment (Session 1): {sorted(enrol.keys())}")
    print(f"Participants in Test (Session 2):      {sorted(test.keys())}")
    for pid in sorted(enrol.keys()):
        n_enrol = len(enrol.get(pid, []))
        n_test = len(test.get(pid, []))
        print(f"  Sub{pid:2d}: {n_enrol:4d} enrolment windows, {n_test:4d} test windows")

