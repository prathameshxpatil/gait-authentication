"""
preprocessing.py — Gait Sensor Signal Preprocessing Pipeline
=============================================================
Project : Gait-Based Biometric Authentication (Phase 2)
Dataset : 200GaitData (smartphone-based 3-axis accelerometer recordings)

This module handles:
  1. Stream separation: Isolating 'G' (Going) and 'R' (Returning) walking bouts.
  2. Discontinuity detection: Splitting at genuine pauses (Δt > 0.5s), NEVER interpolating.
  3. Timestamp normalization: Resetting relative elapsed time to t=0 for each segment.
  4. Zero-phase filtering: 4th-order Butterworth low-pass filter (cutoff 20 Hz, fs ~100 Hz).
  5. Preserving 3D spatial axes (X, Y, Z) and computing acceleration magnitude.
  6. Saving processed segments into data/processed/.

Design principles:
  - Never modifies original raw files on disk.
  - Zero-phase filtering prevents phase shift in gait peaks/valleys.
  - Does not blindly clip or delete genuine heel-strike acceleration spikes.
  - Fully modular, documented, and reproducible.
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import signal

from src.data_loader import load_all_recordings


def butter_lowpass_filter(
    data: np.ndarray,
    cutoff: float = 20.0,
    fs: float = 100.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply a zero-phase 4th-order Butterworth low-pass filter along axis 0.

    Walking gait kinematics occupy frequencies between 0.5 Hz and ~15 Hz
    (fundamental stride frequency ~1-2 Hz plus primary harmonics). A 20 Hz
    cutoff removes high-frequency jitter, sensor noise, and device micro-vibrations
    while preserving genuine biomechanical movement and heel-strike dynamics.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D array of sensor values (samples, channels).
    cutoff : float
        Cutoff frequency in Hz (default: 20.0 Hz).
    fs : float
        Sampling frequency in Hz (default: 100.0 Hz).
    order : int
        Filter order (default: 4).

    Returns
    -------
    np.ndarray
        Filtered array with identical shape, zero phase delay.
    """
    # Nyquist frequency is half the sampling frequency
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist

    # Safety clamp: normalized cutoff must be strictly between 0 and 1
    if normalized_cutoff >= 1.0:
        raise ValueError(
            f"Cutoff frequency ({cutoff} Hz) must be strictly below Nyquist ({nyquist} Hz)."
        )

    # Design Butterworth filter
    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)

    # Use forward-backward zero-phase filtering (filtfilt)
    # If segment length is very short, fallback to minimum length required by filtfilt
    padlen = min(3 * max(len(a), len(b)), len(data) - 1)
    if padlen < 1:
        return data.copy()

    filtered = signal.filtfilt(b, a, data, axis=0, padlen=padlen)
    return filtered


def split_at_discontinuities(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    gap_threshold_sec: float = 0.5
) -> list[pd.DataFrame]:
    """
    Split a DataFrame into continuous segments whenever Δt > gap_threshold_sec.

    Crucial rule: We NEVER interpolate across large recording pauses (20-30s),
    as that would inject synthetic, unphysical walking data into biometric templates.

    Parameters
    ----------
    df : pd.DataFrame
        Sensor stream data for a single label.
    timestamp_col : str
        Name of timestamp column.
    gap_threshold_sec : float
        Threshold above which a gap is treated as a pause/discontinuity (default: 0.5s).

    Returns
    -------
    list[pd.DataFrame]
        List of continuous segment DataFrames.
    """
    if len(df) == 0:
        return []

    timestamps = df[timestamp_col].values
    diffs = np.diff(timestamps)

    # Identify split boundaries where Δt exceeds the discontinuity threshold
    gap_indices = np.where(diffs > gap_threshold_sec)[0]

    if len(gap_indices) == 0:
        return [df.copy().reset_index(drop=True)]

    segments = []
    start_idx = 0
    for gap_idx in gap_indices:
        # Segment from start_idx up to and including gap_idx
        seg = df.iloc[start_idx : gap_idx + 1].copy().reset_index(drop=True)
        if len(seg) > 0:
            segments.append(seg)
        start_idx = gap_idx + 1

    # Final trailing segment
    if start_idx < len(df):
        seg = df.iloc[start_idx:].copy().reset_index(drop=True)
        if len(seg) > 0:
            segments.append(seg)

    return segments


def preprocess_stream_segment(
    segment_df: pd.DataFrame,
    participant_id: int,
    session: int,
    label: str,
    segment_id: int,
    cutoff_hz: float = 20.0,
    filter_order: int = 4,
    timestamp_col: str = "Timestamp",
    axes: list[str] = None
) -> pd.DataFrame:
    """
    Preprocess a single continuous segment of walking data:
      1. Reset timestamps to start at t = 0.0 seconds.
      2. Estimate actual segment sampling rate.
      3. Apply zero-phase 4th order Butterworth filter to each axis.
      4. Compute vector magnitude: a_mag = sqrt(X^2 + Y^2 + Z^2).
      5. Annotate metadata columns.

    Parameters
    ----------
    segment_df : pd.DataFrame
        Continuous walking segment.
    participant_id : int
        Participant ID.
    session : int
        Session number (1 or 2).
    label : str
        'G' (Going) or 'R' (Returning).
    segment_id : int
        Sub-segment index (e.g. 0, 1).
    cutoff_hz : float
        Filter cutoff frequency (default: 20 Hz).
    filter_order : int
        Butterworth order (default: 4).
    timestamp_col : str
        Timestamp column name.
    axes : list[str]
        Sensor axes to filter (default: ['X', 'Y', 'Z']).

    Returns
    -------
    pd.DataFrame
        Preprocessed segment with normalized time and filtered axes.
    """
    if axes is None:
        axes = ["X", "Y", "Z"]

    df = segment_df.copy().reset_index(drop=True)

    # 1. Reset timestamps: elapsed relative time in seconds from segment start
    t0 = df[timestamp_col].iloc[0]
    df["time_rel"] = df[timestamp_col] - t0

    # Estimate sampling frequency for this specific segment
    diffs = np.diff(df[timestamp_col].values)
    pos_diffs = diffs[diffs > 0]
    median_dt = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 0.010
    fs_est = 1.0 / median_dt if median_dt > 0 else 100.0

    # 2. Preserve raw signals for auditing and plotting
    for axis in axes:
        df[f"{axis}_raw"] = df[axis].copy()

    # 3. Filter each axis using zero-phase Butterworth filter
    for axis in axes:
        raw_vals = df[axis].values.astype(float)
        # Apply filter only if segment is sufficiently long
        if len(raw_vals) > 3 * filter_order:
            filtered_vals = butter_lowpass_filter(
                raw_vals, cutoff=cutoff_hz, fs=fs_est, order=filter_order
            )
            df[axis] = filtered_vals
        else:
            # Short segment fallback
            df[axis] = raw_vals

    # 4. Compute vector magnitude: a_mag = sqrt(X^2 + Y^2 + Z^2)
    # Useful for orientation-independent gait cadence and cycle detection
    df["a_mag"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
    df["a_mag_raw"] = np.sqrt(df["X_raw"]**2 + df["Y_raw"]**2 + df["Z_raw"]**2)

    # 5. Metadata columns
    df["participant_id"] = participant_id
    df["session"] = session
    df["label"] = label
    df["segment_id"] = segment_id
    df["fs_est"] = round(fs_est, 2)

    return df


def preprocess_recording(
    df_raw: pd.DataFrame,
    recording_name: str,
    gap_threshold_sec: float = 0.5,
    cutoff_hz: float = 20.0,
    filter_order: int = 4
) -> list[pd.DataFrame]:
    """
    Full preprocessing pipeline for an individual raw CSV recording:
      - Extracts participant and session metadata from the name.
      - Separates 'G' and 'R' walking streams.
      - Detects and splits across any gap > gap_threshold_sec.
      - Filters each continuous sub-segment and normalizes timestamps.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw recording DataFrame.
    recording_name : str
        e.g. 'sub1-rp-s1'.
    gap_threshold_sec : float
        Discontinuity threshold in seconds (default: 0.5s).
    cutoff_hz : float
        Butterworth cutoff in Hz (default: 20.0 Hz).
    filter_order : int
        Butterworth order (default: 4).

    Returns
    -------
    list[pd.DataFrame]
        List of preprocessed continuous segment DataFrames.
    """
    # Parse participant ID and session from name
    m = re.search(r"sub(\d+)-rp-s(\d+)", recording_name)
    if m:
        participant_id = int(m.group(1))
        session = int(m.group(2))
    else:
        participant_id = 0
        session = 1

    processed_segments = []

    # Available movement bout labels (typically ['G', 'R'] or only ['G'])
    labels = sorted(df_raw["Label"].unique()) if "Label" in df_raw.columns else ["all"]

    for label_val in labels:
        if "Label" in df_raw.columns:
            stream_df = df_raw[df_raw["Label"] == label_val].copy().reset_index(drop=True)
        else:
            stream_df = df_raw.copy().reset_index(drop=True)

        if len(stream_df) == 0:
            continue

        # Split stream at timestamp discontinuities
        raw_subsegs = split_at_discontinuities(
            stream_df, timestamp_col="Timestamp", gap_threshold_sec=gap_threshold_sec
        )

        for seg_idx, raw_seg in enumerate(raw_subsegs):
            # Only process segments that contain enough samples for meaningful gait analysis
            # 2 seconds at 100 Hz = 200 samples
            if len(raw_seg) < 200:
                continue

            clean_seg = preprocess_stream_segment(
                segment_df=raw_seg,
                participant_id=participant_id,
                session=session,
                label=str(label_val),
                segment_id=seg_idx,
                cutoff_hz=cutoff_hz,
                filter_order=filter_order,
            )
            clean_seg["recording_id"] = recording_name
            clean_seg["stream_segment_id"] = f"{recording_name}_{label_val}_seg{seg_idx}"
            processed_segments.append(clean_seg)

    return processed_segments


def get_processed_data_dir(project_root: str = None) -> str:
    """Return absolute path to data/processed/."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "data", "processed")


def preprocess_all_recordings(
    raw_dir: str = None,
    processed_dir: str = None,
    save_files: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Run the preprocessing pipeline over all raw recordings in the dataset.
    Saves clean segment files to data/processed/.

    Parameters
    ----------
    raw_dir : str, optional
        Path to raw data directory.
    processed_dir : str, optional
        Path to save processed CSVs (default: data/processed/).
    save_files : bool
        Whether to persist processed segments as CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from segment ID (e.g. 'sub1-rp-s1_G_seg0') to preprocessed DataFrame.
    """
    if processed_dir is None:
        processed_dir = get_processed_data_dir()

    if save_files:
        os.makedirs(processed_dir, exist_ok=True)

    recordings = load_all_recordings(raw_dir)
    all_processed_segments = {}

    print(f"Preprocessing {len(recordings)} raw recordings...")

    for name, df_raw in sorted(recordings.items()):
        segments = preprocess_recording(df_raw, recording_name=name)

        for seg in segments:
            seg_key = seg["stream_segment_id"].iloc[0]
            all_processed_segments[seg_key] = seg

            if save_files:
                out_path = os.path.join(processed_dir, f"{seg_key}.csv")
                seg.to_csv(out_path, index=False)

    print(
        f"✓ Preprocessing complete: Extracted {len(all_processed_segments)} "
        f"continuous, filtered gait segments from {len(recordings)} recordings."
    )
    if save_files:
        print(f"  Processed files saved to: {processed_dir}")

    return all_processed_segments


if __name__ == "__main__":
    segments = preprocess_all_recordings()
    print(f"Successfully processed {len(segments)} segments.")

