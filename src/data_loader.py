"""
data_loader.py — Reusable Data Loading Utilities
=================================================
Project : Gait-Based Biometric Authentication (Phase 1)
Dataset : 200GaitData (smartphone-based gait recordings)

This module provides helper functions to:
  • Discover all CSV files in the raw data directory.
  • Parse participant ID and session number from filenames.
  • Load individual files or the entire dataset into Pandas DataFrames.
  • Perform basic data-quality checks (missing values, duplicates, etc.).

Usage
-----
    from src.data_loader import load_all_recordings, get_file_inventory

Design principles:
  - Never modifies the original CSV files on disk.
  - Does not hard-code column names — they are read from the headers.
  - Returns clean Python objects (DataFrames, dicts) for downstream use.
"""

import os
import re
import glob
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. File Discovery
# ---------------------------------------------------------------------------

def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_raw_data_dir(project_root: str = None) -> str:
    """
    Return the absolute path to the raw data directory.

    Parameters
    ----------
    project_root : str, optional
        Root of the project. If None, we assume this file lives at
        <project_root>/src/data_loader.py and walk up one level.

    Returns
    -------
    str
        Absolute path to data/raw/.
    """
    if project_root is None:
        project_root = get_project_root()
    return os.path.join(project_root, "data", "raw")


def get_manifest_path(project_root: str = None) -> str:
    """
    Return the absolute path to the dataset manifest CSV file.

    Parameters
    ----------
    project_root : str, optional
        Root of the project.

    Returns
    -------
    str
        Absolute path to data/dataset_manifest.csv.
    """
    if project_root is None:
        project_root = get_project_root()
    return os.path.join(project_root, "data", "dataset_manifest.csv")


def load_manifest(manifest_path: str = None) -> pd.DataFrame:
    """
    Load the dataset manifest into a Pandas DataFrame.

    Parameters
    ----------
    manifest_path : str, optional
        Path to dataset_manifest.csv. Uses default location if None.

    Returns
    -------
    pd.DataFrame
        Dataset manifest containing recording IDs, canonical IDs, metadata,
        and paths to raw files.
    """
    if manifest_path is None:
        manifest_path = get_manifest_path()

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Dataset manifest not found at: {manifest_path}. "
            "Please ensure data/dataset_manifest.csv exists."
        )

    return pd.read_csv(manifest_path)


def discover_csv_files(raw_dir: str = None) -> list[str]:
    """
    Find every CSV file inside the raw data directory.

    Handles quirks observed in the 200GaitData download:
      • Filenames may start with a leading space.
      • Some files have a double extension like '.csv.csv'.

    Parameters
    ----------
    raw_dir : str, optional
        Path to the raw data folder. Uses default if None.

    Returns
    -------
    list[str]
        Sorted list of absolute paths to all CSV files found.
    """
    if raw_dir is None:
        raw_dir = get_raw_data_dir()

    # Glob for both .csv and .csv.csv files
    patterns = [
        os.path.join(raw_dir, "*.csv"),
        os.path.join(raw_dir, "*.csv.csv"),
    ]

    found = set()
    for pattern in patterns:
        found.update(glob.glob(pattern))

    # Remove any non-file entries (e.g. .DS_Store matched by accident)
    csv_files = sorted([f for f in found if os.path.isfile(f) and f.endswith(".csv")])
    return csv_files


# ---------------------------------------------------------------------------
# 2. Filename Parsing
# ---------------------------------------------------------------------------

def parse_filename(filepath: str) -> dict:
    """
    Extract metadata from a recording filename.

    Expected naming conventions:
        ' sub<ID>-rp-s<SESSION>.csv'        (Right Pocket)
        ' sub<ID>-lw-s<SESSION>.csv'        (Left Wrist)
        ' sub<ID>-<placement>-s<SESSION>.csv.csv' (Double extension)
        ' sub<ID>-<placement>-s<SESSION>-b<BOUT>.csv' (With optional bout)

    Parameters
    ----------
    filepath : str
        Full or relative path to the CSV file.

    Returns
    -------
    dict
        Keys: 'participant_id' (int), 'placement' (str: 'RP' or 'LW'),
              'session' (int), 'bout' (int), 'filename' (str), 'filepath' (str).
        Returns None values for participant_id / placement / session if parsing fails.
    """
    filename = os.path.basename(filepath).strip()  # strip leading spaces

    # Match sub<number>-(rp|lw)-s<number> with optional bout suffix (-b<number>)
    match = re.search(r"sub(\d+)-(rp|lw)-s(\d+)(?:-b(\d+))?", filename, re.IGNORECASE)

    if match:
        participant_id = int(match.group(1))
        placement = match.group(2).upper()
        session = int(match.group(3))
        bout = int(match.group(4)) if match.group(4) else 1
    else:
        participant_id = None
        placement = None
        session = None
        bout = None

    return {
        "participant_id": participant_id,
        "placement": placement,
        "session": session,
        "bout": bout,
        "filename": filename,
        "filepath": filepath,
    }


def get_file_inventory(raw_dir: str = None,
                       manifest_path: str = None,
                       use_manifest: bool = True) -> pd.DataFrame:
    """
    Build a summary table of every CSV file with parsed metadata and file size.
    Prefers the authoritative manifest if available.

    Parameters
    ----------
    raw_dir : str, optional
        Path to the raw data folder (used if use_manifest=False).
    manifest_path : str, optional
        Explicit path to dataset_manifest.csv.
    use_manifest : bool, optional
        If True (default) and manifest exists, loads metadata from manifest.

    Returns
    -------
    pd.DataFrame
        One row per file with metadata columns.
    """
    if use_manifest:
        try:
            mpath = manifest_path or get_manifest_path()
            if os.path.exists(mpath):
                return pd.read_csv(mpath)
        except Exception:
            pass  # Fall back to raw directory scan

    csv_files = discover_csv_files(raw_dir)
    rows = []
    for fp in csv_files:
        info = parse_filename(fp)
        info["file_size_bytes"] = os.path.getsize(fp)
        rows.append(info)

    df = pd.DataFrame(rows)

    # Sort by participant then placement then session for readability
    if not df.empty and df["participant_id"].notna().all():
        df = df.sort_values(["participant_id", "placement", "session"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 3. Data Loading
# ---------------------------------------------------------------------------

def load_single_recording(filepath: str) -> pd.DataFrame:
    """
    Load one CSV recording into a Pandas DataFrame.

    The 200GaitData CSVs have a quoted index column as the first field.
    This function reads it and cleans up the column names.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded data with cleaned column names.
    """
    # Read the CSV — the first column is an unnamed row-index from R export
    df = pd.read_csv(filepath)

    # Clean column names: strip whitespace and quotes
    df.columns = [col.strip().strip('"') for col in df.columns]

    # If the first column is unnamed (blank or 'Unnamed: 0'), rename it
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "OriginalIndex"})

    return df


def load_all_recordings(raw_dir: str = None,
                        placement: str = None,
                        manifest_path: str = None,
                        use_manifest: bool = True) -> dict[str, pd.DataFrame]:
    """
    Load CSV files into a dictionary of DataFrames.

    Parameters
    ----------
    raw_dir : str, optional
        Path to the raw data folder. Used when use_manifest=False or manifest missing.
    placement : str, optional
        Filter recordings by placement: 'RP' (Right Pocket), 'LW' (Left Wrist),
        or None (load all placements).
    manifest_path : str, optional
        Path to dataset_manifest.csv. Uses data/dataset_manifest.csv by default.
    use_manifest : bool, optional
        If True (default), loads files via the manifest which ensures deterministic
        disambiguation of collision files and unique canonical keys.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are canonical IDs (e.g. 'sub1-rp-s1', 'sub1-lw-s1', 'sub19-lw-s2-b1'),
        values are DataFrames with metadata attributes attached (df.attrs).
    """
    recordings = {}
    project_root = get_project_root()
    mpath = manifest_path or get_manifest_path(project_root)

    if use_manifest and os.path.exists(mpath):
        manifest_df = pd.read_csv(mpath)
        if placement is not None:
            manifest_df = manifest_df[
                manifest_df["placement"].str.upper() == placement.upper()
            ].copy()

        for _, row in manifest_df.iterrows():
            raw_path = row["raw_path"]
            if not os.path.isabs(raw_path):
                filepath = os.path.join(project_root, raw_path)
            else:
                filepath = raw_path

            df = load_single_recording(filepath)
            df.attrs["recording_id"] = row["recording_id"]
            df.attrs["canonical_id"] = row["canonical_id"]
            df.attrs["participant_id"] = int(row["participant_id"])
            df.attrs["placement"] = row["placement"]
            df.attrs["session"] = int(row["session_int"])
            df.attrs["bout"] = int(row["bout"])
            df.attrs["sha256"] = row["sha256"]
            df.attrs["raw_path"] = row["raw_path"]

            key = row["canonical_id"]
            recordings[key] = df

        return recordings

    # Fallback to filesystem scan
    csv_files = discover_csv_files(raw_dir)
    seen_keys = {}

    for fp in csv_files:
        info = parse_filename(fp)
        if placement is not None and info["placement"] is not None:
            if info["placement"].upper() != placement.upper():
                continue

        if info["participant_id"] is not None and info["placement"] is not None:
            base_key = f"sub{info['participant_id']}-{info['placement'].lower()}-s{info['session']}"
        else:
            base_key = info["filename"]

        # Disambiguate if multiple files share the base_key
        if base_key not in seen_keys:
            seen_keys[base_key] = 1
            key = base_key
        else:
            seen_keys[base_key] += 1
            key = f"{base_key}-b{seen_keys[base_key]}"

        df = load_single_recording(fp)
        if info["participant_id"] is not None:
            df.attrs["participant_id"] = info["participant_id"]
            df.attrs["placement"] = info["placement"]
            df.attrs["session"] = info["session"]
            df.attrs["bout"] = info.get("bout", 1)
        recordings[key] = df

    return recordings


def load_recording_by_id(rec_id: str,
                         manifest_path: str = None) -> pd.DataFrame:
    """
    Load a single recording by its recording_id (e.g. 'rec_001') or
    canonical_id (e.g. 'sub1-rp-s1', 'sub19-lw-s2-b1').

    Parameters
    ----------
    rec_id : str
        The unique recording_id or canonical_id.
    manifest_path : str, optional
        Path to manifest file.

    Returns
    -------
    pd.DataFrame
        Loaded recording DataFrame with metadata attached in df.attrs.
    """
    project_root = get_project_root()
    mpath = manifest_path or get_manifest_path(project_root)
    manifest_df = pd.read_csv(mpath)

    match = manifest_df[
        (manifest_df["recording_id"] == rec_id) |
        (manifest_df["canonical_id"] == rec_id)
    ]
    if match.empty:
        raise KeyError(f"Recording ID '{rec_id}' not found in dataset manifest.")

    row = match.iloc[0]
    raw_path = row["raw_path"]
    if not os.path.isabs(raw_path):
        filepath = os.path.join(project_root, raw_path)
    else:
        filepath = raw_path

    df = load_single_recording(filepath)
    df.attrs["recording_id"] = row["recording_id"]
    df.attrs["canonical_id"] = row["canonical_id"]
    df.attrs["participant_id"] = int(row["participant_id"])
    df.attrs["placement"] = row["placement"]
    df.attrs["session"] = int(row["session_int"])
    df.attrs["bout"] = int(row["bout"])
    df.attrs["sha256"] = row["sha256"]
    df.attrs["raw_path"] = row["raw_path"]

    return df


def get_participant_recordings(participant_id: int,
                               placement: str = None,
                               manifest_path: str = None) -> dict[str, pd.DataFrame]:
    """
    Retrieve all recordings for a specific participant.

    Parameters
    ----------
    participant_id : int
        Participant ID (1..93).
    placement : str, optional
        Filter by placement ('RP' or 'LW'), or None for all.
    manifest_path : str, optional
        Path to manifest file.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from canonical_id to DataFrame.
    """
    project_root = get_project_root()
    mpath = manifest_path or get_manifest_path(project_root)
    manifest_df = pd.read_csv(mpath)

    subset = manifest_df[manifest_df["participant_id"] == participant_id]
    if placement is not None:
        subset = subset[subset["placement"].str.upper() == placement.upper()]

    recordings = {}
    for _, row in subset.iterrows():
        raw_path = row["raw_path"]
        if not os.path.isabs(raw_path):
            filepath = os.path.join(project_root, raw_path)
        else:
            filepath = raw_path

        df = load_single_recording(filepath)
        df.attrs["recording_id"] = row["recording_id"]
        df.attrs["canonical_id"] = row["canonical_id"]
        df.attrs["participant_id"] = int(row["participant_id"])
        df.attrs["placement"] = row["placement"]
        df.attrs["session"] = int(row["session_int"])
        df.attrs["bout"] = int(row["bout"])
        df.attrs["sha256"] = row["sha256"]
        df.attrs["raw_path"] = row["raw_path"]

        recordings[row["canonical_id"]] = df

    return recordings


# ---------------------------------------------------------------------------
# 4. Data Quality Checks
# ---------------------------------------------------------------------------

def check_data_quality(df: pd.DataFrame, name: str = "recording") -> dict:
    """
    Run basic data quality checks on a single recording DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A loaded recording.
    name : str
        Human-readable label for log messages.

    Returns
    -------
    dict
        Summary with keys: 'name', 'shape', 'columns', 'dtypes',
        'missing_per_column', 'total_missing', 'duplicate_rows',
        'numeric_columns', 'inf_values_per_column'.
    """
    report = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_per_column": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["numeric_columns"] = numeric_cols

    inf_counts = {}
    for col in numeric_cols:
        inf_count = int(np.isinf(df[col]).sum())
        if inf_count > 0:
            inf_counts[col] = inf_count
    report["inf_values_per_column"] = inf_counts

    return report


def estimate_sampling_frequency(df: pd.DataFrame,
                                timestamp_col: str = "Timestamp") -> dict:
    """
    Estimate the sampling frequency from consecutive timestamp differences.

    Parameters
    ----------
    df : pd.DataFrame
        A loaded recording that contains a timestamp column.
    timestamp_col : str
        Name of the timestamp column.

    Returns
    -------
    dict
        Keys: 'mean_dt', 'median_dt', 'std_dt', 'estimated_freq_hz',
              'min_dt', 'max_dt'.
        Returns None values if the timestamp column is missing.
    """
    if timestamp_col not in df.columns:
        return {
            "mean_dt": None,
            "estimated_freq_hz": None,
            "note": f"Column '{timestamp_col}' not found.",
        }

    timestamps = df[timestamp_col].dropna().values
    diffs = np.diff(timestamps)

    # Filter out zero or negative diffs (shouldn't happen, but be safe)
    positive_diffs = diffs[diffs > 0]

    if len(positive_diffs) == 0:
        return {"mean_dt": None, "estimated_freq_hz": None, "note": "No valid diffs."}

    mean_dt = float(np.mean(positive_diffs))
    median_dt = float(np.median(positive_diffs))
    std_dt = float(np.std(positive_diffs))

    # The timestamps appear to be in milliseconds — estimate Hz
    # We'll try both seconds and milliseconds interpretations
    estimated_freq_hz = 1.0 / median_dt if median_dt != 0 else None

    return {
        "mean_dt": mean_dt,
        "median_dt": median_dt,
        "std_dt": std_dt,
        "min_dt": float(np.min(positive_diffs)),
        "max_dt": float(np.max(positive_diffs)),
        "estimated_freq_hz": estimated_freq_hz,
        "note": (
            "Frequency is 1/median_dt. Units depend on the timestamp column — "
            "inspect the raw values to determine if they are seconds, milliseconds, etc."
        ),
    }


# ---------------------------------------------------------------------------
# 5. Stream Separation & Gap Detection (Validation / Preprocessing Helpers)
# ---------------------------------------------------------------------------

def separate_streams(df: pd.DataFrame, label_col: str = "Label") -> dict[str, pd.DataFrame]:
    """
    Split a recording into distinct DataFrames based on the Label column.

    In the 200GaitData dataset:
      - 'G' represents the outward walk ('Going') along the 320m track.
      - 'R' represents the return walk ('Returning') along the 320m track.
    Both streams contain 3-axis accelerometer data recorded at ~100 Hz.

    Parameters
    ----------
    df : pd.DataFrame
        Full recording DataFrame.
    label_col : str
        Column name indicating movement bout / stream (default 'Label').

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from label value (e.g., 'G', 'R') to the corresponding subset,
        with indices reset.
    """
    if label_col not in df.columns:
        return {"all": df.copy()}

    streams = {}
    for label_val in df[label_col].unique():
        sub_df = df[df[label_col] == label_val].copy().reset_index(drop=True)
        streams[str(label_val)] = sub_df

    return streams


def detect_timestamp_gaps(df: pd.DataFrame,
                          timestamp_col: str = "Timestamp",
                          threshold_sec: float = 0.05) -> pd.DataFrame:
    """
    Identify discontinuities and pauses in a time series where Δt > threshold_sec.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a timestamp column.
    timestamp_col : str
        Name of timestamp column.
    threshold_sec : float
        Discontinuity threshold in seconds (default 0.05 s = 50 ms).

    Returns
    -------
    pd.DataFrame
        DataFrame listing all detected gaps with start/end timestamps and gap durations.
    """
    if timestamp_col not in df.columns:
        return pd.DataFrame()

    ts = df[timestamp_col].values
    diffs = np.diff(ts)
    gap_indices = np.where(diffs > threshold_sec)[0]

    records = []
    for idx in gap_indices:
        records.append({
            "sample_index_before": int(idx),
            "timestamp_before": float(ts[idx]),
            "timestamp_after": float(ts[idx + 1]),
            "gap_duration_sec": float(diffs[idx]),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Convenience: Quick Summary
# ---------------------------------------------------------------------------

def print_dataset_summary(raw_dir: str = None) -> None:
    """
    Print a human-readable summary of the entire dataset to the console.
    Useful for quick sanity checks from the command line.

    Usage:
        python -m src.data_loader
    """
    print("=" * 70)
    print("  200GaitData — Dataset Summary")
    print("=" * 70)

    inventory = get_file_inventory(raw_dir)
    print(f"\nFiles found: {len(inventory)}")
    print(inventory.to_string(index=False))

    recordings = load_all_recordings(raw_dir)

    for key, df in recordings.items():
        print(f"\n{'─' * 50}")
        print(f"Recording: {key}")
        print(f"  Shape     : {df.shape}")
        print(f"  Columns   : {list(df.columns)}")

        quality = check_data_quality(df, key)
        print(f"  Missing   : {quality['total_missing']}")
        print(f"  Duplicates: {quality['duplicate_rows']}")
        if quality["inf_values_per_column"]:
            print(f"  Inf values: {quality['inf_values_per_column']}")

        freq_info = estimate_sampling_frequency(df)
        if freq_info["estimated_freq_hz"] is not None:
            print(f"  Est. freq : {freq_info['estimated_freq_hz']:.2f} Hz "
                  f"(median Δt = {freq_info['median_dt']:.6f})")

    print(f"\n{'=' * 70}")
    print("  Summary complete.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Allow running this module directly for a quick check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_dataset_summary()

