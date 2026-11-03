"""
test_dataset_expansion.py — Unit Tests for Expanded Dataset Integration
========================================================================
Validates:
  • Manifest consistency and SHA-256 integrity.
  • Right Pocket (RP) loading.
  • Left Wrist (LW) loading.
  • Participant, placement, and session identification.
  • Deterministic collision disambiguation (sub19-lw-s2 and sub51-lw-s1).
  • Missing-session handling.
  • Zero recording overwrites.
  • Preprocessing metadata preservation.
"""

import os
import re
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    get_manifest_path,
    load_manifest,
    parse_filename,
    load_single_recording,
    load_all_recordings,
    load_recording_by_id,
    get_participant_recordings,
    get_project_root,
)
from src.preprocessing import (
    get_processed_data_dir,
    get_expanded_processed_data_dir,
)


@pytest.fixture(scope="module")
def manifest_df():
    """Load the dataset manifest once for all tests."""
    return load_manifest()


def test_manifest_consistency(manifest_df):
    """Verify manifest exists, contains all 272 recordings with unique IDs and hashes."""
    assert len(manifest_df) == 272, f"Expected 272 rows in manifest, found {len(manifest_df)}"
    assert manifest_df["recording_id"].nunique() == 272, "recording_id must be globally unique"
    assert manifest_df["canonical_id"].nunique() == 272, "canonical_id must be globally unique"
    assert manifest_df["sha256"].nunique() == 272, "Every file must possess a unique SHA-256 hash"

    # Check placement breakdown
    assert (manifest_df["placement"] == "RP").sum() == 144, "Expected 144 RP recordings"
    assert (manifest_df["placement"] == "LW").sum() == 128, "Expected 128 LW recordings"

    # Check session breakdown
    assert (manifest_df["session_int"] == 1).sum() == 154, "Expected 154 Session 1 recordings"
    assert (manifest_df["session_int"] == 2).sum() == 118, "Expected 118 Session 2 recordings"

    # Check original 20 baseline files
    assert manifest_df["is_original_20"].sum() == 20, "Expected exactly 20 original baseline files"

    # Verify that raw files referenced in manifest actually exist on disk
    project_root = get_project_root()
    for _, row in manifest_df.iterrows():
        raw_path = os.path.join(project_root, row["raw_path"])
        assert os.path.isfile(raw_path), f"Raw file missing: {raw_path}"


def test_rp_loading():
    """Verify loading RP recordings: exactly 144 recordings, correct keys and attributes."""
    rp_recs = load_all_recordings(placement="RP")
    assert len(rp_recs) == 144, f"Expected 144 RP recordings, got {len(rp_recs)}"

    for key, df in rp_recs.items():
        assert re.match(r"^sub\d+-rp-s\d+$", key), f"Unexpected RP key format: {key}"
        assert df.attrs["placement"] == "RP"
        assert df.attrs["participant_id"] >= 1
        assert df.attrs["session"] in (1, 2)
        assert "Timestamp" in df.columns
        assert "X" in df.columns and "Y" in df.columns and "Z" in df.columns


def test_lw_loading():
    """Verify loading LW recordings: exactly 128 recordings, correct keys and attributes."""
    lw_recs = load_all_recordings(placement="LW")
    assert len(lw_recs) == 128, f"Expected 128 LW recordings, got {len(lw_recs)}"

    for key, df in lw_recs.items():
        assert re.match(r"^sub\d+-lw-s\d+(-b\d+)?$", key), f"Unexpected LW key format: {key}"
        assert df.attrs["placement"] == "LW"
        assert df.attrs["participant_id"] >= 1
        assert df.attrs["session"] in (1, 2)


def test_participant_and_session_identification():
    """Verify filename parsing on various quirks: leading space, double extension, bout suffixes."""
    # Standard RP
    res = parse_filename(" sub1-rp-s1.csv")
    assert res["participant_id"] == 1
    assert res["placement"] == "RP"
    assert res["session"] == 1
    assert res["bout"] == 1

    # Standard LW
    res = parse_filename(" sub1-lw-s2.csv")
    assert res["participant_id"] == 1
    assert res["placement"] == "LW"
    assert res["session"] == 2
    assert res["bout"] == 1

    # Double extension
    res = parse_filename(" sub4-rp-s2.csv.csv")
    assert res["participant_id"] == 4
    assert res["placement"] == "RP"
    assert res["session"] == 2

    # Disambiguated bout filename
    res = parse_filename("sub19-lw-s2-b2.csv")
    assert res["participant_id"] == 19
    assert res["placement"] == "LW"
    assert res["session"] == 2
    assert res["bout"] == 2

    # Unrecognized filename
    res = parse_filename("random_noise.csv")
    assert res["participant_id"] is None
    assert res["placement"] is None
    assert res["session"] is None


def test_collision_disambiguation(manifest_df):
    """Verify that both collision pairs are explicitly disambiguated into distinct bouts."""
    # Collision 1: sub19-lw-s2
    c1 = manifest_df[manifest_df["canonical_id"].str.startswith("sub19-lw-s2")]
    assert len(c1) == 2, "Expected 2 entries for sub19-lw-s2"
    assert set(c1["canonical_id"]) == {"sub19-lw-s2-b1", "sub19-lw-s2-b2"}
    assert c1["recording_id"].nunique() == 2
    assert c1["sha256"].nunique() == 2
    assert c1.iloc[0]["t_min"] != c1.iloc[1]["t_min"]

    # Collision 2: sub51-lw-s1
    c2 = manifest_df[manifest_df["canonical_id"].str.startswith("sub51-lw-s1")]
    assert len(c2) == 2, "Expected 2 entries for sub51-lw-s1"
    assert set(c2["canonical_id"]) == {"sub51-lw-s1-b1", "sub51-lw-s1-b2"}
    assert c2["recording_id"].nunique() == 2
    assert c2["sha256"].nunique() == 2
    assert c2.iloc[0]["t_min"] != c2.iloc[1]["t_min"]


def test_missing_session_handling():
    """Verify graceful handling for participants with missing sessions or absent IDs."""
    # Participant 5 has RP_S2, LW_S1, LW_S2 (missing RP_S1)
    p5_recs = get_participant_recordings(5)
    assert len(p5_recs) == 3
    assert "sub5-rp-s2" in p5_recs
    assert "sub5-lw-s1" in p5_recs
    assert "sub5-lw-s2" in p5_recs
    assert "sub5-rp-s1" not in p5_recs

    # Participant 6 has RP_S2, LW_S1 (missing RP_S1, LW_S2)
    p6_recs = get_participant_recordings(6)
    assert len(p6_recs) == 2

    # Participant 73 is absent from the dataset
    p73_recs = get_participant_recordings(73)
    assert p73_recs == {}


def test_no_recording_overwrites():
    """Verify that loading all recordings yields exactly 272 unique entries with zero overwrites."""
    all_recs = load_all_recordings()
    assert len(all_recs) == 272, f"Expected 272 recordings, got {len(all_recs)}"
    # Verify collision files are distinct entries in the dictionary
    assert "sub19-lw-s2-b1" in all_recs
    assert "sub19-lw-s2-b2" in all_recs
    assert "sub51-lw-s1-b1" in all_recs
    assert "sub51-lw-s1-b2" in all_recs


def test_load_recording_by_id():
    """Verify load_recording_by_id supports both recording_id and canonical_id."""
    # By recording_id
    rec1 = load_recording_by_id("rec_001")
    assert rec1.attrs["recording_id"] == "rec_001"
    assert rec1.attrs["canonical_id"] == "sub1-rp-s1"

    # By canonical_id
    rec_c = load_recording_by_id("sub1-rp-s1")
    assert rec_c.attrs["recording_id"] == "rec_001"
    assert len(rec1) == len(rec_c)

    # Invalid ID
    with pytest.raises(KeyError):
        load_recording_by_id("rec_invalid_999")


def test_processed_data_isolation_and_metadata():
    """Verify data/processed/ is untouched and data/processed_expanded/ contains 532 segments with metadata."""
    orig_dir = get_processed_data_dir()
    exp_dir = get_expanded_processed_data_dir()

    # Original baseline processed segments must remain untouched (41 files)
    orig_csvs = [f for f in os.listdir(orig_dir) if f.endswith(".csv")]
    assert len(orig_csvs) == 41, f"Expected 41 original segments, found {len(orig_csvs)}"

    # Expanded processed directory must contain 532 segments
    assert os.path.isdir(exp_dir), "processed_expanded directory must exist"
    exp_csvs = [f for f in os.listdir(exp_dir) if f.endswith(".csv")]
    assert len(exp_csvs) == 532, f"Expected 532 expanded segments, found {len(exp_csvs)}"

    # Sample an expanded segment and verify required metadata columns
    sample_df = pd.read_csv(os.path.join(exp_dir, exp_csvs[0]))
    required_cols = [
        "participant_id",
        "placement",
        "session",
        "bout",
        "label",
        "segment_id",
        "stream_segment_id",
        "a_mag",
        "time_rel"
    ]
    for col in required_cols:
        assert col in sample_df.columns, f"Column {col} missing in preprocessed expanded file"

