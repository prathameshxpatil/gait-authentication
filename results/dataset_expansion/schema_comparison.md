# Schema Comparison: Existing Pipeline vs Expanded Dataset

**Audit Scope:** Comparison of 200GaitData Right Pocket (RP) baseline against the Expanded Dataset (RP + Left Wrist LW).  
**Generated Date:** 2026-11-01 23:48:08  

---

## 1. Column Specifications & Data Types

| Field Name | Pipeline Expected Format (RP) | Expanded Dataset: Right Pocket (RP) | Expanded Dataset: Left Wrist (LW) | Match Status |
| :--- | :---: | :---: | :---: | :---: |
| **Index Column** | `Unnamed: 0` (`int64` row index) | `Unnamed: 0` (`int64` row index) | `Unnamed: 0` (`int64` row index) | **PERFECT MATCH** |
| **Timestamp Column** | `Timestamp` (`float64` seconds) | `Timestamp` (`float64` seconds) | `Timestamp` (`float64` seconds) | **PERFECT MATCH** |
| **X-Axis Acceleration** | `X` (`float64`, units of $g$) | `X` (`float64`, units of $g$) | `X` (`float64`, units of $g$) | **PERFECT MATCH** |
| **Y-Axis Acceleration** | `Y` (`float64`, units of $g$) | `Y` (`float64`, units of $g$) | `Y` (`float64`, units of $g$) | **PERFECT MATCH** |
| **Z-Axis Acceleration** | `Z` (`float64`, units of $g$) | `Z` (`float64`, units of $g$) | `Z` (`float64`, units of $g$) | **PERFECT MATCH** |
| **Activity Label** | `Label` (`object` / `str`: 'G', 'R')| `Label` (`object` / `str`: 'G', 'R')| `Label` (`object` / `str`: 'G', 'R')| **PERFECT MATCH** |
| **Total Columns** | Exactly 6 columns | Exactly 6 columns | Exactly 6 columns | **100% Identical Schema** |

> [!NOTE]
> All 272 files across all 92 participants strictly share the exact identical 6-column tabular layout. No spurious, extra, missing, or renamed columns were discovered.

---

## 2. Physical Sensor & Kinematic Comparison

| Kinematic / Physical Dimension | Pipeline Expected (RP 10-Sub) | Expanded Right Pocket (144 files) | Expanded Left Wrist (128 files) | Scientific Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Nominal Sampling Rate** | 100.00 Hz | **100.06 Hz** (99.90 – 100.51 Hz) | **100.67 Hz** (99.90 – 101.12 Hz) | Synchronous 100 Hz sampling on both devices |
| **Missing Values (NaN / Inf)**| 0 | **0 NaNs across all files** | **0 NaNs across all files** | Complete, clean sensor records |
| **Sensor Modality** | 3-axis Accelerometer | 3-axis Accelerometer | 3-axis Accelerometer | Both placements measure tri-axial acceleration |
| **Physical Units** | $g$ ($1g \approx 9.81\text{ m/s}^2$) | $g$ ($1g \approx 9.81\text{ m/s}^2$) | $g$ ($1g \approx 9.81\text{ m/s}^2$) | Standard gravitational acceleration units |
| **Mean Vector Magnitude** | $\approx 1.06g$ | **1.142g** (range: 1.010 – 1.440g) | **1.069g** (range: 1.024 – 1.423g) | Both center near $1.0g$ baseline; RP exhibits higher gait impact shock |
| **Dominant Gravity Axis** | Y-axis (vertical in pocket) | **Y-axis:** mean = -0.666g | **Varies with arm posture:** X=0.570g, Y=-0.161g, Z=-0.003g | Pocket is vertically constrained; wrist rotates dynamically with arm swing |
| **Mean Duration per File** | $\approx 425$ s | **498.3 s** (308.7 – 638.9 s) | **506.8 s** (178.2 – 821.9 s) | Consistent 320 m corridor walking bouts |

---

## 3. Activity Label Distributions & Bout Segmentation

| Label Characteristic | Pipeline Expected (RP 10-Sub) | Expanded RP Files | Expanded LW Files | Total Expanded Files |
| :--- | :---: | :---: | :---: | :---: |
| **Files with both 'G' and 'R'** | 18 / 20 (90.0%) | 134 / 144 (93.1%) | 109 / 128 (85.2%) | **243 / 272 (89.3%)** |
| **Files with only 'G' ('Going')** | 2 / 20 (10.0%) | 10 / 144 (6.9%) | 19 / 128 (14.8%) | **29 / 272 (10.7%)** |
| **Files with only 'R' ('Returning')** | 0 / 20 (0.0%) | 0 / 144 (0.0%) | 0 / 128 (0.0%) | **0 / 272 (0.0%)** |
| **Files with unexpected labels** | 0 / 20 (0.0%) | 0 / 144 (0.0%) | 0 / 128 (0.0%) | **0 / 272 (0.0%)** |
| **Bout Transition Turnaround Gaps**| 18 / 20 files | 134 / 144 files | 110 / 128 files | **244 / 272 files** |

---

## 4. Current Pipeline Compatibility & Refactoring Requirements

1. **`src/data_loader.py` Hardcoded Regex:**
   - In `parse_filename()`, line 118: `re.search(r"sub(\d+)-rp-s(\d+)", filename)` assumes `-rp-`.
   - *Impact:* Fails on `-lw-` (Left Wrist) files, setting `participant_id = None`.
   - *Refactoring Needed:* Update regex to `re.search(r"sub(\d+)-(rp|lw)-s(\d+)", filename, re.IGNORECASE)`.
2. **Double Extension Handling:**
   - 34 files have `.csv.csv`. While `discover_csv_files()` currently finds them via glob `*.csv.csv`, any strictly `.csv`-matching code that splits on `.` might extract `.csv` as a stem.
3. **Leading Whitespace:**
   - 100% of raw files (272/272) begin with a leading space ` sub...`.
4. **Collision Handling in Data Loader Keys:**
   - `load_all_recordings()` generates dictionary keys via `f"sub{info['participant_id']}-rp-s{info['session']}"`.
   - If Left Wrist is added, keys will collide unless placement is preserved: `f"sub{pid}-{placement.lower()}-s{session}"`.
   - Furthermore, the two collision instances (`sub19-lw-s2` and `sub51-lw-s1`) require disambiguation tags.
