# Complete Read-Only Audit Report: data/raw/

**Dataset:** Expanded 200GaitData Accelerometer Collection  
**Audit Directory:** `data/raw/`  
**Audit Mode:** Strictly Read-Only (Zero files modified, deleted, or moved)  
**Audit Execution Date:** 2026-11-01 23:48:08  

---

## 1. Executive Summary

A comprehensive, non-destructive audit of `data/raw/` was performed across all files. The expanded dataset significantly broadens the experimental foundation from the initial 10-participant benchmark to **92 unique participants** across two sensor placements: **Right Pocket (RP)** and **Left Wrist (LW)**.

### Core Metrics Summary

| Metric Dimension | Value | Verified Finding |
| :--- | :---: | :--- |
| **Total Raw CSV Files** | **272** | Every file is a tri-axial accelerometer recording |
| **Total Dataset Storage Size** | **1,048.42 MB (1.05 GB)** | Mean file size: 3.85 MB (range: 2.37 – 4.91 MB) |
| **Total Unique Participants** | **92** | Participant IDs span 1 to 93 (PID 73 is absent) |
| **Right Pocket (RP) Files** | **144** | 79 Session 1 + 65 Session 2 recordings |
| **Left Wrist (LW) Files** | **128** | 75 Session 1 + 53 Session 2 recordings |
| **Exact SHA-256 Duplicates** | **0** | Every single file possesses a distinct hash |
| **Filename Collisions (Same clean name, different content)** | **2** | `sub19-lw-s2` (2 files) and `sub51-lw-s1` (2 files) |
| **Missing Data / NaNs** | **0** | Zero missing values across all 272 files |
| **Sampling Frequency** | **~100 Hz** | RP mean: 100.06 Hz, LW mean: 100.67 Hz |
| **Original 20 Benchmark Files** | **20 / 20 Confirmed** | 100% of Phase 1–5 files present with matching content |

---

## 2. Participant Coverage & Session Matrix

Out of 92 unique participants, the four-slot coverage matrix ($2\\text{ placements} \\times 2\\text{ sessions}$) is distributed as follows:

```
Total Participants: 92
├── Complete Coverage (4 / 4 slots) : 31 participants (33.7%)
├── Three Slots Filled (3 / 4 slots) : 31 participants (33.7%)
├── Two Slots Filled (2 / 4 slots)   : 23 participants (25.0%)
└── Single Slot Filled (1 / 4 slots) :  7 participants (7.6%)
```

### Research-Ready Participant Subsets

1. **Cross-Session Right Pocket Cohort (RP_S1 + RP_S2):**
   - **55 participants** have both Session 1 and Session 2 for the Right Pocket.
   - *Feasibility:* Immediately suitable for large-scale replication of our cross-session adaptive authentication pipeline (a **5.5× expansion** over our 10-subject baseline).
   - *Included Participants:* [1, 2, 3, 4, 10, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24]... (55 total).
2. **Cross-Session Left Wrist Cohort (LW_S1 + LW_S2):**
   - **42 participants** have both Session 1 and Session 2 for the Left Wrist.
   - *Feasibility:* Enables cross-session wearable wrist gait biometric authentication.
3. **Full Multi-Modal Multi-Session Cohort (All 4 Slots):**
   - **31 participants** possess RP_S1, RP_S2, LW_S1, and LW_S2.
   - *Feasibility:* Enables simultaneous multi-device fusion (Pocket + Wrist) cross-session evaluation.
4. **Placement-Exclusive Cohorts:**
   - **RP Only (No LW data):** 8 participants ([45, 47, 48, 50, 72, 84, 89, 91]).
   - **LW Only (No RP data):** 3 participants ([29, 70, 77]).

---

## 3. Anomaly & Integrity Findings

### 3.1 Filename Quirks
1. **Leading Whitespace:** **272 of 272 files (100%)** begin with a leading space character (` sub...`).
2. **Double Extension (`.csv.csv`):** **34 files (12.5%)** end in `.csv.csv` rather than `.csv`.
   - Examples include original files (`sub4-rp-s2.csv.csv`, `sub18-rp-s2.csv.csv`) and 32 expanded files.

### 3.2 Filename Collisions (Disambiguation Required)
Two instances occur where two distinct files normalize to the same clean filename:
1. **Participant 19 Left Wrist Session 2:**
   - File A: `" sub19-lw-s2.csv"` (3.77 MB, 47,000 rows, $t = 4408.04 \\to 4907.26$ s)
   - File B: `" sub19-lw-s2.csv.csv"` (3.84 MB, 47,000 rows, $t = 31090.85 \\to 31590.75$ s)
   - *Verification:* Both are genuine walking bouts from Sub 19 recorded $\\approx 7.3$ hours apart in Session 2.
2. **Participant 51 Left Wrist Session 1:**
   - File A: `" sub51-lw-s1.csv"` (3.81 MB, 47,000 rows, $t = 6265.31 \\to 6764.61$ s)
   - File B: `" sub51-lw-s1.csv.csv"` (3.73 MB, 46,000 rows, $t = 6999.98 \\to 7489.98$ s)
   - *Verification:* Two consecutive walking bouts recorded with a 235-second gap during Session 1.

### 3.3 Exact SHA-256 Duplicates
- **0 exact duplicates found.** Every file represents unique physical data.

---

## 4. Signal Quality & Schema Verification

- **Schema Parity:** Every file strictly possesses `['Unnamed: 0', 'Timestamp', 'X', 'Y', 'Z', 'Label']`.
- **Missing Data:** Exactly **0 NaNs** or missing values across the entire dataset.
- **Labels:** Strictly `'G'` ("Going", outward 320 m corridor) and `'R'` ("Returning", return 320 m corridor).
  - 243 files contain both `'G'` and `'R'` separated by a turnaround pause.
  - 29 files contain only `'G'` (uninterrupted outward walk).
  - 0 files contain any corrupted or unassigned label.
- **Physical Modality:** Tri-axial accelerometer.
  - RP exhibits vertical gravity vector on the Y-axis (mean $\\approx -0.78g$, vector magnitude $\\approx 1.14g$).
  - LW exhibits rotational variance during arm swing (vector magnitude $\\approx 1.07g$).

---

## 5. Status of Original 20 Recordings

All 20 recordings utilized in Phases 1–5 were successfully identified in `data/raw/`:
- **PIDs:** 1, 2, 3, 4, 10, 13, 14, 15, 16, 18 (RP, S1 and S2).
- **Integrity:** SHA-256 hashes and row counts match historical baseline benchmarks byte-for-byte.

---

## 6. Recommendations for Dataset Organization

To integrate this expanded collection without disrupting existing code or overwriting historical benchmarks:

1. **Keep Raw Data Immutable:** Retain `data/raw/` in its untouched state as the immutable archival source.
2. **Implement Structured Ingestion:**
   - Create a sanitized manifest linking raw filenames to canonical identifiers: `sub<ID>_<placement>_<session>_<bout>`.
   - Disambiguate the two collision files as `sub19_lw_s2_run1` / `run2` and `sub51_lw_s1_run1` / `run2`.
3. **Pipeline Refactoring (`src/data_loader.py` & `src/preprocessing.py`):**
   - Generalize regex from `sub(\d+)-rp-s(\d+)` to `sub(\d+)-(rp|lw)-s(\d+)`.
   - Preserve placement (`placement: 'RP' | 'LW'`) in `GaitWindow` dataclass and stream segment IDs.
4. **Experimental Phasing:**
   - **Phase 6A (RP 55-Subject Scale-Up):** Re-run Candidate-Buffer Dual-Gallery on the 55 complete RP participants to evaluate scalability and poisoning resistance across 5.5× more users.
   - **Phase 6B (LW 42-Subject Evaluation):** Benchmark DTW authentication on Left Wrist gait dynamics.
   - **Phase 6C (Multi-Modal Fusion):** Test score-level fusion on the 31 participants with complete multi-sensor recordings.
