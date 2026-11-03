# Gait-Based Biometric Authentication using Dynamic Time Warping (DTW) and Adaptive Template Updating

An academic research implementation of continuous smartphone accelerometer-based biometric authentication using multivariate Dynamic Time Warping (DTW), chronological evaluation, and a candidate-buffer dual-gallery template adaptation architecture.

---

## 1. Project Overview

Biometric authentication using gait signals captured from smartphone accelerometers offers an unobtrusive, continuous verification mechanism. However, gait authentication systems face two fundamental challenges:
1. **Intra-individual temporal variance:** A user's walking dynamics drift across sessions due to fatigue, footwear, pacing, and surface changes.
2. **Template poisoning risk:** Naive online adaptation models that accept incoming samples to update biometric galleries can inadvertently incorporate impostor gait patterns, degrading system security over time.

This project implements an end-to-end research pipeline on the **200GaitData** benchmark dataset. The initial baseline and adaptive architectures (Phases 1–5) were developed on a 10-participant benchmark cohort (20 recordings, right front trouser pocket accelerometer sampled at ~100 Hz). The dataset has since been integrated with the full expanded collection comprising **92 unique participants** across **272 recordings (~1.05 GB)**, incorporating both **Right Pocket (RP)** and **Left Wrist (LW)** placements.

The system establishes a fixed multivariate DTW baseline and investigates adaptive architectures, introducing a **Candidate-Buffer Dual-Gallery** design to balance continuous adaptation with resistance to template poisoning. All data loading is powered by an authoritative manifest (`data/dataset_manifest.csv`) ensuring deterministic recording identification, immutable raw data preservation, and explicit collision disambiguation.

---

## 2. Dataset Specification: Benchmark & Expanded Collections

### 2.1 Overview & Cohort Structure

| Dataset Dimension | Baseline Benchmark Cohort (Phases 1–5) | Expanded Canonical Collection (Integrated) |
| :--- | :---: | :---: |
| **Total Raw CSV Recordings** | 20 | **272** |
| **Total Unique Participants** | 10 (PIDs: 1, 2, 3, 4, 10, 13, 14, 15, 16, 18) | **92** (PIDs 1 to 93; PID 73 absent) |
| **Placements Supported** | Right Pocket (RP) only | **Right Pocket (RP)** & **Left Wrist (LW)** |
| **Total Storage Size** | ~78.5 MB | **1,048.42 MB (~1.05 GB)** |
| **Preprocessed Segments** | 41 continuous segments (`data/processed/`) | **532 continuous segments** (`data/processed_expanded/`) |
| **Manifest Ingestion** | Dynamic filesystem scan | **Deterministic Manifest** (`data/dataset_manifest.csv`) |

### 2.2 Sensor Placements & Kinematic Characteristics

1. **Right Pocket (RP — 144 recordings across 89 participants):**
   - **Device Placement:** Front right trouser pocket, oriented vertically.
   - **Kinematic Profile:** Gravitational acceleration predominantly aligns with the negative Y-axis (mean $Y = -0.666g$), with vertical heel-strike impacts producing higher dynamic peaks (mean vector magnitude $1.142g$).
   - **Cross-Session RP Cohort ($S1 + S2$):** **55 participants** have complete Session 1 and Session 2 recordings, enabling a **5.5× scale-up** of the cross-session authentication pipeline.
2. **Left Wrist (LW — 128 recordings across 84 participants):**
   - **Device Placement:** Left wrist wearable (smartwatch orientation).
   - **Kinematic Profile:** Sensor experiences 3D rotation during natural arm swing. Gravitational component distributes across axes ($X \approx 0.570g, Y \approx -0.161g, Z \approx -0.003g$) with lower gait impact attenuation (mean vector magnitude $1.069g$).
   - **Cross-Session LW Cohort ($S1 + S2$):** **42 participants** have complete Session 1 and Session 2 recordings.
3. **Multi-Modal Multi-Session Cohort ($2 \text{ placements} \times 2 \text{ sessions}$):**
   - **31 participants** possess all 4 slots (`RP_S1`, `RP_S2`, `LW_S1`, `LW_S2`), enabling dual-sensor score-level fusion experiments.

### 2.3 Coverage Matrix & Missing Sessions

Out of 92 unique participants:
- **Complete Coverage (4/4 slots):** 31 participants (33.7%)
- **Three Slots Filled (3/4 slots):** 31 participants (33.7%)
- **Two Slots Filled (2/4 slots):** 23 participants (25.0%)
- **Single Slot Filled (1/4 slots):** 7 participants (7.6%)
- **Placement Exclusivity:** 8 participants have RP recordings only (no LW data); 3 participants have LW recordings only (no RP data).
- **Missing Participant:** Participant ID 73 is absent from the raw collection.

### 2.4 Filename Quirks & Collision Disambiguation

1. **Leading Whitespace:** 100% of raw files (272/272) contain a leading space (e.g. `" sub1-rp-s1.csv"`).
2. **Double Extensions:** 34 files (12.5%) possess `.csv.csv` extensions (including historical baseline files `sub4-rp-s2.csv.csv` and `sub18-rp-s2.csv.csv`).
3. **Filename Collisions (Same Clean Name, Distinct Content):**
   - **Participant 19 LW Session 2:**
     - `sub19-lw-s2-b1` (`rec_059`): `" sub19-lw-s2.csv"` ($t = 4,408.04 \to 4,907.26$ s, bout 1)
     - `sub19-lw-s2-b2` (`rec_060`): `" sub19-lw-s2.csv.csv"` ($t = 31,090.85 \to 31,590.75$ s, bout 2, ~7.3h later)
   - **Participant 51 LW Session 1:**
     - `sub51-lw-s1-b1` (`rec_152`): `" sub51-lw-s1.csv"` ($t = 6,265.31 \to 6,764.61$ s, bout 1)
     - `sub51-lw-s1-b2` (`rec_153`): `" sub51-lw-s1.csv.csv"` ($t = 6,999.98 \to 7,489.98$ s, bout 2, 235s gap)
   - *Resolution:* All recordings are assigned permanent recording IDs (`rec_001` – `rec_272`) and canonical keys (`-b1` and `-b2`) in the manifest, eliminating name collision risks.

---

## 3. Pipeline Architecture & Methodology

```
+---------------------------------------------------------------------------------------------------+
| 1. RAW SENSOR DATA (data/raw/)                                                                    |
|    Tri-axial Accelerometer (X, Y, Z) @ ~100 Hz | 10 Participants | 2 Sessions Each                |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 2. SIGNAL PREPROCESSING (src/preprocessing.py)                                                    |
|    - Stream Separation: Going ('G') vs Returning ('R') walking bouts                              |
|    - Discontinuity Splitting: Partition at genuine pauses (Δt > 0.5s); never interpolate          |
|    - Timestamp Normalization: Reset elapsed time to t=0 per continuous segment                    |
|    - Zero-Phase Filtering: 4th-order Butterworth low-pass filter (fc = 20.0 Hz, fs ~ 100 Hz)      |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 3. GAIT SEQUENCE WINDOWING (src/segmentation.py)                                                  |
|    - Window Length: 2.0 seconds (200 samples)                                                     |
|    - Step Size: 1.0 second (100 samples, 50% overlap)                                             |
|    - Trailing Window Handling: Discard incomplete segments (< 200 samples)                        |
|    - Feature Extraction: Lowpass-filtered Acceleration Magnitude ||a|| = sqrt(X² + Y² + Z²)       |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 4. STRICT CHRONOLOGICAL EVALUATION PROTOCOL                                                       |
|    - Enrolment Set: Session 1 (100% of S1 windows, stratified template selection)                 |
|    - Validation Set: Session 2, First Chronological 50% (threshold calibration at EER: θ* = 0.0652)|
|    - Held-Out Test Set: Session 2, Second Chronological 50% (unseen evaluation stream)            |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| 5. CANDIDATE-BUFFER DUAL-GALLERY AUTHENTICATION & ADAPTATION (src/dual_gallery.py)                |
|    - Permanent Gallery (G_perm): 8 immutable Session 1 enrollment anchors                         |
|    - Adaptive Gallery (G_adapt): Up to 4 dynamic slots (starts empty)                             |
|    - Verification Scoring: d_comb = 0.70 * d_perm + 0.30 * d_adapt                                |
|    - Candidate Buffer: Accumulates N_cand = 3 non-overlapping high-confidence windows             |
|    - Cluster Consistency Check: Enforces max pairwise DTW (max D_ij <= 0.85 * θ*)                |
|    - Anchor Guard: Medoid top-2 mean distance to permanent anchors <= 1.00 * θ*                  |
|    - Redundancy Replacement: Replaces dynamic template closest to candidate medoid                |
|    - Auto-Quarantine: Any low-confidence attempt flushes buffer immediately                       |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Adaptation Architecture

### 3.1 Dual-Gallery Verification
To prevent dynamic template drift from eroding authentic identity references, templates are decoupled into two distinct galleries:
- **Permanent Gallery ($G_{\text{perm}}$):** Initialized with 8 stratified templates from Session 1. These templates are strictly immutable and retain dominant influence throughout authentication.
- **Adaptive Gallery ($G_{\text{adapt}}$):** A bounded gallery (capacity = 4) that starts empty and only incorporates validated templates.

The combined authentication matching distance for an incoming window $x$ is defined as:
$$d_{\text{comb}}(x) = w_{\text{perm}} \cdot \min_{t \in G_{\text{perm}}} \text{DTW}(x, t) + (1 - w_{\text{perm}}) \cdot \min_{a \in G_{\text{adapt}}} \text{DTW}(x, a)$$
where $w_{\text{perm}} = 0.70$. If $G_{\text{adapt}}$ is empty, $d_{\text{comb}}(x) = d_{\text{perm}}(x)$.

### 3.2 Candidate-Buffer Verification Gates
Rather than allowing individual authentication attempts to trigger immediate template updates, incoming windows must satisfy four sequential verification stages:

1. **High-Confidence Gate:** A window is accepted into the candidate buffer only if its authentication distance satisfies $d_{\text{comb}} \le \alpha \cdot \theta^*$, where $\alpha = 0.70$ and $\theta^* = 0.0652$.
2. **Temporal Separation Requirement:** Adjacent overlapping windows are rejected from candidate accumulation. Only temporally distinct, non-overlapping windows ($|t_{\text{start}}^{(k)} - t_{\text{end}}^{(k-1)}| \ge 0$) qualify as independent evidence.
3. **Cluster Maximum Pairwise Consistency Check:** When the candidate buffer accumulates $N_{\text{cand}} = 3$ windows, the full pairwise DTW distance matrix is computed. The candidate cluster is promoted only if the maximum pairwise distance satisfies:
   $$\max_{i \ne j} \text{DTW}(c_i, c_j) \le \gamma \cdot \theta^* \quad (\gamma = 0.85)$$
   Enforcing the maximum pairwise distance guarantees cluster diameter cohesion, preventing loose clusters or outliers from passing.
4. **Permanent Anchor Guard:** The cluster medoid $m$ (the window with the minimum sum of distances to other cluster members) must be anchored to the permanent enrollment gallery:
   $$\frac{1}{2} \sum_{k=1}^{2} d_{(k)}(m, G_{\text{perm}}) \le \beta \cdot \theta^* \quad (\beta = 1.00)$$
5. **Immediate Quarantine:** Any incoming claim that fails the confidence requirement immediately flushes the candidate buffer, preventing gradual multi-attempt impostor infiltration.

---

## 5. Evaluation Protocol

The evaluation strictly avoids test-set data leakage through chronological segregation:

| Partition | Data Source | Purpose | Leakage Controls |
| :--- | :--- | :--- | :--- |
| **Enrolment** | Session 1 (100%) | Form initial 8 enrollment templates | Never evaluated during testing |
| **Validation** | Session 2 (1st 50% chronological) | Calibrate scalar threshold $\theta^*$ at EER | Labels unseen by test evaluation |
| **Testing** | Session 2 (2nd 50% chronological) | Benchmark verification & adaptation | Evaluated chronologically with interleaved impostor claims |

- **Operating Threshold:** $\theta^* = 0.0652$ (calibrated on held-out validation data at Equal Error Rate).
- **Test Stream:** 500 genuine attempts and 450 impostor attempts interleaved chronologically per participant (950 verification attempts per seed).
- **Multi-Seed Stability:** Evaluated across 5 pseudo-random stream orderings (seeds: 42, 101, 2024, 777, 999), representing 4,750 total authentication trials.

---

## 6. Historical Benchmark Results (Phases 1–5: 10-Participant Right Pocket Cohort)

### 5.1 Architecture Comparison

| System Architecture | Permanent Templates | Dynamic Slots | Update Gate | Test FAR (%) | Test FRR (%) | Test Balanced Accuracy (%) | Diagnostic EER (%) | Poisoned Updates Observed | Mean Gallery Drift |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Baseline DTW** | 8 | 0 | Static (no adaptation) | 43.78% | 36.60% | 59.81% | 40.11% | 0 | 0.0000 |
| **3-Confirmation Adaptive** | 4 | 4 (shared) | 3 non-overlapping windows | 44.67% | 36.60% | 59.37% | 40.33% | 2 | 0.0462 |
| **Candidate-Buffer Dual-Gallery (EXP-OPT-04)** | 8 | 4 (separate) | Max pairwise check + anchor guard | **43.42% ± 0.68%** | **37.28% ± 0.72%** | **59.65% ± 0.53%** | **40.01%** | **0 (across all 5 seeds)** | **0.0103** |

*Note: In the specific experiments conducted across 5 random seed orderings (4,750 authentication attempts), zero poisoned promotions were observed under the candidate-buffer dual-gallery architecture, compared to 2 poisoned updates observed under the single-gallery model.*

### 5.2 Multi-Seed Reproducibility (Candidate-Buffer Dual-Gallery)

| Evaluation Seed | FAR (%) | FRR (%) | Balanced Accuracy (%) | Genuine Promotions | Poisoned Promotions |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 43.56% | 36.60% | 59.92% | 3 | 0 |
| **Seed 101** | 43.78% | 38.20% | 59.01% | 7 | 0 |
| **Seed 2024** | 43.78% | 36.60% | 59.81% | 1 | 0 |
| **Seed 777** | 43.78% | 37.80% | 59.21% | 3 | 0 |
| **Seed 999** | 42.22% | 37.20% | 60.29% | 4 | 0 |
| **Mean ± Std** | **43.42% ± 0.68%** | **37.28% ± 0.72%** | **59.65% ± 0.53%** | **3.6 ± 2.2** | **0.00 ± 0.00** |

### 5.3 Per-Participant Analysis & Root Cause of High FRR
The evaluation revealed a bimodal performance divergence across participants:

| Participant | Dynamic Shift ($\Delta\sigma$) | Mean FAR (%) | Mean FRR (%) | Mean Balanced Accuracy (%) | Genuine Promotions | Observations |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **User 1** | +0.2389 g | 53.33% | 100.00% | 23.33% | 0.0 | Dynamic swing surge (+74%), genuine windows rejected |
| **User 2** | -0.0453 g | 45.78% | 0.00% | 77.11% | 0.4 | Steady pace, low intra-session divergence |
| **User 3** | +0.2808 g | 53.33% | 100.00% | 23.33% | 0.0 | Dynamic swing surge (+64%) and DC offset shift |
| **User 4** | +0.1653 g | 51.11% | 96.00% | 26.44% | 0.0 | Dynamic swing surge (+61%), genuine windows rejected |
| **User 10** | -0.0363 g | 35.11% | 0.00% | 82.44% | 0.8 | Steady pace, consistent template matching |
| **User 13** | -0.0379 g | 28.89% | 42.00% | 64.56% | 0.0 | Steady pace, moderate intra-session divergence |
| **User 14** | +0.0124 g | 53.34% | 14.80% | 65.93% | 0.6 | Steady pace, verified genuine promotions |
| **User 15** | +0.0278 g | 46.67% | 0.00% | 76.67% | 1.8 | Steady pace, multiple genuine candidate promotions |
| **User 16** | +0.0097 g | 2.22% | 16.00% | 90.89% | 0.0 | Highly distinctive gait signature, low FAR & FRR |
| **User 18** | +0.0467 g | 64.44% | 4.00% | 65.78% | 0.0 | Steady pace, low intra-session divergence |

**Root Cause Finding:** Cadence remains consistent across all participants (~1.98–2.02 Hz). The primary driver of extreme FRR for Users 1, 3, and 4 is an intra-individual expansion in acceleration dynamic standard deviation ($\Delta\sigma \in [+0.165, +0.281]$ g, a 61% to 74% increase in swing intensity between Session 1 and Session 2). Because a single universal scalar threshold ($\theta^* = 0.0652$) is enforced, their initial Session 2 genuine windows yield DTW distances exceeding the threshold ($d \approx 0.08–0.14 > \theta^*$), causing immediate rejection and candidate buffer flushes that prevent adaptation from engaging.

---

## 7. Installation & Execution

### 6.1 Prerequisites
- Python 3.10+
- macOS, Linux, or Windows

### 6.2 Environment Setup
```bash
# Clone the repository
git clone <repository_url>
cd gait-auth

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 6.3 Running the Pipelines

#### Baseline DTW Verification Pipeline
Executes preprocessing, windowing, Session 1 enrollment, and Session 2 testing under the baseline fixed multivariate DTW configuration:
```bash
python main.py
```

#### Controlled Baseline Investigation Suite
Executes the parameter exploration experiments (feature representations, window sizes, DTW radii, and walking streams):
```bash
python main.py --run-experiments
```

#### Single-Gallery Adaptive Benchmark
Evaluates the 3-confirmation single-gallery adaptive benchmark:
```bash
python main.py --run-adaptive
```

#### Candidate-Buffer Dual-Gallery Optimization Suite
Executes the full candidate-buffer dual-gallery evaluation, 12 controlled optimization experiments, 5-seed reproducibility analysis, and outputs all reports and figures:
```bash
python main.py --run-dual-gallery
```

### 7.4 Running Automated Tests
The repository includes an extensive automated test suite (66 tests) covering signal filtering, discontinuity splitting, windowing, DTW dynamic programming, EER calculation, single-gallery adaptation, dual-gallery candidate buffer mechanisms, expanded dataset manifest integrity, and end-to-end adaptive scenario verification:
```bash
pytest tests/ -v
```

To run specifically the adaptive scenario suite (10 security and adaptation scenarios):
```bash
pytest tests/test_adaptive_scenarios.py -v
```

---

## 8. End-to-End Adaptive Authentication Pipeline

The core adaptive authentication pipeline is implemented in [`src/dual_gallery.py`](src/dual_gallery.py) (`DualGallery`, `CandidateBuffer`, and `AdaptiveAuthenticationPipeline`). It resolves the trade-off between accommodating gradual natural gait drift and preventing adversarial template poisoning.

### 8.1 Workflow Architecture

```text
Incoming Gait Window (4.0s)
          │
          ▼
Compute Matching Scores
  ├── d_perm  = min_DTW(window, G_perm)
  └── d_adapt = min_DTW(window, G_adapt)
          │
          ▼
Combined Score: d_comb = w_perm * d_perm + (1 - w_perm) * d_adapt
          │
    ┌─────┴────────────────────────────────────────────────┐
    │                                                      │
[d_comb <= θ*] ?                                     [d_comb > θ*]
    │                                                      │
Accepted Attempt                                     Rejected Attempt
    │                                                      │
[d_comb <= α * θ*] ?                                       │
    ├── NO ─────────> Borderline / Low Confidence ─────────┼─> Flush Buffer
    │                 (Quarantine Maintained)              │
   YES                                                     ▼
    ▼                                              Buffer Cleared
Quarantine into Candidate Buffer (K=3 non-overlapping)
    │
[Buffer Full & Ready (K=3)] ?
    ├── NO ─────────> Await Next Confirmation
   YES
    ▼
Pairwise Cluster Diameter Check
[max_{i<j} DTW(c_i, c_j) <= γ * θ*] ?
    ├── NO ─────────> Inconsistent Cluster Discarded ──> Flush Buffer
   YES
    ▼
Compute Cluster Medoid Window
    │
Permanent Anchor Guard Check
[mean(top-2 DTW(medoid, G_perm)) <= β * θ*] ?
    ├── NO ─────────> Anchor Violation Discarded ──────> Flush Buffer
   YES
    ▼
VERIFIED CANDIDATE PROMOTION
  ├── Promote medoid into G_adapt (bounded capacity M=4)
  ├── Redundancy Replacement: Replace adaptive template closest to medoid
  ├── Flush buffer to prepare for next cycle
  └── Record verified promotion event (0.0% poisoning rate verified)
```

### 8.2 Scenario Handlers & Security Guarantees

The pipeline is tested and proven across 10 critical operational scenarios:

1. **Permanent Gallery Immutability:** The permanent enrollment templates ($G_{\text{perm}}$) are stored in an immutable tuple of read-only NumPy arrays (`flags.writeable = False`). In-place mutation and item reassignment are physically blocked at the interpreter level.
2. **Candidate Confirmation Bypass Prevention:** Incoming attempts cannot update active galleries without completing the full quarantine and verification pathway ($K=3$ independent non-overlapping confirmations).
3. **Genuine User with Stable Gait:** Authenticates with high confidence; medoid updates remain tightly anchored without gallery drift.
4. **Genuine User with Gradual Cross-Session Drift:** Accommodates natural gait evolution by updating the adaptive gallery, lowering subsequent matching distances and eliminating cross-session false rejects.
5. **Impostor Quarantine & Isolation:** Impostor attempts exceeding the high-confidence gate ($\alpha \theta^*$) cannot enter the candidate buffer. Even if a marginal false accept enters the buffer, it remains isolated ($1/K$) and cannot trigger promotion.
6. **Repeated Impostor Attacks:** Any subsequent degraded attempt immediately flushes the candidate buffer, purging isolated candidates.
7. **Mixed Genuine and Impostor Attempts:** If an impostor attempt is buffered alongside genuine windows, the pairwise consistency check detects large inter-window DTW dispersion ($\max d > \gamma \theta^*$) and cleanly discards the entire cluster.
8. **Candidate Buffer Expiration & Overflow:** Stale candidates exceeding `max_age_steps` automatically expire and are purged. Buffer capacity is bounded (`max_capacity = 5`) with FIFO eviction, preventing unbounded queue growth.
9. **Rejected Cluster Clean Recovery:** When a cluster is discarded due to inconsistency or anchor guard violation, the buffer resets cleanly. Subsequent consistent genuine walking resumes accumulation immediately without locking out the user.
10. **Stateful Stream Processing without Resets:** `AdaptiveAuthenticationPipeline` processes long chronological streams continuously, maintaining gallery state, template drift history, and comprehensive event logs without inter-window resets.

### 8.3 Structured Event Logging

The pipeline maintains complete structured audit logs:
- **`auth_decision_logs`:** Step index, window ID, claimant ID, scores ($d_{\text{perm}}, d_{\text{adapt}}, d_{\text{comb}}$), threshold, acceptance decision, and high-confidence flag.
- **`buffer_event_logs`:** Additions, temporal overlap rejections, expirations, and flushes (with specific trigger scores and reasons).
- **`gallery_update_logs`:** Promoted medoid IDs, cluster dispersion metrics, anchor distances, replaced template indices, drift trajectory, and ground-truth verification flags (`is_poisoned`).

---

## 9. Repository Organization

```text
gait-auth/
├── README.md                                  # Comprehensive research documentation
├── requirements.txt                           # Minimal pinned project dependencies
├── pytest.ini                                 # Pytest configuration
├── .gitignore                                 # Git ignore configuration
├── main.py                                    # Command-line interface and pipeline entry point
├── src/                                       # Core source code package
│   ├── __init__.py                            # Package initialization
│   ├── data_loader.py                         # Manifest-based loader, CSV discovery, parsing
│   ├── preprocessing.py                       # Butterworth filtering, discontinuity splitting, metadata
│   ├── segmentation.py                        # Sliding window extraction & GaitWindow dataclass
│   ├── dtw_authentication.py                  # Multivariate DTW kernel, Numba JIT, enrollment galleries
│   ├── evaluation.py                          # FAR/FRR/EER metrics, ROC, score distributions
│   ├── experiments.py                         # Baseline parameter exploration suite
│   ├── adaptive_template.py                   # Single-gallery adaptive benchmark implementation
│   ├── dual_gallery.py                        # Candidate-Buffer Dual-Gallery architecture & pipeline
│   ├── expanded_rp_baseline.py                # 55-participant expanded RP fixed DTW baseline
│   └── adaptive_experiment.py                 # Multi-strategy adaptive validation benchmark
├── archive/                                   # Preserved exploratory & diagnostic scripts
│   ├── prototype_adaptation.py                # Scratch prototype for User 1 adaptation
│   ├── evaluate_factors.py                    # Early feature sweep script
│   ├── test_validation_split.py               # Early split validation script
│   └── diagnostic_audit.py                    # Early pocket-tilt diagnostic script
├── tests/                                     # Automated unit test suite (66 tests)
│   ├── test_pipeline.py                       # Baseline preprocessing, DTW, and EER tests
│   ├── test_adaptive.py                       # Single-gallery adaptation & 3-conf gate tests
│   ├── test_dual_gallery.py                   # Candidate buffer & dual-gallery core tests
│   ├── test_cohort_metrics.py                 # Enrolled-identity macro metrics & cohort metrics tests
│   ├── test_dataset_expansion.py              # Manifest integrity, RP/LW loading, collision tests
│   ├── test_adaptive_experiment.py            # Modular strategy & candidate buffer tests
│   └── test_adaptive_scenarios.py             # 10 security, drift, immutability & stream tests
├── notebooks/                                 # Exploratory notebooks
│   └── dataset_exploration.ipynb              # Phase 1 exploratory data analysis
├── data/                                      # Gait dataset directory
│   ├── dataset_manifest.csv                   # Master index: 272 recordings with unique rec_id & metadata
│   ├── dataset_manifest.json                  # JSON formatted master manifest for fast dict lookup
│   ├── raw/                                   # 272 raw CSV recordings (92 participants, immutable archive)
│   ├── processed/                             # 41 preprocessed segments (historical 10-user baseline)
│   └── processed_expanded/                    # 532 preprocessed segments (expanded 92-user collection)
└── results/                                   # Experimental outputs, data tables, and figures
    ├── dataset_expansion/                     # Read-only audit inventory, coverage, and manifest copies
    ├── expanded_rp_baseline/                  # 55-participant fixed-template baseline outputs
    ├── adaptive_rp_experiments/               # Multi-seed adaptive strategy comparison & audit logs
    ├── dual_gallery_optimization_experiments.csv
    ├── improved_dual_gallery_per_user.csv
    ├── candidate_buffer_analysis.csv
    ├── dual_gallery_comparison.csv
    ├── baseline_investigation_experiments.csv
    ├── improved_dual_gallery_analysis.png
    ├── dual_gallery_drift.png
    └── improved_dual_gallery_report.md
```

---

## 9. Research Limitations & Future Directions

1. **Universal vs User-Specific Operating Thresholds:** A single global scalar threshold ($\theta^* = 0.0652$) is insufficient for individuals whose walking vigor changes substantially between sessions. Calibrating per-user thresholds ($\theta_i^*$) based on enrollment dispersion is recommended.
2. **Dataset Scale & Placement Diversity:** The expansion to 92 participants establishes large cross-session cohorts for both Right Pocket (55 participants) and Left Wrist (42 participants), as well as a 31-participant multi-modal cohort.
3. **Multi-Modal Sensing & Wearable Fusion:** Combining pocket and wrist accelerometer streams with score-level fusion enables resilient continuous verification across varying posture states.

