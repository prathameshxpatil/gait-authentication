# Gait-Based Biometric Authentication using Dynamic Time Warping (DTW) and Adaptive Template Updating

An academic research implementation of continuous smartphone accelerometer-based biometric authentication using multivariate Dynamic Time Warping (DTW), chronological evaluation, and a candidate-buffer dual-gallery template adaptation architecture.

---

## 1. Project Overview

Biometric authentication using gait signals captured from smartphone accelerometers offers an unobtrusive, continuous verification mechanism. However, gait authentication systems face two fundamental challenges:
1. **Intra-individual temporal variance:** A user's walking dynamics drift across sessions due to fatigue, footwear, pacing, and surface changes.
2. **Template poisoning risk:** Naive online adaptation models that accept incoming samples to update biometric galleries can inadvertently incorporate impostor gait patterns, degrading system security over time.

This project implements an end-to-end research pipeline on the **200GaitData** benchmark dataset (10 participants, 20 recordings, right front trouser pocket accelerometer sampled at ~100 Hz). The system establishes a fixed multivariate DTW baseline and investigates adaptive architectures, introducing a **Candidate-Buffer Dual-Gallery** design to balance adaptation with resistance to template poisoning.

---

## 2. Pipeline Architecture & Methodology

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

## 3. Adaptation Architecture

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

## 4. Evaluation Protocol

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

## 5. Experimental Results

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

## 6. Installation & Execution

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

### 6.4 Running Automated Tests
The repository includes 38 automated unit tests covering signal filtering, discontinuity splitting, windowing, DTW dynamic programming, EER calculation, single-gallery adaptation, and dual-gallery candidate buffer mechanisms:
```bash
pytest tests/ -v
```

---

## 7. Repository Organization

```text
gait-auth/
├── README.md                                  # This documentation
├── requirements.txt                           # Minimal pinned project dependencies
├── .gitignore                                 # Git ignore configuration
├── main.py                                    # Command-line interface and pipeline entry point
├── src/                                       # Core source code package
│   ├── __init__.py                            # Minimal package initialization
│   ├── data_loader.py                         # Raw CSV discovery, filename parsing, integrity checks
│   ├── preprocessing.py                       # Butterworth filtering & discontinuity splitting
│   ├── segmentation.py                        # Sliding window extraction & GaitWindow dataclass
│   ├── dtw_authentication.py                  # Multivariate DTW kernel, Numba JIT, enrollment galleries
│   ├── evaluation.py                          # FAR/FRR/EER metrics, ROC, score distributions
│   ├── experiments.py                         # Baseline parameter exploration suite
│   ├── adaptive_template.py                   # Single-gallery adaptive benchmark implementation
│   └── dual_gallery.py                        # Candidate-Buffer Dual-Gallery architecture implementation
├── archive/                                   # Preserved exploratory & diagnostic scripts
│   ├── prototype_adaptation.py                # Scratch prototype for User 1 adaptation
│   ├── evaluate_factors.py                    # Early feature sweep script
│   ├── test_validation_split.py               # Early split validation script
│   └── diagnostic_audit.py                    # Early pocket-tilt diagnostic script
├── tests/                                     # Automated unit test suite (38 passing tests)
│   ├── test_pipeline.py                       # Baseline preprocessing, DTW, and EER tests
│   ├── test_adaptive.py                       # Single-gallery adaptation & 3-conf gate tests
│   └── test_dual_gallery.py                   # Candidate buffer, max pairwise check & dual-gallery tests
├── notebooks/                                 # Exploratory notebooks
│   └── dataset_exploration.ipynb              # Phase 1 exploratory data analysis
├── data/                                      # Gait dataset directory
│   ├── raw/                                   # 20 raw CSV recordings (10 participants, 2 sessions)
│   └── processed/                             # 41 preprocessed continuous gait segments
└── results/                                   # Experimental outputs, data tables, and figures
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

## 8. Research Limitations & Future Directions

1. **Universal vs User-Specific Operating Thresholds:** A single global scalar threshold ($\theta^* = 0.0652$) is insufficient for individuals whose walking vigor changes substantially between sessions. Calibrating per-user thresholds ($\theta_i^*$) based on enrollment dispersion is recommended.
2. **Dataset Scale:** The 200GaitData dataset comprises 10 healthy participants with smartphone placement restricted to the right trouser pocket. Evaluating cross-body placements (e.g. jacket pocket, bag, hand) and larger participant cohorts remains an open research avenue.
3. **Multi-Modal Sensing:** Combining accelerometer magnitude with gyroscopic rotational rate and step-interval temporal features may provide greater discrimination during vigorous walking pace transitions.

