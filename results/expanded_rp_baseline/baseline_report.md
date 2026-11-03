# Baseline Authentication Report: Expanded Right Pocket Cohort (55 Participants)

**Cohort:** 55 Participants with cross-session Right Pocket accelerometer data  
**Sensor Placement:** Right front trouser pocket  
**Execution Date:** 2026-11-02 12:34:22  
**Evaluation Mode:** Fixed-template multivariate Dynamic Time Warping (DTW) — Zero Adaptation  

---

## 1. Executive Summary

A fixed-template multivariate DTW biometric baseline was established across **55 participants** from the expanded 200GaitData Right Pocket collection. The evaluation strictly enforces zero boundary leakage and uses **cohort-calibrated threshold selection using validation scores from the 55 enrolled participants**.

### Aggregate Benchmark Results

| Evaluation Level | Metric | Value | Attempt Counts |
| :--- | :--- | :---: | :--- |
| **Calibration (Validation)** | Calibrated Threshold $\theta^*$ | **0.1336** | 6,310 Genuine vs 340,740 Exhaustive Impostors |
| | Validation EER | **28.86%** | Equal Error Rate operating point |
| **Test (Micro-Averaged)** | Test FAR (Micro) | **29.15%** | 99014 false accepts / 339,714 impostor attempts |
| | Test FRR (Micro) | **26.55%** | 1670 false rejects / 6,291 genuine attempts |
| | Balanced Accuracy (Micro) | **72.15%** | $100\% - (\text{FAR} + \text{FRR})/2$ |
| **Test (Macro-Averaged)** | Test FAR (Macro) | **29.14%** | Mean $\text{FAR}_i$ across 55 enrolled galleries |
| | Test FRR (Macro) | **25.92%** | Mean $\text{FRR}_i$ across 55 enrolled identities |
| | Balanced Accuracy (Macro)| **72.47%** | Mean Balanced Accuracy across 55 enrolled identities |
| **Test Diagnostic EER** | Diagnostic Test EER | **27.54%** | Post-hoc EER on held-out test distribution |

---

## 2. Cohort Accounting & Leakage Controls

### 2.1 Cohort Selection & Exclusions
- **Included (55 Participants):** PIDs [1, 2, 3, 4, 10, 13, 14, 15, 16, 18]... (55 total) having both RP Session 1 and RP Session 2.
- **Excluded (37 Participants):**
  - 24 participants missing RP Session 2 (PIDs 11, 17, 19, 35, 36, 37, 39, 40, 41, 44, 45, 47, 48, 50, 55, 57, 59, 61, 63, 66, 69, 72, 85, 93).
  - 10 participants missing RP Session 1 (PIDs 5, 6, 7, 8, 9, 12, 34, 65, 75, 92).
  - 3 participants with Left Wrist data only (PIDs 29, 70, 77).
  - Participant 73 is absent from the dataset.

### 2.2 Reconciled Window Partitioning
- **Session 1 (Enrollment):** 12,933 windows (8 stratified templates selected per participant).
- **Session 2 Validation:** 6,310 windows (first chronological 50%).
- **Boundary Overlap Discards:** 40 windows systematically discarded ($t_{\text{start, test}} \ge t_{\text{end, val}} - 10^{-6}$ s).
- **Session 2 Test:** 6,291 windows (second chronological 50%, held-out).
- **Zero-Leakage Assertion:** 100% verified. Every window belongs to exactly one partition. Zero raw accelerometer samples are shared across the validation/test boundary.

---

## 3. Threshold Calibration Methodology

The operating threshold was determined via **cohort-calibrated threshold selection using validation scores from the 55 enrolled participants**:
- Operating point: $\theta^* = 0.1336$ where Validation $\text{FAR}(\theta) = \text{FRR}(\theta) = 28.86\%$.
- The threshold was frozen prior to test evaluation. No test-set scores were used for tuning.

---

## 4. Per-Enrolled-Identity Macro Metrics

Metrics are calculated per enrolled gallery identity $i$:
$$\text{FAR}_i = \frac{\text{False Accepts against gallery } i}{N_{\text{imp}, i}}, \quad \text{FRR}_i = \frac{\text{False Rejects for identity } i}{N_{\text{gen}, i}}$$
where $N_{\text{imp}, i} = \sum_{j \neq i} N_j = N_{\text{test}} - N_{\text{gen}, i}$ (averaging $\approx 6,177$ impostor attempts per gallery; $339,714$ pooled impostor attempts in total).
