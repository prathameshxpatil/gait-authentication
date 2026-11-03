# Fair Exhaustive Comparison: Fixed Baseline vs Dual-Gallery Adaptation

**Cohort:** 55 Enrolled Participants (Right Pocket Collection)  
**Evaluation Partition:** Session 2 Validation Stream ONLY (Held-Out Test Partition 100% Isolated)  
**Evaluation Protocol:** Exhaustive Matching (Every validation window tested against all 55 galleries: 6,310 genuine + 340,740 impostor = 347,050 trials per run)  
**Frozen Operating Threshold:** $\theta^* = 0.1336$  
**Evaluated Seeds:** [42, 100, 2024]  
**Date:** 2026-11-02 19:15:50  

---

## 1. Executive Summary & Aggregate Benchmark

| Evaluation Metric | Fixed Baseline | Dual-Gallery Standard | Dual-Gallery Conservative |
| :--- | :---: | :---: | :---: |
| **Micro Balanced Accuracy (%)** | **71.14 ± 0.00** | **71.12 ± 0.03** | **71.14 ± 0.00** |
| **Micro FAR (%)** | 28.87 ± 0.00 | 28.71 ± 0.21 | 28.87 ± 0.00 |
| **Micro FRR (%)** | 28.86 ± 0.00 | 29.05 ± 0.20 | 28.86 ± 0.00 |
| **Macro Balanced Accuracy (%)** | 71.47 ± 0.00 | 71.45 ± 0.03 | 71.47 ± 0.00 |
| **Macro FAR (%)** | 28.86 ± 0.00 | 28.70 ± 0.21 | 28.86 ± 0.00 |
| **Macro FRR (%)** | 28.20 ± 0.00 | 28.40 ± 0.21 | 28.20 ± 0.00 |
| **Genuine Accepts ($N_{gen}=6,310$)** | 4489 | 4477 | 4489 |
| **False Accepts ($N_{imp}=340,740$)**| 98358 | 97822 | 98358 |
| **Total Accepted Updates** | 0.0 | 0.7 | 0.0 |
| **Poisoned Updates** | **0.0 (0.0%)** | **0.7 (66.7%)** | **0.0 (0.0%)** |
| **Mean Gallery Drift** | 0.0000 | 0.0010 | 0.0000 |

---

## 2. Methodological Differences: Exhaustive Evaluation vs 1:1 Representative Experiment

| Protocol Aspect | Earlier 1:1 Representative Experiment | Fair Exhaustive Evaluation |
| :--- | :--- | :--- |
| **Impostor Sampling** | 1:1 balanced sampling ($N_{imp} = N_{gen} \approx 115$ per user). | **Exhaustive Matching:** Every non-owner window ($N_{imp} = 6,310 - N_i \approx 6,195$ per user). |
| **Total Trials per Seed** | 12,620 authentication trials ($6,310 \text{ gen} + 6,310 \text{ imp}$). | **347,050 authentication trials** ($6,310 \text{ gen} + 340,740 \text{ imp}$). |
| **Impostor Prevalence** | 50.0% ($1$ impostor attempt per $1$ genuine attempt). | **98.2%** ($54$ impostor attempts per $1$ genuine attempt). |
| **Adversarial Stress** | Low-density intrusion; long periods of pure genuine walking. | **High-density adversarial bombardment** (54× higher attack frequency). |
| **Buffer Disruption Pressure** | Infrequent buffer quarantine events. | Degraded impostor attempts arrive rapidly, continuously testing quarantine resilience. |
| **Baseline Comparability** | Baseline evaluated on same 1:1 stream (FAR = 26.18%). | Baseline evaluates all 340,740 non-owner trials, **matching the exact validation baseline denominator** (FAR = 28.87%). |
