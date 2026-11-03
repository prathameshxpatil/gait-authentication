# Adaptive Stream Validation Report: Separated Operational & Exhaustive Evaluation

**Cohort:** 55 Enrolled Participants (RP Session 1 Enrollment)  
**Validation Stream:** Session 2 Validation Partition ONLY (6,310 genuine windows)  
**Held-Out Test Partition:** 100% Isolated (6,291 windows untouched)  
**Frozen Baseline Operating Threshold:** $\theta^* = 0.133622$  
**Evaluated Random Seeds:** [42, 100, 2024]  

---

## 1. Operational Stream Performance (1:1 Ratio, Multi-Confirmation & Anchor Guards)

| Strategy | Stream Bal. Acc. (%) | Stream FAR (%) | Stream FRR (%) | Genuine Updates | Poisoned Updates | Poisoning Rate (%) | Mean Drift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Baseline** | 72.48 ± 0.09 | 26.18 ± 0.18 | 28.86 ± 0.00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0% | 0.0000 |
| **Dual-Gallery Standard** | 72.61 ± 0.14 | 25.59 ± 0.15 | 29.18 ± 0.21 | 4.0 ± 1.7 | 0.0 ± 0.0 | 0.0% | 0.0052 |
| **Dual-Gallery Conservative** | 72.62 ± 0.03 | 25.87 ± 0.08 | 28.88 ± 0.02 | 2.0 ± 1.7 | 0.0 ± 0.0 | 0.0% | 0.0022 |

---

## 2. Exhaustive Verification Performance (All 347,050 Genuine & Impostor Pairs)

| Strategy | Exhaustive Bal. Acc. (%) | Exhaustive FAR (%) | Exhaustive FRR (%) | Macro Bal. Acc. (%) | Macro FAR (%) | Macro FRR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Baseline** | 71.14 ± 0.00 | 28.87 ± 0.00 | 28.86 ± 0.00 | 71.47 ± 0.00 | 28.86 ± 0.00 | 28.20 ± 0.00 |
| **Dual-Gallery Standard** | 71.31 ± 0.35 | 27.72 ± 0.29 | 29.67 ± 0.67 | 71.62 ± 0.35 | 27.72 ± 0.29 | 29.05 ± 0.69 |
| **Dual-Gallery Conservative** | 71.35 ± 0.12 | 28.41 ± 0.28 | 28.90 ± 0.05 | 71.68 ± 0.12 | 28.41 ± 0.28 | 28.23 ± 0.05 |
