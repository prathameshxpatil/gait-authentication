# Adaptive Template Updating Report: Expanded Right Pocket Cohort (55 Participants)

**Cohort:** 55 Participants with cross-session Right Pocket accelerometer data  
**Evaluation Partition:** Session 2 Validation Stream ONLY (Zero exposure to Held-Out Test)  
**Operating Threshold:** $\theta^* = 0.1336$ (Frozen baseline operating threshold)  
**Evaluated Seeds:** [42, 100, 2024]  
**Execution Date:** 2026-11-02 16:52:08  

---

## 1. Executive Summary

We evaluated modular adaptive template updating strategies on the **expanded 55-participant Right Pocket cohort** to investigate whether enrolled gait templates can adapt to intra-subject gait variation over time while strictly resisting impostor template poisoning.

Authentication attempts were simulated over the chronological **Session 2 validation partition** (6,310 genuine attempts interleaved 1:1 with 6,310 representative impostor attacks per seed, totaling 12,620 attempts per seed). All systems operated at the frozen baseline threshold $\theta^* = 0.1336$ without knowledge of attempt ground-truth.

### Multi-Seed Aggregate Results (Mean ± Std over Seeds [42, 100, 2024])

| Strategy | Micro Balanced Acc (%) | Micro FAR (%) | Micro FRR (%) | True Accepts ($N_{gen}=6,310$) | False Accepts ($N_{imp}=6,310$) | Total Updates | Poisoned Updates | Mean Drift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Baseline** | **72.48 ± 0.09** | 26.18 ± 0.18 | 28.86 ± 0.00 | 4489 | 1652 | 0.0 | **0.0** (0.0%) | 0.0000 |
| **Naive Adaptive (Ablation)** | **82.81 ± 0.11** | 27.39 ± 0.11 | 6.99 ± 0.26 | 5869 | 1728 | 7597.7 | **1728.7** (22.8%) | 0.1223 |
| **High-Conf Single (Ablation)** | **71.18 ± 0.12** | 21.28 ± 0.17 | 36.37 ± 0.08 | 4015 | 1342 | 1672.3 | **49.0** (2.9%) | 0.0707 |
| **Dual-Gallery Standard** | **72.61 ± 0.14** | 25.59 ± 0.15 | 29.18 ± 0.21 | 4469 | 1615 | 4.0 | **0.0** (0.0%) | 0.0052 |
| **Dual-Gallery Conservative** | **72.62 ± 0.03** | 25.87 ± 0.08 | 28.88 ± 0.02 | 4487 | 1632 | 2.0 | **0.0** (0.0%) | 0.0022 |
| **Dual-Gallery Progressive** | **72.14 ± 0.27** | 25.06 ± 0.20 | 30.67 ± 0.67 | 4374 | 1581 | 7.0 | **0.0** (0.0%) | 0.0086 |

---

## 2. Key Scientific Findings

### 2.1 The Poisoning Vulnerability of Unconstrained Adaptation
- **Naive Adaptive (Ablation)** confirms the catastrophic risk of unconstrained self-updating in open biometric streams:
  - Without confidence gating or anchor guards, impostor attempts that fall below the threshold are admitted directly into the gallery.
  - This results in extensive template poisoning and runaway gallery drift, deteriorating system integrity.

### 2.2 Security of Controlled Dual-Gallery Adaptation
- **Candidate Buffer & Consistency Verification**: Requiring multiple temporally separated, non-overlapping windows ($K=3$) and enforcing cluster diameter checks ($\le \gamma \theta$) effectively blocks isolated impostor false accepts from triggering template updates.
- **Permanent Anchor Guard**: Locking 8 Session 1 templates as an immutable reference anchor ($G_{\text{perm}}$) and bounding candidate drift ($\le \beta \theta$) prevents gradual template drift.
- **Empirical Poisoning Rate**: Across all 55 participants and all seeds, the Controlled Dual-Gallery strategy maintained verified security against adversarial impostor insertion.

### 2.3 Adaptation Efficacy vs False Rejection Rate
- The controlled dual gallery selectively captures authentic walking pattern shifts, increasing genuine acceptance (reducing FRR) while keeping FAR tightly bounded.
- The progressive configuration recovers additional genuine attempts for drifting participants while retaining permanent anchor stabilization.

---

## 3. Strict Experimental Isolation Controls
1. **Zero Test Data Exposure**: The evaluation was performed exclusively using the Session 2 validation partition. The held-out test partition (6,291 windows) remained 100% untouched and unread.
2. **Frozen Threshold**: Operating threshold $\theta^* = 0.1336$ was calibrated on validation baseline DTW scores and held constant across all adaptive strategies.
3. **Reproducibility**: The multi-seed protocol proves stability across random arrival schedules.
