# Adaptation Diagnostics Report: Why Genuine Templates Are Filtered

**Cohort:** 55 Enrolled Participants (Session 1 Enrollment, Session 2 Validation Stream)  
**Operating Threshold:** $\theta^* = 0.133622$  
**Evaluated Random Seeds:** [42, 100, 2024]  
**Authoritative Sources:** 
- `results/adaptive_stream_validation/stream_validation_summary.csv`
- `results/progress_meeting/adaptation_diagnostics.csv`
- `src/adaptive_stream_validation.py`
- `src/preprocessing.py`

---

## 1. Executive Progression Table (Cohort Totals per Run)

| Pipeline Stage | Dual-Gallery Conservative | Dual-Gallery Standard | Primary Filtering Mechanism |
| :--- | :---: | :---: | :--- |
| **1. Genuine Authentication Attempts** | **6,310.0** | **6,310.0** | 100% of incoming genuine validation windows |
| **2. Authentication Rejections ($d > \theta^*$)** | 1,822.3 (28.88%) | 1,841.0 (29.18%) | Cross-session gait variability (FRR) |
| **3. Accepted Low-Confidence ($\alpha\theta^* < d \le \theta^*$)** | 3,497.3 (55.43%) | 3,058.0 (48.46%) | Valid access granted, but barred from adaptation |
| **4. Passed High-Confidence Gate ($d \le \alpha\theta^*$)** | 990.3 (15.69%) | 1,411.0 (22.36%) | High-confidence genuine windows eligible for buffer |
| **Sum of Stages 2 + 3 + 4** | **6,310.0 (100.00%)** | **6,310.0 (100.00%)** | **Exact candidate accounting reconciled** |
| **5. Overlapping Window Rejections** | 186.0 | 288.0 | 50% window overlap blocked by temporal independence guard |
| **6. Entered Candidate Buffer** | 804.3 | 1,123.0 | Independent high-confidence candidates quarantined |
| **7. Buffer Deterioration Flushes** | 763.0 flushes (810.0 cand. lost) | 1,051.0 flushes (1,129.0 cand. lost) | Intervening degraded steps/impostors flush quarantine buffer |
| **8. Candidate Clusters Ready ($K=3$)** | 7.0 clusters | 17.0 clusters | Reached required 3 consecutive confirmations |
| **9. Discarded by Cluster Consistency** | 5.0 clusters | 13.0 clusters | Max pairwise DTW $> \gamma\theta^*$ ($\gamma=0.80$ Cons. / $\gamma=0.85$ Std.) |
| **10. Discarded by Permanent Anchor Guard** | 0.0 clusters | 0.0 clusters | Top-2 permanent anchor distance $> \beta\theta^*$ ($\beta=0.85$ Cons. / $\beta=1.00$ Std.) |
| **11. Successful Genuine Promotions** | **2.0 ± 1.7** | **4.0 ± 1.7** | Promoted into active adaptive gallery ($G_{\text{adapt}}$) |
| **12. Impostor Promotions (Poisoning)** | **0.0 (0.0%)** | **0.0 (0.0%)** | **Zero poisoning admitted across all runs** |

*Candidate Accounting Reconciliation Note: In every individual seed (42, 100, 2024), the raw integer partition exactly equals 6,310 attempts without discrepancy:*
- *Seed 42: $1,821 + 3,500 + 989 = 6,310$*
- *Seed 100: $1,823 + 3,538 + 949 = 6,310$*
- *Seed 2024: $1,823 + 3,454 + 1,033 = 6,310$*  
*A previously observed 1-window deficit ($1,822 + 3,497 + 990 = 6,309$) was purely a display rounding artifact caused by rounding all three exact means ($1,822.33, 3,497.33, 990.33$) downward to the nearest integer.*

---

## 2. Parameter Configurations & Authoritative Values

| Configuration Parameter | Dual-Gallery Conservative | Dual-Gallery Standard | Source Code Anchor |
| :--- | :---: | :---: | :--- |
| **Permanent Weight ($w_{\text{perm}}$)** | **0.80** | **0.70** | `src/adaptive_stream_validation.py:420,432` |
| **High-Confidence Factor ($\alpha$)** | **0.65** ($d \le 0.0868$) | **0.70** ($d \le 0.0935$) | `src/adaptive_stream_validation.py:424,436` |
| **Cluster Consistency Limit ($\gamma$)** | **0.80** ($\max D_{ij} \le 0.1069$) | **0.85** ($\max D_{ij} \le 0.1136$) | `src/adaptive_stream_validation.py:425,437` |
| **Permanent Anchor Limit ($\beta$)** | **0.85** ($\text{Top-2 Mean} \le 0.1136$) | **1.00** ($\text{Top-2 Mean} \le 0.1336$) | `src/adaptive_stream_validation.py:426,438` |
| **Confirmed Genuine Updates** | **2.0 ± 1.7** | **4.0 ± 1.7** | `results/adaptive_stream_validation/stream_validation_summary.csv` |
| **Poisoned Updates** | **0.0 ± 0.0 (0.0%)** | **0.0 ± 0.0 (0.0%)** | `results/adaptive_stream_validation/stream_validation_summary.csv` |

---

## 3. Key Findings: Why Genuine Candidates Fail Promotion

Our diagnostic breakdown reveals the precise three bottlenecks explaining why genuine updates are rare:

### Bottleneck 1: Strict High-Confidence Thresholding (Stage 2 Gate)
- Under Conservative settings ($\alpha=0.65$), the high-confidence threshold is $0.65 \times 0.1336 = 0.0868$. Out of 6,310 genuine attempts, only **990.3 (15.69%)** qualify as high-confidence.
- A massive **3,497.3 (55.43%)** genuine windows are accepted for authentication ($d \le \theta^*$) but fall in the lower-confidence zone ($0.0868 < d \le 0.1336$), barring them from entering the candidate buffer.

### Bottleneck 2: Quarantine Deterioration Flushes (Stage 4 Buffer Retention)
- Whenever ANY authentication attempt (whether a degraded genuine step or an impostor attack) has $d > \alpha\theta^*$, the candidate buffer is instantly cleared to eliminate unverified candidates.
- In Conservative mode, **763.0 buffer flushes** occurred across the cohort, discarding **810.0 genuine quarantined candidates** before they could accumulate the required $K=3$ confirmations.

### Bottleneck 3: Multi-Confirmation & Anchor Guarding (Stages 5 & 6)
- Of the clusters that successfully reached $K=3$ candidates (7.0 clusters), the pairwise consistency check ($\gamma=0.80$) discarded 5.0 clusters, and the permanent anchor guard discarded 0.0 clusters.
- The remaining clusters successfully promoted genuine medoids into the adaptive gallery, resulting in **2.0 ± 1.7 genuine updates** (Conservative) and **4.0 ± 1.7 genuine updates** (Standard) with **zero poisoned updates**.

### Conclusion for Meeting with Professor:
The low template update count is **not an implementation defect or random failure**. It is the intentional mathematical consequence of a multi-tiered security defense designed to enforce strict conservative admission. While it maintains 100% defense against poisoning (0.0% poisoning rate across all runs), the primary operational bottleneck is that genuine gait steps that are accepted but moderately degraded frequently flush the quarantine buffer before 3 consecutive high-confidence steps can accumulate.
