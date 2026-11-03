"""
src/generate_architecture_diagram.py — System Architecture Diagram Generator.

Generates a publication-grade architecture diagram showing the end-to-end
dual-gallery adaptive gait authentication pipeline, explicitly highlighting:
  1. Immutable Permanent Gallery (G_perm)
  2. Dynamic Adaptive Gallery (G_adapt)
  3. Preprocessing and DTW matching
  4. Authentication decision
  5. Multi-tier security gates (High-confidence, Temporal independence,
     Candidate buffer, Cluster consistency, Permanent anchor guard, Redundancy replacement)

Output saved to:
  results/progress_meeting/architecture_diagram.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle


def draw_architecture_diagram(output_path: str = "results/progress_meeting/architecture_diagram.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 11), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Background subtle styling
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Helper function for fancy boxes
    def add_box(xy, width, height, title, subtitle="", box_color="#ffffff", border_color="#2c3e50", title_color="#2c3e50", rx=0.15):
        box = FancyBboxPatch(
            xy, width, height,
            boxstyle=f"round,pad=0.08,rounding_size={rx}",
            facecolor=box_color,
            edgecolor=border_color,
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(box)
        cx = xy[0] + width / 2.0
        cy = xy[1] + height / 2.0
        if subtitle:
            ax.text(cx, cy + height * 0.18, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=title_color, zorder=3)
            ax.text(cx, cy - height * 0.18, subtitle, ha="center", va="center", fontsize=8.5, color="#555555", zorder=3)
        else:
            ax.text(cx, cy, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=title_color, zorder=3)
        return (cx, cy)

    def draw_arrow(start, end, text="", color="#2c3e50", style="-|>", lw=1.6, connectionstyle=None):
        props = dict(arrowstyle=style, color=color, lw=lw)
        if connectionstyle:
            props["connectionstyle"] = connectionstyle
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=props,
            zorder=4,
        )
        if text:
            mx = (start[0] + end[0]) / 2.0
            my = (start[1] + end[1]) / 2.0
            ax.text(mx, my + 0.15, text, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color, zorder=5)

    # -----------------------------------------------------------------------
    # Main Title & Subtitle Header
    # -----------------------------------------------------------------------
    ax.text(8.0, 10.5, "Dual-Gallery Adaptive Gait Authentication Architecture", ha="center", va="center", fontsize=16, fontweight="bold", color="#1a252f")
    ax.text(8.0, 10.15, "Biometric Template Adaptation with Multi-Tier Poisoning Defense & Permanent Anchor Protection", ha="center", va="center", fontsize=10.5, color="#5a6268")

    # -----------------------------------------------------------------------
    # Region 1: Signal Ingestion & Feature Preprocessing (Left Column)
    # -----------------------------------------------------------------------
    add_box((0.6, 7.8), 2.8, 1.2, "Raw Accelerometer Window", "Tri-axial (X, Y, Z)\nSize: 400 samples (4.0s @ 100Hz)", box_color="#ebf5fb", border_color="#2980b9", title_color="#1b4f72")
    draw_arrow((2.0, 7.8), (2.0, 6.7), "Extract")

    add_box((0.6, 5.5), 2.8, 1.2, "Preprocessing & Features", "Butterworth Low-Pass (20 Hz, Zero-Phase)\nZ-Normalized Magnitude", box_color="#ebf5fb", border_color="#2980b9", title_color="#1b4f72")
    draw_arrow((3.4, 6.1), (4.5, 6.1), "Input Window W")

    # -----------------------------------------------------------------------
    # Region 2: Dual-Gallery Matching Engine (Upper Center)
    # -----------------------------------------------------------------------
    # Big Container for Dual-Gallery
    outer_gallery = FancyBboxPatch((4.5, 5.0), 6.8, 4.4, boxstyle="round,pad=0.15,rounding_size=0.25", facecolor="#ffffff", edgecolor="#34495e", linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(outer_gallery)
    ax.text(7.9, 9.15, "DUAL-GALLERY MATCHING ENGINE", ha="center", va="center", fontsize=11, fontweight="bold", color="#2c3e50")

    # Sub-Box A: Permanent Gallery (IMMUTABLE)
    add_box((4.8, 7.4), 2.8, 1.4, "Permanent Gallery (G_perm)", "8 S1 Stratified Templates\n[STRICTLY IMMUTABLE]\nC-level Read-Only Arrays", box_color="#e8f8f5", border_color="#16a085", title_color="#0e6252")

    # Sub-Box B: Adaptive Gallery (DYNAMIC)
    add_box((8.2, 7.4), 2.8, 1.4, "Adaptive Gallery (G_adapt)", "Up to M=4 Templates\n[DYNAMIC BOUNDED]\nVerified Updates Only", box_color="#fef9e7", border_color="#f39c12", title_color="#7d6608")

    # Matching & Fusion Box
    add_box((5.5, 5.3), 4.8, 1.4, "Multivariate DTW Matching & Fusion", "d_perm = min DTW(W, G_perm)\nd_adapt = min DTW(W, G_adapt)\nd_comb = 0.80*d_perm + 0.20*d_adapt", box_color="#f4f6f7", border_color="#7f8c8d", title_color="#2c3e50")

    draw_arrow((6.2, 7.4), (6.2, 6.7), "d_perm", color="#16a085")
    draw_arrow((9.6, 7.4), (9.6, 6.7), "d_adapt", color="#f39c12")

    # -----------------------------------------------------------------------
    # Region 3: Primary Authentication Decision (Right Column)
    # -----------------------------------------------------------------------
    draw_arrow((10.3, 6.0), (12.2, 6.0), "Combined Score d_comb")

    # Diamond / Decision Box
    add_box((12.2, 5.2), 3.2, 1.6, "Authentication Gate", "Score d_comb <= θ* ?\n(Baseline θ* = 0.1336)", box_color="#eaf2f8", border_color="#2980b9", title_color="#1b4f72")

    # Outputs of Auth Gate
    add_box((12.4, 7.5), 2.8, 1.0, "ACCESS DENIED", "Impostor or Degraded Gait\nCandidate Buffer Cleared", box_color="#fadbd8", border_color="#c0392b", title_color="#922b21")
    draw_arrow((13.8, 6.8), (13.8, 7.5), "No (d > θ*)", color="#c0392b")

    add_box((12.4, 3.2), 2.8, 1.0, "ACCESS GRANTED", "Valid Authentication\nUser Admitted", box_color="#d5f5e3", border_color="#27ae60", title_color="#196f3d")
    draw_arrow((13.8, 5.2), (13.8, 4.2), "Yes (d <= θ*)", color="#27ae60")

    # -----------------------------------------------------------------------
    # Region 4: Multi-Tier Adaptation Pipeline (Bottom Container)
    # -----------------------------------------------------------------------
    outer_adapt = FancyBboxPatch((0.6, 0.5), 14.8, 2.3), # wait, coordinate tuple
    outer_adapt = FancyBboxPatch((0.6, 0.4), 14.8, 2.4, boxstyle="round,pad=0.15,rounding_size=0.25", facecolor="#ffffff", edgecolor="#8e44ad", linewidth=2.0, linestyle="-.", zorder=1)
    ax.add_patch(outer_adapt)
    ax.text(8.0, 2.55, "MULTI-TIER ADAPTIVE SECURITY & PROMOTION PIPELINE", ha="center", va="center", fontsize=11, fontweight="bold", color="#8e44ad")

    # Path from Access Granted to High-Confidence Filter
    draw_arrow((12.4, 3.7), (1.8, 1.9), "Evaluate Adaptation", color="#8e44ad", connectionstyle="arc3,rad=-0.25")

    # Tier 1: High-Confidence Admission Gate
    add_box((0.9, 0.7), 2.3, 1.4, "Tier 1: High-Conf Gate", "Score <= α * θ*\n(α = 0.65 / 0.70)\nFlushes if Degraded", box_color="#f5eef8", border_color="#8e44ad", title_color="#512e5f")

    # Tier 2: Candidate Buffer & Temporal Check
    draw_arrow((3.2, 1.4), (4.0, 1.4), "High Conf")
    add_box((4.0, 0.7), 2.5, 1.4, "Tier 2: Candidate Buffer", "Non-Overlapping Check\nFIFO Age Expiration\nTarget Size: K = 3", box_color="#f5eef8", border_color="#8e44ad", title_color="#512e5f")

    # Tier 3: Pairwise Cluster Consistency Guard
    draw_arrow((6.5, 1.4), (7.3, 1.4), "Buffer Ready")
    add_box((7.3, 0.7), 2.5, 1.4, "Tier 3: Cluster Consistency", "Max Pairwise DTW\n<= γ * θ* (γ = 0.80 / 0.85)\nRejects Incoherent Clusters", box_color="#f5eef8", border_color="#8e44ad", title_color="#512e5f")

    # Tier 4: Permanent Anchor Guard
    draw_arrow((9.8, 1.4), (10.6, 1.4), "Consistent")
    add_box((10.6, 0.7), 2.5, 1.4, "Tier 4: Permanent Anchor", "Top-2 Anchor Mean\n<= β * θ* (β = 0.85 / 1.00)\nPrevents Catastrophic Drift", box_color="#f5eef8", border_color="#8e44ad", title_color="#512e5f")

    # Tier 5: Redundancy Promotion
    draw_arrow((13.1, 1.4), (13.7, 1.4), "Verified")
    add_box((13.7, 0.7), 1.5, 1.4, "Promote", "Medoid to\nG_adapt\n(Capacity M=4)", box_color="#d4efdf", border_color="#27ae60", title_color="#196f3d")

    # Feedback loop arrow from Promotion up into Adaptive Gallery
    draw_arrow((14.5, 2.1), (11.0, 8.1), "Update G_adapt ONLY\n(G_perm strictly untouched)", color="#27ae60", lw=2.2, connectionstyle="arc3,rad=-0.30")

    # Security callout badge
    badge = FancyBboxPatch((4.7, 4.4), 6.4, 0.5),
    badge_box = FancyBboxPatch((4.7, 4.4), 6.4, 0.5, boxstyle="round,pad=0.05,rounding_size=0.1", facecolor="#e8f8f5", edgecolor="#16a085", linewidth=1.2, zorder=3)
    ax.add_patch(badge_box)
    ax.text(7.9, 4.65, "KEY SECURITY GUARANTEE: Permanent Gallery is NEVER modified | Updates require K=3 consecutive confirmations", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#0e6252", zorder=4)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved system architecture diagram to: {output_path}")


if __name__ == "__main__":
    draw_architecture_diagram()

