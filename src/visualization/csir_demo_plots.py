"""Publication-grade visualization suite for CSIR-CSIO meeting & verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _save_fig(fig: plt.Figure, out_path: str | Path, dpi: int = 180) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dataset_15scans_overview(records: List[Any], out_path: str | Path) -> None:
    """Plot a 3x5 gallery of all 15 CSV matrices in Polymer sample data."""
    n = len(records)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10.5), constrained_layout=True)
    axes_flat = axes.ravel()

    for idx, rec in enumerate(records):
        ax = axes_flat[idx]
        arr = rec.array
        p2, p98 = np.percentile(arr, [2.0, 98.0])
        if p98 <= p2:
            p2, p98 = float(arr.min()), float(arr.max()) + 1e-5
        im = ax.imshow(arr, cmap="turbo", vmin=p2, vmax=p98, aspect="equal")
        cohort_tag = "Cohort A" if "cohort_a" in rec.cohort else "Cohort B"
        ax.set_title(
            f"{rec.stem}\n{cohort_tag} | {rec.shape[0]}x{rec.shape[1]} | {rec.frequency_ghz:.0f} GHz",
            fontsize=8.5,
            fontweight="bold",
        )
        ax.set_xlabel("X (px)", fontsize=7.5)
        ax.set_ylabel("Y (px)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=6.5)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "CSIR-CSIO Raw THz CSV Dataset Audit (All 15 Scans in 'Polymer sample data/')",
        fontsize=13,
        fontweight="bold",
    )
    _save_fig(fig, out_path)


def plot_preprocessing_and_hotstrip(samples: List[Dict[str, Any]], out_path: str | Path) -> None:
    """Visualize 5-stage preprocessing on representative Cohort A & B scans."""
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(17.5, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "1. Raw THz CSV (Unfiltered)",
        "2. Hot-Strip Masked + Norm [Ch 0]",
        "3. Abs Local DoG Anomaly [Ch 1]",
        "4. Sobel Edge Magnitude [Ch 2]",
        "5. Reference Defect Mask",
    ]
    for r_idx, item in enumerate(samples):
        raw = item["raw"]
        tensor = item["tensor"]  # (3, H, W)
        mask = item["mask"]
        label = item["label"]

        panels = [
            (raw, "turbo", None, None),
            (tensor[0], "turbo", 0.0, 1.0),
            (tensor[1], "inferno", 0.0, 1.0),
            (tensor[2], "magma", 0.0, 1.0),
            (mask, "gray", 0.0, 1.0),
        ]
        for c_idx, (img, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r_idx, c_idx]
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            if c_idx == 4:
                ax.contour(mask, levels=[0.5], colors=["#00FF7F"], linewidths=1.2)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=9.5, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Signal Conditioning Pipeline: Suppressing Border Hot Strips & Exposing Dip/Peak Defects",
        fontsize=12.5,
        fontweight="bold",
    )
    _save_fig(fig, out_path)


def plot_deeplabv3plus_rgb_demo(
    rgb_img: np.ndarray,
    gt_mask: np.ndarray,
    intermediates: Dict[str, Any],
    out_path: str | Path,
) -> None:
    """Create CSIR meeting explainer figure showing DeepLabV3+ blocks on an RGB specimen."""
    fig = plt.figure(figsize=(18, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 5)

    # Row 0: Input, GT overlay, Encoder Low-Level 1/4, Encoder High 1/16, Decoder Output
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_img)
    ax0.set_title("1. RGB Polymer Specimen\n(Surface + Blind + Buried Defects)", fontsize=9.5, fontweight="bold")
    ax0.axis("off")

    low_feat = intermediates["low_level_feat"][0].detach().cpu().numpy().mean(axis=0)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(low_feat, cmap="viridis")
    ax1.set_title(f"2. Encoder Low-Level (1/4)\nShape: {low_feat.shape[0]}x{low_feat.shape[1]} (Edges)", fontsize=9.5, fontweight="bold")
    ax1.axis("off")

    high_feat = intermediates["encoder_high_feat"][0].detach().cpu().numpy().mean(axis=0)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(high_feat, cmap="plasma")
    ax2.set_title(f"3. Encoder Bottleneck (1/16)\nShape: {high_feat.shape[0]}x{high_feat.shape[1]} (Semantic)", fontsize=9.5, fontweight="bold")
    ax2.axis("off")

    dec_feat = intermediates["decoder_fused"][0].detach().cpu().numpy().mean(axis=0)
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(dec_feat, cmap="magma")
    ax3.set_title(f"4. Decoder Fusion (ASPP + 1/4)\nShape: {dec_feat.shape[0]}x{dec_feat.shape[1]}", fontsize=9.5, fontweight="bold")
    ax3.axis("off")

    probs = intermediates["probs"][0, 0].detach().cpu().numpy()
    ax4 = fig.add_subplot(gs[0, 4])
    ax4.imshow(rgb_img)
    ax4.imshow(probs, cmap="jet", alpha=0.52, vmin=0.0, vmax=1.0)
    ax4.contour(probs >= 0.5, levels=[0.5], colors=["#00FF7F"], linewidths=1.8)
    ax4.contour(gt_mask >= 0.5, levels=[0.5], colors=["white"], linestyles="--", linewidths=1.2)
    ax4.set_title("5. Final Pixel Segmentation\n(Green: Pred, Dashed: Ground Truth)", fontsize=9.5, fontweight="bold")
    ax4.axis("off")

    # Row 1: All 5 ASPP parallel branches
    branches = intermediates["aspp_branches"]
    branch_titles = {
        "1x1": "ASPP Branch 1: 1x1 Conv\n(Pointwise Features)",
        "rate_6": "ASPP Branch 2: Dilated r=6\n(Local Defect Scale)",
        "rate_12": "ASPP Branch 3: Dilated r=12\n(Medium Void / Seam Scale)",
        "rate_18": "ASPP Branch 4: Dilated r=18\n(Broad Plate Context)",
        "image_pool": "ASPP Branch 5: Image Pooling\n(Global Context Vector)",
    }
    for col_idx, (key, b_tensor) in enumerate(branches.items()):
        ax_b = fig.add_subplot(gs[1, col_idx])
        b_map = b_tensor[0].detach().cpu().numpy().mean(axis=0)
        ax_b.imshow(b_map, cmap="inferno")
        ax_b.set_title(branch_titles.get(key, f"ASPP {key}"), fontsize=9.0, fontweight="bold")
        ax_b.axis("off")

    fig.suptitle(
        "DeepLabV3+ Architecture Walkthrough on RGB Polymer Specimen (ResNet-34 + ASPP r=[6, 12, 18] + Low-Level Decoder)",
        fontsize=13,
        fontweight="bold",
    )
    _save_fig(fig, out_path)


def plot_thz_deeplabv3plus_activations(
    activation_items: List[Dict[str, Any]],
    out_path: str | Path,
) -> None:
    """Show THz-adapted DeepLabV3+ (128x128, OS=8, r=1,2,4) internal activations on real THz scans."""
    n = len(activation_items)
    fig, axes = plt.subplots(n, 6, figsize=(18.5, 3.4 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = [
        "Input THz (Ch0 Norm)",
        "Low-Level 1/4 (32x32)",
        "ASPP r=1 (Local Spot)",
        "ASPP r=2 (Beam Waist)",
        "ASPP r=4 (Array Context)",
        "Decoder Pred + Contour",
    ]
    for r_idx, item in enumerate(activation_items):
        inter = item["intermediates"]
        x_in = item["input_ch0"]
        probs = inter["probs"][0, 0].detach().cpu().numpy()
        low = inter["low_level_feat"][0].detach().cpu().numpy().mean(axis=0)
        br = inter["aspp_branches"]
        r1 = br["rate_1"][0].detach().cpu().numpy().mean(axis=0)
        r2 = br["rate_2"][0].detach().cpu().numpy().mean(axis=0)
        r4 = br["rate_4"][0].detach().cpu().numpy().mean(axis=0)

        panels = [
            (x_in, "turbo"),
            (low, "viridis"),
            (r1, "inferno"),
            (r2, "inferno"),
            (r4, "inferno"),
            (probs, "hot"),
        ]
        for c_idx, (img, cmap) in enumerate(panels):
            ax = axes[r_idx, c_idx]
            if c_idx == 5:
                ax.imshow(x_in, cmap="gray")
                ax.imshow(probs, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0)
                ax.contour(probs >= 0.5, levels=[0.5], colors=["#00FF7F"], linewidths=1.5)
            else:
                ax.imshow(img, cmap=cmap)
            if r_idx == 0:
                ax.set_title(headers[c_idx], fontsize=9.2, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(item["label"], fontsize=8.8, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "THz-Adapted DeepLabV3+ Internal Feature Maps (128x128 Upsample, ResNet-18 OS=8, ASPP r=[1,2,4])",
        fontsize=12.5,
        fontweight="bold",
    )
    _save_fig(fig, out_path)


def plot_thz_localization_gallery(
    eval_items: List[Dict[str, Any]],
    out_path: str | Path,
) -> None:
    """Plot end-to-end localization results and OpenCV metrology overlays across THz scans."""
    n = len(eval_items)
    fig, axes = plt.subplots(n, 4, figsize=(15.5, 3.1 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Raw THz CSV",
        "Physics Anomaly [Ch 1 DoG]",
        "THz-DeepLabV3+ Probability",
        "Metrology Overlay (Centroids & Diam)",
    ]
    for r_idx, item in enumerate(eval_items):
        raw = item["raw"]
        dog = item["dog_ch1"]
        probs = item["prob_native"]
        binary = item["binary_native"]
        defects = item["defects"]
        title = item["label"]

        # Col 0: Raw CSV
        ax0 = axes[r_idx, 0]
        p2, p98 = np.percentile(raw, [2.0, 98.0])
        ax0.imshow(raw, cmap="turbo", vmin=p2, vmax=max(p98, p2 + 1e-4))
        ax0.set_ylabel(title, fontsize=8.5, fontweight="bold")

        # Col 1: DoG
        ax1 = axes[r_idx, 1]
        ax1.imshow(dog, cmap="inferno", vmin=0.0, vmax=1.0)

        # Col 2: Probabilities
        ax2 = axes[r_idx, 2]
        im2 = ax2.imshow(probs, cmap="magma", vmin=0.0, vmax=1.0)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)

        # Col 3: Overlay + centroids
        ax3 = axes[r_idx, 3]
        ax3.imshow(item["norm_ch0"], cmap="gray")
        ax3.imshow(probs, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
        if binary.max() > 0:
            ax3.contour(binary, levels=[0.5], colors=["#00FF7F"], linewidths=1.5)
        for d in defects[:5]:
            cx = d["centroid_px"]["x"]
            cy = d["centroid_px"]["y"]
            ax3.plot(cx, cy, marker="+", color="white", markersize=8, markeredgewidth=1.5)
            ax3.text(
                cx + 1.2,
                cy - 1.2,
                f"d={d['beam_corrected_diameter_mm']:.1f}mm",
                color="yellow",
                fontsize=6.8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.6, lw=0),
            )
        status = "PASS (0 defects)" if len(defects) == 0 else f"FAIL ({len(defects)} defect{'s' if len(defects) > 1 else ''})"
        ax3.set_xlabel(status, fontsize=8, fontweight="bold", color=("green" if len(defects) == 0 else "crimson"))

        for c_idx in range(4):
            if r_idx == 0:
                axes[r_idx, c_idx].set_title(col_titles[c_idx], fontsize=9.5, fontweight="bold")
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])

    fig.suptitle(
        "End-to-End THz-DeepLabV3+ Subsurface Defect Localization & OpenCV Metrology",
        fontsize=12.5,
        fontweight="bold",
    )
    _save_fig(fig, out_path)


def plot_model1_separability(
    cohort_b_rows: List[Dict[str, Any]],
    out_path: str | Path,
) -> None:
    """Plot material (100% infill) and infill (15%-100%) separability with frequency annotations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)

    # Panel 1: 100% Infill across 4 polymers
    mat_rows = [r for r in cohort_b_rows if abs(r["infill_pct"] - 100.0) < 1e-3]
    mat_names = [f"{r['material']}\n({r['frequency_ghz']:.0f} GHz)" for r in mat_rows]
    medians = [r["core_median"] for r in mat_rows]
    iqrs = [r["core_iqr"] for r in mat_rows]
    colors = ["#2b8cbe" if abs(r["frequency_ghz"] - 400.0) < 1.0 else "#e34a33" for r in mat_rows]

    bars = ax1.bar(mat_names, medians, yerr=[q / 2.0 for q in iqrs], capsize=5, color=colors, edgecolor="black", alpha=0.85)
    for b_patch, val in zip(bars, medians):
        ax1.text(b_patch.get_x() + b_patch.get_width() / 2.0, val + 0.004, f"{val:+.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax1.set_title("Model 1 Audit: 100% Infill Material Interior Medians\n(Red = Frequency Confounder at 450 GHz vs 400 GHz)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Interior Median Signal (+/- IQR/2)", fontsize=9.5)
    ax1.grid(axis="y", alpha=0.25)

    # Panel 2: Infill progression for PLA and PETG
    for mat, marker, col in [("PLA", "o", "#1b9e77"), ("PETG", "s", "#d95f02")]:
        sub = sorted([r for r in cohort_b_rows if r["material"] == mat], key=lambda d: d["infill_pct"])
        xs = [r["infill_pct"] for r in sub]
        ys = [r["core_median"] for r in sub]
        ax2.plot(xs, ys, marker=marker, color=col, linewidth=2.0, markersize=7, label=f"{mat}")
        for r in sub:
            ax2.annotate(
                f"{r['frequency_ghz']:.0f}G",
                (r["infill_pct"], r["core_median"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7.5,
                color=col,
            )
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax2.set_title("Model 1 Audit: Infill % vs Interior Signal\n(Labels Show Acquisition Frequency in GHz)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Infill Percentage (%)", fontsize=9.5)
    ax2.set_ylabel("Interior Median Signal", fontsize=9.5)
    ax2.set_xticks([15, 45, 75, 100])
    ax2.legend()
    ax2.grid(alpha=0.25)

    fig.suptitle(
        "Model 1 Global Characterizer Feasibility Audit on Cohort B Scans (Why Fixed-Frequency CSIR Scans Are Essential)",
        fontsize=12,
        fontweight="bold",
    )
    _save_fig(fig, out_path)
