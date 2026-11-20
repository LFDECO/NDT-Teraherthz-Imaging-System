"""Self-contained 10-slide widescreen presentation PDF builder for CSIR-CSIO meeting."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


NAVY = "#0F2942"
TEAL = "#136F8A"
ACCENT = "#D95F02"
GREEN = "#1B7F4B"
SLATE = "#334155"
LIGHT_BG = "#F4F7FA"
BORDER = "#CBD5E1"


def _create_slide(
    slide_num: int,
    total_slides: int,
    title: str,
    subtitle: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    # Full-canvas coordinate axis for clean chrome & annotations
    bg_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)
    bg_ax.axis("off")

    # Header bar
    bg_ax.add_patch(mpatches.Rectangle((0.0, 0.895), 1.0, 0.105, facecolor=NAVY, edgecolor="none"))
    bg_ax.add_patch(mpatches.Rectangle((0.0, 0.890), 1.0, 0.005, facecolor=TEAL, edgecolor="none"))
    bg_ax.text(0.025, 0.955, title, color="white", fontsize=17.5, fontweight="bold", va="center")
    bg_ax.text(0.025, 0.918, subtitle, color="#BFD7ED", fontsize=10.5, va="center")
    bg_ax.text(
        0.975,
        0.942,
        f"Slide {slide_num} / {total_slides}",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=TEAL, edgecolor="none"),
    )

    # Footer bar
    bg_ax.add_patch(mpatches.Rectangle((0.0, 0.0), 1.0, 0.036, facecolor=LIGHT_BG, edgecolor="none"))
    bg_ax.text(
        0.025,
        0.017,
        "THz-AI Polymer NDT Diagnostics  |  DeepLabV3+ Architecture, Real RGB Verification & CSIR THz Findings",
        color="#64748B",
        fontsize=8.5,
        va="center",
    )
    bg_ax.text(
        0.975,
        0.017,
        "SRM University & CSIR-CSIO Research Collaboration",
        color="#64748B",
        fontsize=8.5,
        ha="right",
        va="center",
    )
    return fig, bg_ax


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body_lines: List[str],
    header_color: str = NAVY,
    fill_color: str = LIGHT_BG,
    title_size: float = 10.5,
    body_size: float = 8.8,
) -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=fill_color,
            edgecolor=BORDER,
            linewidth=1.1,
        )
    )
    hdr_h = min(0.044, h * 0.25)
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y + h - hdr_h),
            w,
            hdr_h,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor=header_color,
            edgecolor="none",
        )
    )
    ax.text(x + 0.012, y + h - hdr_h * 0.5, title, color="white", fontsize=title_size, fontweight="bold", va="center")

    cur_y = y + h - hdr_h - 0.022
    step = min(0.030, (h - hdr_h - 0.024) / max(len(body_lines), 1))
    for line in body_lines:
        ax.text(x + 0.012, cur_y, line, color=SLATE, fontsize=body_size, va="top")
        cur_y -= step


def _place_image(
    fig: plt.Figure,
    rect: List[float],
    img_path: Path,
) -> None:
    ax = fig.add_axes(rect)
    arr = np.asarray(Image.open(img_path).convert("RGB"))
    ax.imshow(arr)
    ax.axis("off")


def _draw_table(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    col_labels: List[str],
    rows: List[List[str]],
    col_widths: List[float] | None = None,
    font_size: float = 8.4,
) -> None:
    ax_tbl = ax.figure.add_axes([x, y, w, h])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 == 1 else "#F8FAFC")
            cell.set_text_props(color=SLATE)


def _save_slide(
    fig: plt.Figure,
    pdf: PdfPages,
    slide_num: int,
    slide_png_dir: Path | None = None,
    dpi: int = 150,
) -> None:
    pdf.savefig(fig, dpi=dpi)
    if slide_png_dir is not None:
        slide_png_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(slide_png_dir / f"slide_{slide_num:02d}.png", dpi=dpi)
    plt.close(fig)


def build_csir_meeting_pdf(
    out_pdf_path: str | Path,
    figures_dir: Path,
    rgb_metrics: Dict[str, Any],
    thz_metrics: Dict[str, Any],
    rgb_manifest_rows: List[Dict[str, str]],
    slide_png_dir: Path | None = None,
) -> None:
    """Build self-contained 10-slide presentation PDF embedding architecture, RGB & THz tests."""
    out_path = Path(out_pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 10

    with PdfPages(out_path) as pdf:
        # ------------------------------------------------------------------
        # SLIDE 1: Title, Scope & Decoupled Two-Model Architecture
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            1,
            total,
            "Terahertz Non-Destructive Testing (NDT) via DeepLabV3+",
            "Executive Summary, Optical Physics Constraints & Decoupled Two-Model Diagnostic Pipeline",
        )
        _draw_box(
            ax,
            0.025,
            0.47,
            0.44,
            0.39,
            "1. Project Objective, Scope & 15-Scan Inventory",
            [
                "• Goal: Automated 'X-ray style' NDT inspection of 3D-printed FDM polymers.",
                "• Outputs: (1) Material, (2) Infill %, (3) Pattern, (4) Defect Mask + Size/XY.",
                "• Scope: Standalone AI models & OpenCV metrology (no printer/stage drivers).",
                "• Physics: 0.135–0.450 THz reflection; 3–4 mm beam blurs voids; Δn ≈ 0.55.",
                "• 15 Scans: 4 polymers (PLA, PETG, ABS, TPU), 1 shared pattern, 15%–100%",
                "  infill; spans No-Defect (ND), Single-Void (1C/1S), 3x3 Arrays (3C/3S),",
                "  and Multi-Defect mixed plates (Cohort B).",
            ],
            header_color=NAVY,
        )
        _draw_box(
            ax,
            0.025,
            0.06,
            0.44,
            0.38,
            "2. Why We Decouple Into Two Specialized Models",
            [
                "• Small-Data Bottleneck: Only 15 THz scans exist (high scan cost & ~1000 hr",
                "  photomixer life) -> monolithic multi-task networks suffer gradient conflict.",
                "• Model 1 (Global Characterizer): ResNet-34 multi-head network for bulk",
                "  Material, Printing Pattern, and Infill % regression.",
                "• Model 2 (Defect Segmentor): THz-Adapted DeepLabV3+ focused solely",
                "  on pixel-level localization of blind and buried discontinuities.",
                "• Today's Deliverable: Prove DeepLabV3+ on real multi-sample RGB defect",
                "  data (24 images) AND verify localization on all 15 real THz CSV scans.",
            ],
            header_color=TEAL,
        )

        # Right side pipeline flow diagram
        _draw_box(
            ax,
            0.49,
            0.06,
            0.485,
            0.80,
            "End-to-End 6-Layer Diagnostic Architecture",
            [],
            header_color=NAVY,
            fill_color="#FFFFFF",
        )
        blocks = [
            (0.515, 0.72, 0.435, 0.085, "Layer 1: Signal Conditioning & Physics Stack",
             "Mask hot-strip columns -> Wiener filter -> [Norm Int, |Local DoG|, Sobel]", "#1E3A5F"),
            (0.515, 0.56, 0.205, 0.11, "Layer 3: Model 1\nGlobal Characterizer",
             "ResNet-34 Multi-Head:\nMat | Pattern | Infill %", TEAL),
            (0.745, 0.56, 0.205, 0.11, "Layer 4: Model 2\nDeepLabV3+ Segmentor",
             "ResNet-18 (OS=8) + ASPP\nFocal+Dice Pixel Mask", ACCENT),
            (0.515, 0.36, 0.435, 0.125, "Layer 5: Deterministic OpenCV Metrology",
             "Threshold (p>=0.5) -> Morph Open -> Connected Components & Moments\n"
             "Centroid (X,Y) mm | Area mm^2 | Beam-Corrected Diam sqrt(d_app^2 - d_beam^2)", GREEN),
            (0.515, 0.14, 0.435, 0.15, "Standardized 'X-Ray Style' Diagnostic Output (JSON + Overlay)",
             "• Sample Status: PASS / FAIL  • Defect Count & Centroids (mm)\n"
             "• Material / Infill / Pattern Metadata  • Containerized Docker GPU Training", "#334155"),
        ]
        for bx, by, bw, bh, bt, bb, col in blocks:
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (bx, by), bw, bh, boxstyle="round,pad=0.005,rounding_size=0.01",
                    facecolor=col, edgecolor="none",
                )
            )
            ax.text(bx + bw / 2, by + bh * 0.66, bt, color="white", fontsize=9.3, fontweight="bold", ha="center", va="center")
            ax.text(bx + bw / 2, by + bh * 0.27, bb, color="#F1F5F9", fontsize=8.0, ha="center", va="center")

        # Arrows connecting layers
        arrow_kw = dict(arrowstyle="-|>", color=NAVY, lw=1.8, mutation_scale=13)
        ax.annotate("", xy=(0.617, 0.675), xytext=(0.67, 0.715), arrowprops=arrow_kw)
        ax.annotate("", xy=(0.847, 0.675), xytext=(0.79, 0.715), arrowprops=arrow_kw)
        ax.annotate("", xy=(0.847, 0.49), xytext=(0.847, 0.555), arrowprops=arrow_kw)
        ax.annotate("", xy=(0.617, 0.295), xytext=(0.617, 0.555), arrowprops=arrow_kw)
        ax.annotate("", xy=(0.732, 0.295), xytext=(0.732, 0.355), arrowprops=arrow_kw)

        _save_slide(fig, pdf, 1, slide_png_dir=slide_png_dir)

        # ------------------------------------------------------------------
        # SLIDE 2: DeepLabV3+ Architecture & Standard vs THz Adaptation
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            2,
            total,
            "DeepLabV3+ Architecture & Critical Adaptation for THz C-Scans",
            "How Atrous Spatial Pyramid Pooling (ASPP) + Low-Level Decoder Fusion Works — and Why Stride-16 Fails on 46x46 THz Grids",
        )
        # Top Vector Block Diagram of DeepLabV3+
        ax.add_patch(
            mpatches.FancyBboxPatch((0.025, 0.48), 0.95, 0.38, boxstyle="round,pad=0.005", facecolor="#F8FAFC", edgecolor=BORDER)
        )
        ax.text(0.04, 0.835, "DeepLabV3+ Internal Dataflow (Encoder -> Multi-Rate ASPP -> 1/4 Low-Level Skip Decoder)",
                color=NAVY, fontsize=11, fontweight="bold")

        diag_nodes = [
            (0.04, 0.63, 0.11, 0.14, "Input Image\n3-Ch Tensor", "RGB: 128x128\nTHz: 128x128", NAVY),
            (0.18, 0.63, 0.13, 0.14, "ResNet-18\nBackbone", "Stage1: 1/4 (32x32)\nStage4: 1/8 (16x16)", "#1E40AF"),
            (0.35, 0.71, 0.17, 0.085, "ASPP Parallel Branches", "1x1 | r=d1 | r=d2 | r=d3 | Pool", TEAL),
            (0.35, 0.52, 0.17, 0.095, "Low-Level Skip (1/4)", "1x1 Conv: 64 -> 48 ch\nPreserves crisp edges", "#0F766E"),
            (0.56, 0.63, 0.15, 0.14, "Decoder Fusion", "Upsample ASPP 2x/4x\nConcat + 3x3 Refine", ACCENT),
            (0.75, 0.63, 0.19, 0.14, "Pixel Defect Mask", "Upsample -> 128x128\nSigmoid + Focal/Dice", GREEN),
        ]
        for nx, ny, nw, nh, nt, nb, nc in diag_nodes:
            ax.add_patch(mpatches.FancyBboxPatch((nx, ny), nw, nh, boxstyle="round,pad=0.004,rounding_size=0.008", facecolor=nc, edgecolor="none"))
            ax.text(nx + nw / 2, ny + nh * 0.65, nt, color="white", fontsize=8.8, fontweight="bold", ha="center", va="center")
            ax.text(nx + nw / 2, ny + nh * 0.25, nb, color="#E2E8F0", fontsize=7.6, ha="center", va="center")

        arr = dict(arrowstyle="-|>", color=SLATE, lw=1.6, mutation_scale=12)
        ax.annotate("", xy=(0.178, 0.70), xytext=(0.152, 0.70), arrowprops=arr)
        ax.annotate("", xy=(0.348, 0.75), xytext=(0.312, 0.72), arrowprops=arr)
        ax.annotate("", xy=(0.348, 0.57), xytext=(0.245, 0.625), arrowprops=arr)
        ax.annotate("", xy=(0.558, 0.72), xytext=(0.522, 0.75), arrowprops=arr)
        ax.annotate("", xy=(0.558, 0.66), xytext=(0.522, 0.57), arrowprops=arr)
        ax.annotate("", xy=(0.748, 0.70), xytext=(0.712, 0.70), arrowprops=arr)

        # Bottom left: Math explanation of why Stride 16 fails
        _draw_box(
            ax,
            0.025,
            0.06,
            0.44,
            0.39,
            "Why Textbook DeepLabV3+ Breaks on Raw THz",
            [
                "• Native THz scans are only 38x38 to 46x46 pixels (1 mm/px pitch).",
                "• Textbook DeepLabV3+ uses ResNet-101 with Output Stride = 16:",
                "      floor(46 / 16) = 2x2 or 3x3 bottleneck feature grid!",
                "• ASPP dilation rates r = (6, 12, 18) on a 3x3 grid sample 90%+",
                "  outside image borders (pure zero-padding) — destroying localization.",
                "• ResNet-101 (~44M params) instantly memorizes N=5..15 scans.",
                "• Our Fix: Bicubic upsample -> 128x128 + ResNet-18 (OS=8 -> 16x16)",
                "  + physics-scaled ASPP r=(1, 2, 4) spanning 7.5 to 22.5 mm.",
            ],
            header_color="#B91C1C",
        )
        # Bottom right: Side-by-side table
        _draw_table(
            ax,
            0.485,
            0.06,
            0.49,
            0.39,
            ["Component", "Textbook RGB DeepLabV3+", "Our THz-Adapted DeepLabV3+"],
            [
                ["Input Grid", "512 x 512 RGB", "38–66 px -> 128 x 128 (3-Ch Physics)"],
                ["Backbone", "ResNet-101 (~44M params)", "ResNet-18 (~11M params, dilated)"],
                ["Output Stride", "OS = 16 (3x3 on THz!)", "OS = 8 (16x16 high-level map)"],
                ["Low-Level Skip", "1/4 res (64 ch -> 48 ch)", "32 x 32 res (preserves 3-4 mm edges)"],
                ["ASPP Rates", "r = (6, 12, 18)", "r = (1, 2, 4) tuned to beam & 3x3 arrays"],
                ["Loss Function", "Pixel Cross-Entropy", "Focal + Soft Dice (imbalanced <5% voids)"],
            ],
            col_widths=[0.24, 0.36, 0.40],
            font_size=8.5,
        )
        _save_slide(fig, pdf, 2, slide_png_dir=slide_png_dir)

        # ------------------------------------------------------------------
        # SLIDE 3: Architectural Rationale & Why Over All Alternatives
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            3,
            total,
            "Architectural Rationale: Why This Design Over All Alternatives",
            "Trade-Off Analysis Across Segmentation Paradigms, Loss Formulation & Physics-Safe Augmentation Policy",
        )
        _draw_table(
            ax,
            0.025,
            0.40,
            0.95,
            0.46,
            ["Candidate Architecture", "Role in Project", "Strengths", "Why Rejected (or Restricted) for Primary THz Defect Metrology"],
            [
                ["Decoupled Model 1 +\nTHz-DeepLabV3+ (Ours)", "PRIMARY\nPIPELINE", "Isolates global vs pixel gradients;\nASPP + 1/4 fusion traces blurred flaws", "Requires stride-8 & dilation scaling\n(solved in our THz adaptation)"],
                ["Monolithic Multi-Task Network", "Rejected", "Single forward pass for all 4 outputs", "Severe gradient conflict on N=15..60 scans\ndestabilizes infill % & pixel masks"],
                ["YOLOv8-seg / Mask R-CNN", "Rejected", "Fast bounding-box instance segmentation", "Anchor boxes need sharp edges; fails on\n3-4 mm diffraction blur & seams"],
                ["PatchCore / PaDiM\n(Unsupervised)", "Screening\nReference", "Needs only defect-free nominal scans", "Uncalibrated anomaly heatmaps without\ncrisp closed contours for mm^2 area"],
                ["Standard U-Net (SmallUNet)", "Ablation\nBaseline", "Lightweight; strong on central blobs", "Lacks parallel multi-rate ASPP kernels\nto separate infill from wide seams"],
                ["Vision Transformers\n(SegFormer / Swin)", "Rejected\n(For Now)", "Strong long-range global attention", "Lacks CNN translation equivariance;\noverfits severely under N < 200 scans"],
            ],
            col_widths=[0.22, 0.11, 0.31, 0.36],
            font_size=8.1,
        )
        _draw_box(
            ax,
            0.025,
            0.06,
            0.46,
            0.31,
            "Loss Function: Why Compound Focal + Soft Dice",
            [
                "• Real defects occupy only 0.2%–2.0% of pixels (e.g., MT_Crack = 0.19%,",
                "  PLA_1S = 1.47%). Plain BCE collapses to predicting all-zero background.",
                "• Focal Loss (gamma=2.0, alpha=0.80) down-weights easy background pixels",
                "  and focuses gradients on uncertain diffraction-blurred void boundaries.",
                "• Soft Dice Loss directly optimizes region overlap (IoU/F1) across sparse",
                "  small defects and thin linear cracks.",
            ],
            header_color=TEAL,
        )
        _draw_box(
            ax,
            0.515,
            0.06,
            0.46,
            0.31,
            "Physics-Safe Augmentations (Zero Geometric Distortion)",
            [
                "• Allowed: 90/180/270 deg rotations (linear THz polarization invariance),",
                "  H/V flips, +-1 px shifts (matches observed cross-frequency offset),",
                "  +-10% gain jitter, and Gaussian-blurred synthetic defect injection.",
                "• Forbidden: Elastic warps, arbitrary-angle shear, or color jitter (destroys",
                "  millimeter pixel calibration & physical attenuation relationships).",
                "• Evaluation Guard: Strict train/test & Leave-One-Scan-Out splits.",
            ],
            header_color=ACCENT,
        )
        _save_slide(fig, pdf, 3, slide_png_dir=slide_png_dir)

        # ------------------------------------------------------------------
        # SLIDE 4: Benchmark 1 — Real Industrial RGB Defect Dataset (24 Samples)
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            4,
            total,
            "Benchmark 1: Curated Real Industrial RGB Defect Dataset (24 Samples)",
            "Demonstrating DeepLabV3+ Defect Identification & Pixel Localization on Real Ground-Truth Industrial Images",
        )
        _draw_box(
            ax,
            0.025,
            0.52,
            0.36,
            0.34,
            "Why Add Real RGB Benchmark",
            [
                "• Only 5 Cohort-A THz defect scans exist",
                "  without external CAD masks.",
                "• We pulled 24 real industrial defect images",
                "  + verified pixel masks from the public",
                "  Magnetic-Tile-Defect dataset.",
                "• Strict Split: 16 Train (4/class) and",
                "  8 Held-Out Test (2/class, unseen).",
            ],
            header_color=NAVY,
        )
        _draw_table(
            ax,
            0.025,
            0.06,
            0.36,
            0.43,
            ["Category", "Train / Test", "Defect Area %", "Polymer NDT Analogy"],
            [
                ["MT_Blowhole", "4 / 2", "0.26% – 0.48%", "Blind circular voids (1C)"],
                ["MT_Crack", "4 / 2", "0.19% – 1.84%", "Layer-3 cold-joint seams"],
                ["MT_Break", "4 / 2", "0.29% – 21.9%", "Pocket / edge flaws (1S)"],
                ["MT_Free", "4 / 2", "0.00% (Clean)", "Nominal control (PLA_ND)"],
            ],
            col_widths=[0.24, 0.20, 0.24, 0.32],
            font_size=8.2,
        )

        # Right side: 2x4 preview grid of real RGB images + masks
        cat_examples = []
        seen_cats = set()
        for row in rgb_manifest_rows:
            c = row["category"]
            if row["split"] == "test" and c not in seen_cats:
                seen_cats.add(c)
                cat_examples.append(row)
        # Ensure 4 categories shown cleanly (2 rows x 4 cols: Image + GT Mask per category)
        grid_ax_title = ax.text(0.69, 0.845, "Held-Out Real Test Exemplars & Ground-Truth Masks (1 per Category)",
                                color=NAVY, fontsize=10.5, fontweight="bold", ha="center")
        _ = grid_ax_title
        for idx, row in enumerate(cat_examples[:4]):
            col_x = 0.41 + (idx % 2) * 0.285
            row_y = 0.48 if idx < 2 else 0.08
            img_p = Path(row["image_path"]) if Path(row["image_path"]).exists() else figures_dir.parents[1] / row["image_path"]
            msk_p = Path(row["mask_path"]) if Path(row["mask_path"]).exists() else figures_dir.parents[1] / row["mask_path"]
            img_arr = np.asarray(Image.open(img_p).convert("RGB"))
            msk_arr = np.asarray(Image.open(msk_p).convert("L"))

            ax_im = fig.add_axes([col_x, row_y + 0.03, 0.13, 0.30])
            ax_im.imshow(img_arr)
            ax_im.set_title(f"{row['category']}\n({row['native_height']}x{row['native_width']})", fontsize=8.5, fontweight="bold")
            ax_im.axis("off")

            ax_mk = fig.add_axes([col_x + 0.135, row_y + 0.03, 0.13, 0.30])
            ax_mk.imshow(msk_arr, cmap="gray")
            ax_mk.set_title(f"GT Mask\n(FG={row['defect_area_pct']}%)", fontsize=8.5, fontweight="bold")
            ax_mk.axis("off")

        _save_slide(fig, pdf, 4, slide_png_dir=slide_png_dir)

        # ------------------------------------------------------------------
        # SLIDE 5: Real RGB Layer-by-Layer DeepLabV3+ Flow (Figure 07)
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            5,
            total,
            "How DeepLabV3+ Works Inside: Real RGB Layer-by-Layer Flow",
            "Live Feature Activations on Unseen Held-Out Test Samples (MT_Blowhole & MT_Crack) Through ResNet-18, ASPP & Decoder",
        )
        _place_image(fig, [0.02, 0.11, 0.96, 0.76], figures_dir / "07_real_rgb_deeplabv3plus_architecture_flow.png")
        _draw_box(
            ax,
            0.025,
            0.045,
            0.95,
            0.06,
            "Key Architectural Takeaway",
            [
                "Low-Level 1/4 (32x32) preserves fine defect edges despite vertical brushed texture; ASPP branches (r=2,4,8 + Global Pool) separate localized pits & hairline cracks from global illumination; Decoder fusion sharpens contours.",
            ],
            header_color=TEAL,
            title_size=9.2,
            body_size=8.3,
        )
        _save_slide(fig, pdf, 5, slide_png_dir=slide_png_dir)

        # ------------------------------------------------------------------
        # SLIDE 6: Real RGB Held-Out Test Results Gallery & Metrics (Figure 08)
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            6,
            total,
            "Real RGB Held-Out Test Results: Identification & Localization",
            "Evaluation on All 8 Strictly Unseen Industrial Test Samples (6 Defective + 2 Nominal Defect-Free Controls)",
        )
        _place_image(fig, [0.015, 0.08, 0.695, 0.77], figures_dir / "08_real_rgb_heldout_test_samples_results.png")

        summary_rgb = rgb_metrics["summary"]
        per_sample_rows = [
            [
                r["sample_id"].replace("MT_", ""),
                r["ground_truth_status"],
                r["predicted_status"],
                f"{r['ap']:.3f}" if r["ap"] is not None else "N/A",
                f"{r['dice']:.3f}",
            ]
            for r in rgb_metrics["per_sample_test_metrics"]
        ]
        _draw_box(
            ax,
            0.718,
            0.60,
            0.26,
            0.26,
            "Held-Out Test Summary (N = 8)",
            [
                f"• Identification Acc: {summary_rgb['identification_accuracy']*100:.1f}% ({summary_rgb['correct_identifications']}/8)",
                f"• Mean Defective AP: {summary_rgb['mean_defective_ap']:.3f}",
                f"• Mean Defective Dice: {summary_rgb['mean_defective_dice']:.3f}",
                f"• Mean Defective IoU: {summary_rgb['mean_defective_iou']:.3f}",
                f"• Clean Control False Alarms: {summary_rgb['clean_false_alarm_pixels']} px",
            ],
            header_color=GREEN,
            title_size=9.5,
            body_size=8.2,
        )
        _draw_table(
            ax,
            0.718,
            0.06,
            0.26,
            0.52,
            ["Test Sample", "GT", "Pred", "AP", "Dice"],
            per_sample_rows,
            col_widths=[0.44, 0.12, 0.14, 0.15, 0.15],
            font_size=7.4,
        )
        _save_slide(fig, pdf, 6, slide_png_dir=slide_png_dir)

        # ------------------------------------------------------------------
        # SLIDE 7: Benchmark 2 — 15 CSIR THz Scans & Conditioning (Figs 01 & 02)
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            7,
            total,
            "Benchmark 2: CSIR-CSIO Terahertz Polymer Dataset (15 Scans)",
            "CSIR Physical Distance (mm) Scales & High-Saturation Contrast Across All 4 Materials, Infills & Defect Cases",
        )
        _place_image(fig, [0.015, 0.21, 0.54, 0.66], figures_dir / "01_dataset_15scans_overview.png")
        _place_image(fig, [0.565, 0.21, 0.42, 0.66], figures_dir / "02_preprocessing_and_hotstrip_correction.png")
        _draw_box(
            ax,
            0.025,
            0.048,
            0.95,
            0.145,
            "15-Scan Taxonomy & CSIR Scale / Saturation Upgrade",
            [
                "• Defect Cases: 1 No-Defect control (ND), 2 Single-Void scans (1C/1S), 2 Uniform 3x3 Arrays (3C/3S), and 10 Multi-Defect Mixed Plates (Cohort B).",
                "• Material & Infill Coverage: 4 polymers (PLA, PETG, ABS, TPU), 1 shared printing pattern, across 15%, 45%, 75%, and 100% infill (135–450 GHz).",
                "• CSIR Visual Standard: Physical Distance (mm) axes + interior-anchored (p4–p96) jet stretch removes hot-strip compression so defects pop clearly.",
            ],
            header_color=NAVY,
            title_size=9.6,
            body_size=8.2,
        )
        _save_slide(fig, pdf, 7, slide_png_dir=slide_png_dir)

        # ------------------------------------------------------------------
        # SLIDE 8: THz-Adapted DeepLabV3+ Internal Activations (Figure 04)
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            8,
            total,
            "How THz-Adapted DeepLabV3+ Works on Real CSIR Scans",
            "Internal Feature Maps (mm Scales & Saturated Overlays) on Held-Out PLA_1C (450 GHz) & PLA_3S (200 GHz)",
        )
        _place_image(fig, [0.02, 0.14, 0.96, 0.72], figures_dir / "04_thz_deeplabv3plus_internal_activations.png")
        _draw_box(
            ax,
            0.025,
            0.048,
            0.95,
            0.08,
            "What These Real THz Activations Show",
            [
                "Calibrated in Distance (mm) with high-saturation overlays: ResNet-18 (OS=8) yields 32x32 edges and 16x16 ASPP grid—r=1 isolates single 3–4 mm cavities (PLA_1C), while r=2 & r=4 trace 3x3 arrays (PLA_3S).",
            ],
            header_color=ACCENT,
            title_size=9.5,
            body_size=8.2,
        )
        _save_slide(fig, pdf, 8, slide_png_dir=slide_png_dir)

        # ------------------------------------------------------------------
        # SLIDE 9: CSIR THz Test Samples — LOSO Localization & Metrology (Fig 05)
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            9,
            total,
            "CSIR THz Test Samples: Held-Out LOSO Localization & Metrology",
            "Leave-One-Scan-Out (THz-DeepLabV3+ vs SmallUNet) + Multi-Defect Plate Inference & mm Metrology",
        )
        _place_image(fig, [0.015, 0.048, 0.61, 0.83], figures_dir / "05_thz_defect_localization_all_samples.png")

        thz_loso = thz_metrics["thz_loso_verification"]
        thz_rows = []
        for f in thz_loso["per_fold_metrics"]:
            thz_rows.append([
                f["held_out_scan"].replace("_100p", ""),
                f"{f['deeplabv3plus_ap']:.3f}" if f["deeplabv3plus_ap"] is not None else "N/A (ND)",
                f"{f['unet_ap']:.3f}" if f["unet_ap"] is not None else "N/A (ND)",
                f"{f['deeplabv3plus_dice']:.3f}",
                str(f["deeplab_defects_detected"]),
            ])
        thz_rows.append([
            "MEAN (4 Defect)",
            f"{thz_loso['mean_deeplabv3plus_ap']:.3f}",
            f"{thz_loso['mean_unet_ap']:.3f}",
            f"{thz_loso['mean_deeplabv3plus_dice']:.3f}",
            "—",
        ])
        _draw_table(
            ax,
            0.635,
            0.46,
            0.345,
            0.40,
            ["Held-Out Scan", "DL AP", "UNet AP", "DL Dice", "Defects"],
            thz_rows,
            col_widths=[0.40, 0.14, 0.16, 0.15, 0.15],
            font_size=7.6,
        )
        _draw_box(
            ax,
            0.635,
            0.055,
            0.345,
            0.385,
            "Metrology Findings & Honest Caveats",
            [
                "• PLA_1S (325 GHz): AP=0.999, Dice=0.984 (1 defect,",
                "  beam-corrected diameter d=5.3 mm at center).",
                "• PLA_ND (300 GHz): PASS with 0 false-alarm pixels.",
                "• PLA_1C (450 GHz): Localized center void (24.5, 21.9 mm,",
                "  d=5.5 mm, conf=0.959) + triggered on sample bevel edge.",
                "• Multi-Defect Cohort B: Zero-shot inference highlights",
                "  mixed plate anomalies on ABS & PLA under mm scales.",
                "• Caveat: Masks are physics-derived until CAD truth arrives.",
            ],
            header_color=NAVY,
            title_size=9.5,
            body_size=8.0,
        )
        _save_slide(fig, pdf, 9, slide_png_dir=slide_png_dir)

        # ------------------------------------------------------------------
        # SLIDE 10: Model 1 Audit, Constraints & Future Improvements
        # ------------------------------------------------------------------
        fig, ax = _create_slide(
            10,
            total,
            "Model 1 Audit, Key Constraints & Future Improvements Roadmap",
            "4-Material & Multi-Infill Audit (Same Pattern) + Actionable Next Steps for CSIR-CSIO & SRM GPU Lab",
        )
        _place_image(fig, [0.02, 0.25, 0.54, 0.61], figures_dir / "06_model1_material_and_infill_separability.png")
        _draw_box(
            ax,
            0.02,
            0.05,
            0.54,
            0.18,
            "Model 1 Frequency Confounder Discovered in Cohort B",
            [
                "• At 400 GHz & 100% infill, PLA (+0.0581), ABS (+0.0600), and TPU (+0.0619) overlap within IQR.",
                "• PETG appears lower (+0.0300 at 100%, -0.0863 at 15%) ONLY because it ran at 450/425 GHz",
                "  vs 400/375 GHz. All 15 scans also share 1 pattern -> fixed-freq multi-pattern scans needed.",
            ],
            header_color="#B91C1C",
            title_size=9.4,
            body_size=8.0,
        )
        _draw_box(
            ax,
            0.575,
            0.47,
            0.405,
            0.39,
            "Future Model & Pipeline Improvements",
            [
                "1. Transfer Learning: Pretrain ResNet-18 DeepLabV3+ on",
                "   industrial defect RGB data -> fine-tune on THz scans.",
                "2. Plate-Mask Extraction: Segment outer polymer boundary",
                "   first to suppress bevel edge false alarms (seen in PLA_1C).",
                "3. Multi-Class & Phase Fusion: Separate blind/buried defect",
                "   kinds via 350+500 GHz magnitude + ToF phase fusion.",
                "4. SRM GPU Lab Scaling: Run 5-fold GroupKFold sweeps in",
                "   our CUDA Docker container (docker/Dockerfile) at N>=45.",
            ],
            header_color=TEAL,
            title_size=9.6,
            body_size=8.2,
        )
        _draw_box(
            ax,
            0.575,
            0.05,
            0.405,
            0.39,
            "Priority Requests for CSIR-CSIO Next Batch",
            [
                "1. Fixed Acquisition Settings: Same frequency (e.g., 400",
                "   or 500 GHz) + fixed gain + empty-holder reference scan.",
                "2. Ground-Truth Metrology Table: True CAD/caliper defect",
                "   diameter, depth (mm), and center (X, Y) for calibration.",
                "3. Confirmed Buried & Mixed Defects: Layer-3 power-cut",
                "   prints paired with clean twins (~25-45 scans).",
                "4. Pattern Diversity: Grid, Gyroid, Concentric prints at",
                "   20%-60% infill to enable Model 1 Pattern classification.",
            ],
            header_color=GREEN,
            title_size=9.6,
            body_size=8.2,
        )
        _save_slide(fig, pdf, 10, slide_png_dir=slide_png_dir)
