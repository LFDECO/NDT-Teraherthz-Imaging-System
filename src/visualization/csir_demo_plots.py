"""Publication-grade visualization suite for CSIR-CSIO meeting & verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator


# Physical field-of-view extents (left, right, bottom, top) in mm matching CSIR MATLAB exports
_STEM_EXTENTS_MM: Dict[str, Tuple[float, float, float, float]] = {
    "PLA_ND_300THz_100p": (0.0, 40.0, 40.0, 0.0),
    "PLA_1C_450THz_100p": (0.0, 40.0, 40.0, 0.0),
    "PLA_1S_325THz_100p": (0.0, 40.0, 40.0, 0.0),
    "PLA_3C_135THz_100p": (0.0, 39.0, 39.0, 0.0),
    "PLA_3S_200THz_100p": (0.0, 38.0, 38.0, 0.0),
    "ABS_0.4THz_100p": (0.0, 60.0, 60.0, 0.0),
    "PLA_0.4THz_100p": (0.0, 60.0, 60.0, 0.0),
    "TPU_0.4THz_100p": (0.0, 60.0, 60.0, 0.0),
    "PETG_0.45THz_100p": (0.0, 60.0, 60.0, 0.0),
    "PLA_0.375THz_15p": (0.0, 55.0, 55.0, 0.0),
    "PLA_0.375THz_45p": (0.0, 55.0, 55.0, 0.0),
    "PLA_0.375THz_75p": (0.0, 55.0, 55.0, 0.0),
    "PETG_0.425THz_15p": (0.0, 55.0, 55.0, 0.0),
    "PETG_0.375THz_45p": (0.0, 55.0, 55.0, 0.0),
    "PETG_0.375THz_75p": (0.0, 55.0, 55.0, 0.0),
}

_GALLERY_ORDER_AND_TAGS: List[Tuple[str, str]] = [
    ("PLA_ND_300THz_100p", "No Defect"),
    ("PLA_1C_450THz_100p", "1 Circular"),
    ("PLA_1S_325THz_100p", "1 Square"),
    ("PLA_3C_135THz_100p", "3x3 Circular"),
    ("PLA_3S_200THz_100p", "3x3 Square"),
    ("PLA_0.375THz_15p", "Multi-Defect Plate"),
    ("PLA_0.375THz_45p", "Multi-Defect Plate"),
    ("PLA_0.375THz_75p", "Multi-Defect Plate"),
    ("PLA_0.4THz_100p", "Multi-Defect Plate"),
    ("ABS_0.4THz_100p", "Multi-Defect Plate"),
    ("PETG_0.425THz_15p", "Multi-Defect Plate"),
    ("PETG_0.375THz_45p", "Multi-Defect Plate"),
    ("PETG_0.375THz_75p", "Multi-Defect Plate"),
    ("PETG_0.45THz_100p", "Multi-Defect Plate"),
    ("TPU_0.4THz_100p", "Multi-Defect Plate"),
]


def _get_physical_extent_mm(
    stem: str | None,
    shape: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Return physical ``(0.0, W_mm, H_mm, 0.0)`` extent matching CSIR MATLAB figures."""
    if stem and stem in _STEM_EXTENTS_MM:
        return _STEM_EXTENTS_MM[stem]
    return (0.0, float(shape[1]), float(shape[0]), 0.0)


def _interior_saturated_limits(
    arr: np.ndarray,
    border_px: int = 4,
    low_pct: float = 4.0,
    high_pct: float = 96.0,
) -> Tuple[float, float]:
    """Compute tight interior-anchored ``(vmin, vmax)`` ignoring hot border flares."""
    h, w = arr.shape[:2]
    b = min(border_px, max(1, min(h, w) // 8))
    core = arr[b : h - b, b : w - b] if (h > 2 * b and w > 2 * b) else arr
    vmin, vmax = np.percentile(core, [low_pct, high_pct])
    vmin_f, vmax_f = float(vmin), float(vmax)
    if not np.isfinite(vmin_f) or not np.isfinite(vmax_f) or vmax_f <= vmin_f + 1e-5:
        vmin_f = float(np.nanmin(arr))
        vmax_f = max(float(np.nanmax(arr)), vmin_f + 1e-4)
    return vmin_f, vmax_f


def _pixel_center_grids(
    shape: Tuple[int, int],
    extent: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(X_grid, Y_grid)`` aligned to pixel centers under ``imshow(..., extent=extent)``."""
    h, w = int(shape[0]), int(shape[1])
    x0, x1, y1, y0 = extent
    dx = (x1 - x0) / max(w, 1)
    dy = (y1 - y0) / max(h, 1)
    xs = np.linspace(x0 + 0.5 * dx, x1 - 0.5 * dx, w)
    ys = np.linspace(y0 + 0.5 * dy, y1 - 0.5 * dy, h)
    return np.meshgrid(xs, ys)


def _build_saturated_prob_rgba(
    probs: np.ndarray,
    min_prob: float = 0.12,
    max_alpha: float = 0.85,
    cmap_name: str = "jet",
) -> np.ndarray:
    """Create RGBA heatmap where low-prob background is transparent and defects glow vividly."""
    p = np.clip(probs.astype(np.float32), 0.0, 1.0)
    rgba = plt.get_cmap(cmap_name)(p)
    ramp = np.clip((p - min_prob) / max(1.0 - min_prob, 1e-5), 0.0, 1.0)
    alpha = np.where(p < min_prob, 0.0, max_alpha * (ramp ** 0.65))
    rgba[..., 3] = np.clip(alpha, 0.0, max_alpha)
    return rgba


def _safe_percentile_limits(arr: np.ndarray, low: float = 5.0, high: float = 95.0) -> Tuple[float, float]:
    vmin, vmax = np.percentile(arr, [low, high])
    vmin_f, vmax_f = float(vmin), float(vmax)
    if not np.isfinite(vmin_f) or not np.isfinite(vmax_f) or vmax_f <= vmin_f + 1e-5:
        vmin_f = float(np.nanmin(arr))
        vmax_f = max(float(np.nanmax(arr)), vmin_f + 1e-4)
    return vmin_f, vmax_f


def _format_mm_axis(ax: plt.Axes, n_ticks: int = 4, tick_size: float = 7.0) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks, integer=True))
    ax.tick_params(labelsize=tick_size, length=2.5, pad=2)


def _add_compact_colorbar(
    fig: plt.Figure,
    mappable: Any,
    ax: plt.Axes,
    label: str = "",
    extend: str = "neither",
    tick_size: float = 6.5,
) -> None:
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.025, extend=extend)
    cbar.ax.tick_params(labelsize=tick_size, pad=1.5)
    if label:
        cbar.set_label(label, fontsize=tick_size, labelpad=2)


def _save_fig(fig: plt.Figure, out_path: str | Path, dpi: int = 180) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dataset_15scans_overview(records: List[Any], out_path: str | Path) -> None:
    """Plot a storytelling 3x5 gallery of all 15 CSV matrices with mm scales & saturated jet."""
    by_stem = {r.stem: r for r in records}
    ordered_items: List[Tuple[Any, str]] = []
    seen = set()
    for stem, tag in _GALLERY_ORDER_AND_TAGS:
        if stem in by_stem:
            ordered_items.append((by_stem[stem], tag))
            seen.add(stem)
    for rec in records:
        if rec.stem not in seen:
            tag = "Cohort A" if "cohort_a" in rec.cohort else "Multi-Defect Plate"
            ordered_items.append((rec, tag))

    n = len(ordered_items)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(19.0, 11.2), constrained_layout=True)
    axes_flat = axes.ravel()

    for idx, (rec, tag) in enumerate(ordered_items):
        ax = axes_flat[idx]
        arr = rec.array
        extent = _get_physical_extent_mm(rec.stem, rec.shape)
        vmin, vmax = _interior_saturated_limits(arr, border_px=4, low_pct=4.0, high_pct=96.0)
        im = ax.imshow(arr, cmap="jet", vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
        cohort_tag = "Cohort A" if "cohort_a" in rec.cohort else "Cohort B"
        w_mm, h_mm = int(round(extent[1])), int(round(extent[2]))
        ax.set_title(
            f"{rec.stem}\n{cohort_tag}: {tag} | {w_mm}x{h_mm}mm | {rec.frequency_ghz:.0f}GHz",
            fontsize=8.2,
            fontweight="bold",
        )
        ax.set_xlabel("Distance (mm)", fontsize=7.5)
        ax.set_ylabel("Distance (mm)", fontsize=7.5)
        _format_mm_axis(ax, n_ticks=4, tick_size=7.0)
        _add_compact_colorbar(fig, im, ax, label="Amp", extend="both", tick_size=6.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "CSIR-CSIO Raw THz CSV Dataset Audit (All 15 Scans in 'Polymer sample data/')\n"
        "4 Materials (PLA, PETG, ABS, TPU) | Same Pattern | 15%–100% Infill | Interior-Saturated Jet Scale (p4–p96)",
        fontsize=12.8,
        fontweight="bold",
    )
    _save_fig(fig, out_path)


def plot_preprocessing_and_hotstrip(samples: List[Dict[str, Any]], out_path: str | Path) -> None:
    """Visualize 5-stage preprocessing with physical mm scales, colorbars & saturated overlays."""
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(19.2, 3.35 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "1. Raw THz CSV (Sat. Jet)",
        "2. Hot-Strip Masked [Ch 0]",
        "3. Abs Local DoG [Ch 1]",
        "4. Sobel Edge [Ch 2]",
        "5. Reference Mask Overlay",
    ]
    for r_idx, item in enumerate(samples):
        raw = item["raw"]
        tensor = item["tensor"]  # (3, H, W)
        mask = item["mask"]
        label = item["label"]
        extent = _get_physical_extent_mm(item.get("stem"), raw.shape)
        raw_vmin, raw_vmax = _interior_saturated_limits(raw, border_px=4, low_pct=4.0, high_pct=96.0)
        xg, yg = _pixel_center_grids(mask.shape, extent)

        panels = [
            (raw, "jet", raw_vmin, raw_vmax, "both", "Amp"),
            (tensor[0], "jet", 0.05, 0.95, "both", "Norm"),
            (tensor[1], "inferno", 0.0, 0.80, "max", "DoG"),
            (tensor[2], "magma", 0.0, 0.80, "max", "Edge"),
            (mask, "jet", 0.0, 1.0, "neither", "Mask"),
        ]
        for c_idx, (img, cmap, vmin, vmax, ext_mode, cb_lbl) in enumerate(panels):
            ax = axes[r_idx, c_idx]
            if c_idx == 4:
                ax.imshow(tensor[0], cmap="gray", vmin=0.05, vmax=0.95, extent=extent, aspect="equal")
                rgba = _build_saturated_prob_rgba(mask, min_prob=0.10, max_alpha=0.72, cmap_name="jet")
                ax.imshow(rgba, extent=extent, aspect="equal")
                if float(mask.max()) >= 0.5:
                    ax.contour(xg, yg, mask, levels=[0.5], colors=["#00FF66"], linewidths=1.4)
                sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="jet")
                sm.set_array([])
                _add_compact_colorbar(fig, sm, ax, label=cb_lbl, extend="neither", tick_size=6.2)
            else:
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
                _add_compact_colorbar(fig, im, ax, label=cb_lbl, extend=ext_mode, tick_size=6.2)

            _format_mm_axis(ax, n_ticks=4, tick_size=6.8)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=9.4, fontweight="bold")
            if r_idx == n - 1:
                ax.set_xlabel("Distance (mm)", fontsize=8.0)
            if c_idx == 0:
                ax.set_ylabel(f"{label}\nDistance (mm)", fontsize=8.3, fontweight="bold")

    fig.suptitle(
        "Signal Conditioning Pipeline: Border Hot-Strip Suppression, Interior-Saturated Scales & Dip/Peak Anomalies",
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
    fig = plt.figure(figsize=(19.0, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 5)
    h_px, w_px = rgb_img.shape[:2]
    extent_px = (0.0, float(w_px), float(h_px), 0.0)
    xg, yg = _pixel_center_grids((h_px, w_px), extent_px)

    # Row 0: Input, Encoder Low-Level 1/4, Encoder High 1/16, Decoder Output, Final Segmentation
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb_img, extent=extent_px)
    ax0.set_title("1. RGB Polymer Specimen\n(Surface + Blind + Buried Defects)", fontsize=9.4, fontweight="bold")
    ax0.set_xlabel("X (px)", fontsize=7.5)
    ax0.set_ylabel("Y (px)", fontsize=7.5)
    _format_mm_axis(ax0, n_ticks=4, tick_size=6.8)

    low_feat = intermediates["low_level_feat"][0].detach().cpu().numpy().mean(axis=0)
    lv_min, lv_max = _safe_percentile_limits(low_feat, 4.0, 96.0)
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(low_feat, cmap="turbo", vmin=lv_min, vmax=lv_max, extent=extent_px)
    ax1.set_title(f"2. Encoder Low-Level (1/4)\nShape: {low_feat.shape[0]}x{low_feat.shape[1]} (Edges)", fontsize=9.4, fontweight="bold")
    _format_mm_axis(ax1, n_ticks=4, tick_size=6.8)
    _add_compact_colorbar(fig, im1, ax1, extend="both")

    high_feat = intermediates["encoder_high_feat"][0].detach().cpu().numpy().mean(axis=0)
    hv_min, hv_max = _safe_percentile_limits(high_feat, 4.0, 96.0)
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(high_feat, cmap="plasma", vmin=hv_min, vmax=hv_max, extent=extent_px)
    ax2.set_title(f"3. Encoder Bottleneck (1/16)\nShape: {high_feat.shape[0]}x{high_feat.shape[1]} (Semantic)", fontsize=9.4, fontweight="bold")
    _format_mm_axis(ax2, n_ticks=4, tick_size=6.8)
    _add_compact_colorbar(fig, im2, ax2, extend="both")

    dec_feat = intermediates["decoder_fused"][0].detach().cpu().numpy().mean(axis=0)
    dv_min, dv_max = _safe_percentile_limits(dec_feat, 4.0, 96.0)
    ax3 = fig.add_subplot(gs[0, 3])
    im3 = ax3.imshow(dec_feat, cmap="magma", vmin=dv_min, vmax=dv_max, extent=extent_px)
    ax3.set_title(f"4. Decoder Fusion (ASPP + 1/4)\nShape: {dec_feat.shape[0]}x{dec_feat.shape[1]}", fontsize=9.4, fontweight="bold")
    _format_mm_axis(ax3, n_ticks=4, tick_size=6.8)
    _add_compact_colorbar(fig, im3, ax3, extend="both")

    probs = intermediates["probs"][0, 0].detach().cpu().numpy()
    ax4 = fig.add_subplot(gs[0, 4])
    ax4.imshow(rgb_img, extent=extent_px)
    ax4.imshow(_build_saturated_prob_rgba(probs, min_prob=0.12, max_alpha=0.85, cmap_name="jet"), extent=extent_px)
    if float(probs.max()) >= 0.5:
        ax4.contour(xg, yg, (probs >= 0.5).astype(np.float32), levels=[0.5], colors=["#00FF66"], linewidths=1.8)
    if float(gt_mask.max()) >= 0.5:
        ax4.contour(xg, yg, (gt_mask >= 0.5).astype(np.float32), levels=[0.5], colors=["white"], linestyles="--", linewidths=1.2)
    ax4.set_title("5. Final Pixel Segmentation\n(Green: Pred, Dashed: GT)", fontsize=9.4, fontweight="bold")
    _format_mm_axis(ax4, n_ticks=4, tick_size=6.8)
    sm4 = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="jet")
    sm4.set_array([])
    _add_compact_colorbar(fig, sm4, ax4, label="Prob")

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
        bmin, bmax = _safe_percentile_limits(b_map, 4.0, 96.0)
        im_b = ax_b.imshow(b_map, cmap="inferno", vmin=bmin, vmax=bmax, extent=extent_px)
        ax_b.set_title(branch_titles.get(key, f"ASPP {key}"), fontsize=8.8, fontweight="bold")
        ax_b.set_xlabel("X (px)", fontsize=7.5)
        if col_idx == 0:
            ax_b.set_ylabel("Y (px)", fontsize=7.5)
        _format_mm_axis(ax_b, n_ticks=4, tick_size=6.8)
        _add_compact_colorbar(fig, im_b, ax_b, extend="both")

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
    """Show THz-adapted DeepLabV3+ internal activations with physical mm scales & saturated contrast."""
    n = len(activation_items)
    fig, axes = plt.subplots(n, 6, figsize=(20.5, 3.6 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = [
        "Input THz (Ch0 Sat.)",
        "Low-Level 1/4 (32x32)",
        "ASPP r=1 (Local Spot)",
        "ASPP r=2 (Beam Waist)",
        "ASPP r=4 (Array Context)",
        "Decoder Pred + Contour",
    ]
    for r_idx, item in enumerate(activation_items):
        inter = item["intermediates"]
        x_in = item["input_ch0"]
        native_shape = tuple(item.get("native_shape", x_in.shape))
        extent = _get_physical_extent_mm(item.get("stem"), native_shape)
        probs = inter["probs"][0, 0].detach().cpu().numpy()
        low = inter["low_level_feat"][0].detach().cpu().numpy().mean(axis=0)
        br = inter["aspp_branches"]
        r1 = br["rate_1"][0].detach().cpu().numpy().mean(axis=0)
        r2 = br["rate_2"][0].detach().cpu().numpy().mean(axis=0)
        r4 = br["rate_4"][0].detach().cpu().numpy().mean(axis=0)
        xg, yg = _pixel_center_grids(probs.shape, extent)

        l_min, l_max = _safe_percentile_limits(low, 5.0, 95.0)
        r1_min, r1_max = _safe_percentile_limits(r1, 5.0, 95.0)
        r2_min, r2_max = _safe_percentile_limits(r2, 5.0, 95.0)
        r4_min, r4_max = _safe_percentile_limits(r4, 5.0, 95.0)

        panels = [
            (x_in, "jet", 0.05, 0.95, "both", "Norm"),
            (low, "turbo", l_min, l_max, "both", "Act"),
            (r1, "inferno", r1_min, r1_max, "both", "Act"),
            (r2, "inferno", r2_min, r2_max, "both", "Act"),
            (r4, "inferno", r4_min, r4_max, "both", "Act"),
            (probs, "jet", 0.0, 1.0, "neither", "Prob"),
        ]
        for c_idx, (img, cmap, vmin, vmax, ext_mode, cb_lbl) in enumerate(panels):
            ax = axes[r_idx, c_idx]
            if c_idx == 5:
                ax.imshow(x_in, cmap="gray", vmin=0.05, vmax=0.95, extent=extent, aspect="equal")
                ax.imshow(_build_saturated_prob_rgba(probs, min_prob=0.12, max_alpha=0.85, cmap_name="jet"), extent=extent, aspect="equal")
                if float(probs.max()) >= 0.5:
                    ax.contour(xg, yg, (probs >= 0.5).astype(np.float32), levels=[0.5], colors=["#00FF66"], linewidths=1.6)
                sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="jet")
                sm.set_array([])
                _add_compact_colorbar(fig, sm, ax, label=cb_lbl, extend="neither", tick_size=6.3)
            else:
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
                _add_compact_colorbar(fig, im, ax, label=cb_lbl, extend=ext_mode, tick_size=6.3)

            _format_mm_axis(ax, n_ticks=4, tick_size=6.8)
            if r_idx == 0:
                ax.set_title(headers[c_idx], fontsize=9.2, fontweight="bold")
            if r_idx == n - 1:
                ax.set_xlabel("Distance (mm)", fontsize=8.0)
            if c_idx == 0:
                ax.set_ylabel(f"{item['label']}\nDistance (mm)", fontsize=8.3, fontweight="bold")

    fig.suptitle(
        "THz-Adapted DeepLabV3+ Internal Feature Maps (128x128 Upsample, ResNet-18 OS=8, ASPP r=[1,2,4], mm Scales)",
        fontsize=12.6,
        fontweight="bold",
    )
    _save_fig(fig, out_path)


def plot_thz_localization_gallery(
    eval_items: List[Dict[str, Any]],
    out_path: str | Path,
) -> None:
    """Plot end-to-end localization results with physical mm scales, colorbars & saturated overlays."""
    n = len(eval_items)
    fig, axes = plt.subplots(n, 4, figsize=(17.5, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "1. Raw THz CSV (Sat. Jet)",
        "2. Physics Anomaly [Ch 1 DoG]",
        "3. THz-DeepLabV3+ Probability",
        "4. Metrology Overlay (mm & Diam)",
    ]
    for r_idx, item in enumerate(eval_items):
        raw = item["raw"]
        dog = item["dog_ch1"]
        probs = item["prob_native"]
        binary = item["binary_native"]
        defects = item["defects"]
        title = item["label"]

        extent = _get_physical_extent_mm(item.get("stem"), raw.shape)
        _, w_mm, h_mm, _ = extent
        h_px, w_px = binary.shape
        xg, yg = _pixel_center_grids(binary.shape, extent)
        raw_vmin, raw_vmax = _interior_saturated_limits(raw, border_px=4, low_pct=4.0, high_pct=96.0)

        # Col 0: Raw CSV
        ax0 = axes[r_idx, 0]
        im0 = ax0.imshow(raw, cmap="jet", vmin=raw_vmin, vmax=raw_vmax, extent=extent, aspect="equal")
        _add_compact_colorbar(fig, im0, ax0, label="Amp", extend="both", tick_size=6.3)
        ax0.set_ylabel(f"{title}\nDistance (mm)", fontsize=8.2, fontweight="bold")

        # Col 1: DoG
        ax1 = axes[r_idx, 1]
        im1 = ax1.imshow(dog, cmap="inferno", vmin=0.0, vmax=0.80, extent=extent, aspect="equal")
        _add_compact_colorbar(fig, im1, ax1, label="DoG", extend="max", tick_size=6.3)

        # Col 2: Probabilities
        ax2 = axes[r_idx, 2]
        im2 = ax2.imshow(probs, cmap="jet", vmin=0.0, vmax=1.0, extent=extent, aspect="equal")
        _add_compact_colorbar(fig, im2, ax2, label="Prob", extend="neither", tick_size=6.3)

        # Col 3: Overlay + centroids in mm coordinates
        ax3 = axes[r_idx, 3]
        ax3.imshow(item["norm_ch0"], cmap="gray", vmin=0.05, vmax=0.95, extent=extent, aspect="equal")
        ax3.imshow(_build_saturated_prob_rgba(probs, min_prob=0.12, max_alpha=0.82, cmap_name="jet"), extent=extent, aspect="equal")
        if binary.max() > 0:
            ax3.contour(xg, yg, binary.astype(np.float32), levels=[0.5], colors=["#00FF66"], linewidths=1.6)
        for d in defects[:5]:
            cx_px = float(d["centroid_px"]["x"])
            cy_px = float(d["centroid_px"]["y"])
            x_mm = (cx_px + 0.5) * (w_mm / max(w_px, 1))
            y_mm = (cy_px + 0.5) * (h_mm / max(h_px, 1))
            ax3.plot(x_mm, y_mm, marker="+", color="white", markersize=8.5, markeredgewidth=1.6)
            tx = min(x_mm + 1.2, w_mm * 0.72)
            ty = max(y_mm - 1.4, h_mm * 0.09)
            ax3.text(
                tx,
                ty,
                f"d={d['beam_corrected_diameter_mm']:.1f}mm",
                color="#FFEE33",
                fontsize=6.8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.14", fc="black", alpha=0.72, lw=0),
            )
        sm3 = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="jet")
        sm3.set_array([])
        _add_compact_colorbar(fig, sm3, ax3, label="Prob", extend="neither", tick_size=6.3)

        status = "PASS (0 defects)" if len(defects) == 0 else f"FAIL ({len(defects)} defect{'s' if len(defects) > 1 else ''})"
        status_col = "#15803D" if len(defects) == 0 else "#B91C1C"
        ax3.text(
            0.03,
            0.96,
            status,
            transform=ax3.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", fc=status_col, alpha=0.88, lw=0),
        )

        for c_idx in range(4):
            ax_cur = axes[r_idx, c_idx]
            _format_mm_axis(ax_cur, n_ticks=4, tick_size=6.8)
            if r_idx == 0:
                ax_cur.set_title(col_titles[c_idx], fontsize=9.4, fontweight="bold")
            if r_idx == n - 1:
                ax_cur.set_xlabel("Distance (mm)", fontsize=8.0)

    fig.suptitle(
        "End-to-End THz-DeepLabV3+ Subsurface Defect Localization & OpenCV Metrology (Physical mm Scales)",
        fontsize=12.6,
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


def plot_real_rgb_deeplab_flow(
    flow_items: List[Dict[str, Any]],
    out_path: str | Path,
) -> None:
    """Plot layer-by-layer DeepLabV3+ activations on held-out RGB test images with scales & colorbars."""
    n = len(flow_items)
    fig, axes = plt.subplots(n, 8, figsize=(21.5, 3.55 * n), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "1. Held-Out RGB Image",
        "2. Local DoG [Ch 1]",
        "3. Low-Level 1/4 (32x32)",
        "4. ASPP r=2 (Local)",
        "5. ASPP r=4 (Mid)",
        "6. ASPP r=8 (Wide)",
        "7. Decoder Fusion (32x32)",
        "8. Pred (Green) vs GT (Cyan)",
    ]
    for r_idx, item in enumerate(flow_items):
        inter = item["intermediates"]
        rgb = item["rgb"]
        dog = item["dog_ch1"]
        gt = item["gt_mask"]
        probs = inter["probs"][0, 0].detach().cpu().numpy()

        h_px, w_px = rgb.shape[:2]
        extent_px = (0.0, float(w_px), float(h_px), 0.0)
        xg, yg = _pixel_center_grids((h_px, w_px), extent_px)

        low = inter["low_level_feat"][0].detach().cpu().numpy().mean(axis=0)
        br = inter["aspp_branches"]
        b_r2 = br["rate_2"][0].detach().cpu().numpy().mean(axis=0)
        b_r4 = br["rate_4"][0].detach().cpu().numpy().mean(axis=0)
        b_r8 = br["rate_8"][0].detach().cpu().numpy().mean(axis=0)
        dec = inter["decoder_fused"][0].detach().cpu().numpy().mean(axis=0)

        l_lim = _safe_percentile_limits(low, 4.0, 96.0)
        r2_lim = _safe_percentile_limits(b_r2, 4.0, 96.0)
        r4_lim = _safe_percentile_limits(b_r4, 4.0, 96.0)
        r8_lim = _safe_percentile_limits(b_r8, 4.0, 96.0)
        dec_lim = _safe_percentile_limits(dec, 4.0, 96.0)

        maps = [
            (rgb, None, None, None, "neither"),
            (dog, "inferno", 0.0, 0.85, "max"),
            (low, "turbo", l_lim[0], l_lim[1], "both"),
            (b_r2, "plasma", r2_lim[0], r2_lim[1], "both"),
            (b_r4, "plasma", r4_lim[0], r4_lim[1], "both"),
            (b_r8, "plasma", r8_lim[0], r8_lim[1], "both"),
            (dec, "magma", dec_lim[0], dec_lim[1], "both"),
            (probs, "jet", 0.0, 1.0, "neither"),
        ]
        for c_idx, (img, cmap, vmin, vmax, ext_mode) in enumerate(maps):
            ax = axes[r_idx, c_idx]
            if c_idx == 0:
                ax.imshow(img, extent=extent_px)
                ax.set_ylabel(f"{item['label']}\nY (px)", fontsize=8.4, fontweight="bold")
            elif c_idx == 7:
                ax.imshow(rgb, extent=extent_px)
                ax.imshow(_build_saturated_prob_rgba(probs, min_prob=0.15, max_alpha=0.85, cmap_name="jet"), extent=extent_px)
                if probs.max() >= 0.55:
                    ax.contour(xg, yg, (probs >= 0.55).astype(np.float32), levels=[0.5], colors=["#00FF66"], linewidths=1.7)
                if gt.max() >= 0.5:
                    ax.contour(xg, yg, (gt >= 0.5).astype(np.float32), levels=[0.5], colors=["#00FFFF"], linestyles="--", linewidths=1.3)
                sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="jet")
                sm.set_array([])
                _add_compact_colorbar(fig, sm, ax, extend="neither", tick_size=6.0)
            else:
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent_px)
                _add_compact_colorbar(fig, im, ax, extend=ext_mode, tick_size=6.0)

            ax.set_xticks([0, 64, 128])
            ax.set_yticks([0, 64, 128])
            ax.tick_params(labelsize=6.4, length=2.0, pad=1.5)
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=8.6, fontweight="bold")
            if r_idx == n - 1:
                ax.set_xlabel("X (px)", fontsize=7.5)

    fig.suptitle(
        "Real Industrial RGB Test Samples: Layer-by-Layer DeepLabV3+ Feature Activations (ResNet-18, OS=8, ASPP r=[2,4,8])",
        fontsize=12.5,
        fontweight="bold",
    )
    _save_fig(fig, out_path)


def plot_real_rgb_test_gallery(
    test_items: List[Dict[str, Any]],
    out_path: str | Path,
) -> None:
    """Plot all 8 held-out RGB test samples in a wide 4x8 super-grid with scales & saturated defects."""
    n = len(test_items)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 8, figsize=(19.8, 2.9 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    half_titles = [
        "1. Test Image",
        "2. GT Mask",
        "3. Prob Map",
        "4. Localization",
    ]
    for idx, item in enumerate(test_items):
        r_idx = idx % rows
        c_off = (idx // rows) * 4

        rgb = item["rgb"]
        gt = item["gt_mask"]
        probs = item["probs"]
        binary = item["binary"]
        centroids = item["centroids"]

        h_px, w_px = rgb.shape[:2]
        extent_px = (0.0, float(w_px), float(h_px), 0.0)
        xg, yg = _pixel_center_grids((h_px, w_px), extent_px)

        axes[r_idx, c_off + 0].imshow(rgb, extent=extent_px)
        axes[r_idx, c_off + 0].set_ylabel(item["label"], fontsize=7.8, fontweight="bold")

        axes[r_idx, c_off + 1].imshow(gt, cmap="gray", vmin=0.0, vmax=1.0, extent=extent_px)
        if gt.max() >= 0.5:
            axes[r_idx, c_off + 1].contour(xg, yg, (gt >= 0.5).astype(np.float32), levels=[0.5], colors=["#00FFFF"], linewidths=1.1)

        im2 = axes[r_idx, c_off + 2].imshow(probs, cmap="jet", vmin=0.0, vmax=1.0, extent=extent_px)
        _add_compact_colorbar(fig, im2, axes[r_idx, c_off + 2], extend="neither", tick_size=5.8)

        ax_loc = axes[r_idx, c_off + 3]
        ax_loc.imshow(rgb, extent=extent_px)
        ax_loc.imshow(_build_saturated_prob_rgba(probs, min_prob=0.15, max_alpha=0.85, cmap_name="jet"), extent=extent_px)
        if binary.max() > 0:
            ax_loc.contour(xg, yg, binary.astype(np.float32), levels=[0.5], colors=["#00FF66"], linewidths=1.6)
        for cx, cy in centroids[:4]:
            ax_loc.plot(cx + 0.5, cy + 0.5, marker="+", color="white", markersize=7.5, markeredgewidth=1.4)

        status_col = "#15803D" if item["pred_status"] == item["gt_status"] else "#B91C1C"
        ap_txt = f"AP={item['ap']:.2f}" if item["ap"] is not None else "Clean"
        ax_loc.set_xlabel(
            f"{item['pred_status']} (GT:{item['gt_status']}) | {ap_txt} | D={item['dice']:.2f}",
            fontsize=7.2,
            fontweight="bold",
            color=status_col,
        )

        for local_c in range(4):
            ax_cur = axes[r_idx, c_off + local_c]
            if r_idx == 0:
                ax_cur.set_title(half_titles[local_c], fontsize=8.8, fontweight="bold")
            if local_c == 0:
                ax_cur.set_xticks([0, 64, 128])
                ax_cur.set_yticks([0, 64, 128])
                ax_cur.tick_params(labelsize=6.0, length=2.0, pad=1.2)
            else:
                ax_cur.set_xticks([0, 64, 128])
                ax_cur.set_yticks([])
                ax_cur.tick_params(labelsize=6.0, length=2.0, pad=1.2)

    fig.suptitle(
        "Real Industrial RGB Test Set Evaluation (8 Strictly Unseen Samples Across 4 Defect Categories | Saturated Overlays)",
        fontsize=12.4,
        fontweight="bold",
    )
    _save_fig(fig, out_path)
