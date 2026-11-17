"""Signal conditioning, hot-strip border masking, and physics tensor construction."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import wiener


def mask_hot_strips(
    arr: np.ndarray,
    border_px: int = 4,
    replace_with_interior_median: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppress outer hot-strip / stage-edge artifacts observed in THz CSV scans.

    Returns ``(cleaned_array, interior_mask)`` where ``interior_mask`` is True inside
    the valid sample core.
    """
    h, w = arr.shape
    b = min(border_px, max(1, min(h, w) // 8))
    interior_mask = np.zeros((h, w), dtype=bool)
    interior_mask[b : h - b, b : w - b] = True

    cleaned = arr.astype(np.float32).copy()
    if replace_with_interior_median:
        core_med = float(np.median(cleaned[interior_mask]))
        # Replace outer artifact frame gently by blending or clamping extreme border flares
        p5, p95 = np.percentile(cleaned[interior_mask], [5.0, 95.0])
        spread = max(float(p95 - p5), 1e-5)
        low_clamp = p5 - 0.5 * spread
        high_clamp = p95 + 0.5 * spread
        border = ~interior_mask
        cleaned[border] = np.clip(cleaned[border], low_clamp, high_clamp)
        # Specifically neutralize the left-edge hot-strip columns (0..b-1)
        cleaned[:, : max(2, b - 1)] = core_med
    return cleaned, interior_mask


def robust_interior_normalize(
    arr: np.ndarray,
    border_px: int = 4,
    use_wiener: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Safely normalize both positive (Cohort A) and signed-negative (Cohort B) scans.

    Avoids raw division by negative medians by shifting and scaling relative to
    interior median and interquartile / percentile spread.
    """
    cleaned, interior_mask = mask_hot_strips(arr, border_px=border_px)
    if use_wiener:
        # Adaptive 3x3 Wiener denoising preserves 3-4 mm beam-blurred edges
        w_out = wiener(cleaned, mysize=(3, 3))
        cleaned = np.where(np.isfinite(w_out), w_out, cleaned).astype(np.float32)

    core_vals = cleaned[interior_mask]
    core_med = float(np.median(core_vals))
    p5, p95 = np.percentile(core_vals, [5.0, 95.0])
    scale = max(float(p95 - p5), 1e-4)

    # Map robust interior span [p5, p95] around [0.15, 0.85] and clip to [0, 1]
    norm = (cleaned - p5) / scale
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

    stats = {
        "interior_median": core_med,
        "interior_p5": float(p5),
        "interior_p95": float(p95),
        "interior_iqr": float(np.percentile(core_vals, 75.0) - np.percentile(core_vals, 25.0)),
    }
    return norm, stats


def build_physics_tensor(
    arr: np.ndarray,
    border_px: int = 4,
    dog_sigma_small: float = 1.0,
    dog_sigma_large: float = 5.5,
) -> np.ndarray:
    """Build a 3-channel THz physics representation ``(3, H, W)`` in ``[0, 1]``.

    Channels:
      - Ch 0: Robustly normalized THz reflection amplitude.
      - Ch 1: Absolute Local Difference-of-Gaussians (DoG) anomaly map (highlights
        both destructive attenuation dips like 1C/1S and constructive peaks like 3C/3S).
      - Ch 2: Sobel gradient magnitude (captures diffraction-blurred void perimeters).
    """
    norm, _ = robust_interior_normalize(arr, border_px=border_px, use_wiener=True)

    # Local DoG anomaly map computed on cleaned array
    g_small = gaussian_filter(norm, sigma=dog_sigma_small)
    g_large = gaussian_filter(norm, sigma=dog_sigma_large)
    abs_dog = np.abs(g_small - g_large)

    # Suppress border DoG transition caused by hot-strip frame
    _, interior_mask = mask_hot_strips(arr, border_px=border_px + 1, replace_with_interior_median=False)
    abs_dog = abs_dog * interior_mask.astype(np.float32)
    dog_max = max(float(np.percentile(abs_dog[interior_mask], 99.5)), 1e-4)
    dog_norm = np.clip(abs_dog / dog_max, 0.0, 1.0).astype(np.float32)

    # Sobel edge magnitude inside interior
    gx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy) * interior_mask.astype(np.float32)
    grad_max = max(float(np.percentile(grad[interior_mask], 99.5)), 1e-4)
    grad_norm = np.clip(grad / grad_max, 0.0, 1.0).astype(np.float32)

    return np.stack([norm, dog_norm, grad_norm], axis=0).astype(np.float32)


def extract_reference_mask(
    arr: np.ndarray,
    defect_code: str,
    border_px: int = 7,
) -> np.ndarray:
    """Extract physics-consistent reference defect mask ``(H, W)`` in ``{0, 1}``.

    Note: Because CAD/caliper masks have not yet been supplied by CSIR-CSIO,
    these serve as reproducible reference pseudo-masks for pilot training and
    visual verification:
      - ``ND`` (No Defect): strictly all-zeros mask.
      - ``1C`` / ``1S``: isolates the primary central dip defect inside the plate core.
      - ``3C`` / ``3S``: isolates the 3x3 periodic defect peaks inside the plate core.
      - Cohort B plates: isolates significant localized circular anomalies inside core.
    """
    h, w = arr.shape
    code = defect_code.upper()
    if code == "ND":
        return np.zeros((h, w), dtype=np.float32)

    cleaned, _ = mask_hot_strips(arr, border_px=4, replace_with_interior_median=True)
    # Exclude outer bevels / edge roll-offs from defect candidate search
    b = min(border_px, max(4, min(h, w) // 5))
    valid_roi = np.zeros((h, w), dtype=bool)
    valid_roi[b : h - b, b : w - b] = True

    smooth_bg = gaussian_filter(cleaned, sigma=5.0)
    smooth_fg = gaussian_filter(cleaned, sigma=0.9)
    residual = smooth_fg - smooth_bg

    if code in {"1C", "1S"}:
        # Central void manifests as strong negative dip relative to surrounding plateau
        inner_b = max(b, min(h, w) // 4)
        center_roi = np.zeros((h, w), dtype=bool)
        center_roi[inner_b : h - inner_b, inner_b : w - inner_b] = True
        dip = np.maximum(0.0, -residual) * center_roi.astype(np.float32)
        if dip.max() <= 1e-5:
            return np.zeros((h, w), dtype=np.float32)
        thresh = 0.45 * float(dip.max())
        mask = (dip >= thresh).astype(np.uint8)
    elif code in {"3C", "3S"}:
        # 3x3 array at 135/200 GHz manifests as localized positive reflection peaks
        core_vals = cleaned[valid_roi]
        peak_score = np.maximum(0.0, gaussian_filter(cleaned, sigma=0.8) - gaussian_filter(cleaned, sigma=3.5))
        peak_score = peak_score * valid_roi.astype(np.float32)
        if peak_score.max() <= 1e-5:
            return np.zeros((h, w), dtype=np.float32)
        thresh = 0.34 * float(np.percentile(peak_score[valid_roi], 98.0))
        mask = (peak_score >= max(thresh, 1e-4)).astype(np.uint8)
    else:
        # General localized anomaly (either dip or peak) via robust z-score of residual
        core_res = residual[valid_roi]
        med = float(np.median(core_res))
        mad = max(float(np.median(np.abs(core_res - med))), 1e-4)
        z = np.abs(residual - med) / (1.4826 * mad)
        mask = ((z >= 2.8) & valid_roi).astype(np.uint8)

    # Clean single-pixel speckles while preserving multi-spot arrays
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Remove isolated 1-2 pixel dust
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean_mask = np.zeros_like(mask, dtype=np.float32)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= 3:
            clean_mask[labels == idx] = 1.0
    return clean_mask
