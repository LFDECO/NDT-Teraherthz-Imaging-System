"""OpenCV spatial-moment metrology and JSON report generation for THz NDT."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def extract_defect_metrology(
    prob_map: np.ndarray,
    native_shape: Tuple[int, int],
    pixel_pitch_mm: float = 1.0,
    prob_threshold: float = 0.50,
    morph_kernel_size: int = 3,
    min_area_mm2: float = 2.0,
    beam_width_mm: float = 3.5,
    border_ignore_px: int = 4,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Convert probability map ``(H_m, W_m)`` into physical millimeter defect metrology.

    Maps coordinates from model grid back to ``native_shape`` and converts pixels
    to millimeters via ``pixel_pitch_mm``. Applies Rayleigh beam-blur deconvolution
    ``d_true = sqrt(max(0.25, d_app^2 - d_beam^2))``.
    """
    native_h, native_w = int(native_shape[0]), int(native_shape[1])
    if prob_map.shape != (native_h, native_w):
        prob_native = cv2.resize(
            prob_map.astype(np.float32),
            (native_w, native_h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        prob_native = prob_map.astype(np.float32)

    binary = (prob_native >= prob_threshold).astype(np.uint8)

    # Zero out outer border ring to prevent scanner edge artifacts from triggering false positives
    b = min(border_ignore_px, max(1, min(native_h, native_w) // 8))
    binary[:b, :] = 0
    binary[native_h - b :, :] = 0
    binary[:, :b] = 0
    binary[:, native_w - b :] = 0

    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    defects: List[Dict[str, Any]] = []
    clean_binary = np.zeros_like(binary)

    pixel_area_mm2 = float(pixel_pitch_mm) ** 2
    defect_idx = 0
    for lbl in range(1, num_labels):
        px_area = int(stats[lbl, cv2.CC_STAT_AREA])
        area_mm2 = px_area * pixel_area_mm2
        if area_mm2 < min_area_mm2:
            continue

        comp_mask = (labels == lbl).astype(np.uint8)
        m = cv2.moments(comp_mask)
        if m["m00"] <= 0:
            continue

        cx_px = float(m["m10"] / m["m00"])
        cy_px = float(m["m01"] / m["m00"])
        cx_mm = cx_px * pixel_pitch_mm
        cy_mm = cy_px * pixel_pitch_mm

        d_app = 2.0 * math.sqrt(area_mm2 / math.pi)
        # First-order Rayleigh PSF beam-blur correction
        d_corrected = math.sqrt(max(0.25, d_app**2 - beam_width_mm**2))
        mean_conf = float(prob_native[comp_mask > 0].mean())

        defect_idx += 1
        clean_binary[comp_mask > 0] = 1
        defects.append(
            {
                "defect_id": defect_idx,
                "category": "Subsurface Blind/Buried Discontinuity",
                "centroid_px": {"x": round(cx_px, 2), "y": round(cy_px, 2)},
                "centroid_mm": {"x": round(cx_mm, 2), "y": round(cy_mm, 2)},
                "pixel_area": px_area,
                "apparent_area_mm2": round(area_mm2, 2),
                "apparent_diameter_mm": round(d_app, 2),
                "beam_corrected_diameter_mm": round(d_corrected, 2),
                "mean_mask_confidence": round(mean_conf, 3),
            }
        )

    return clean_binary, defects


def build_inspection_report(
    specimen_id: str,
    native_shape: Tuple[int, int],
    frequency_ghz: float,
    pixel_pitch_mm: float,
    defects: List[Dict[str, Any]],
    material_type: str = "PLA",
    material_confidence: Optional[float] = None,
    infill_pattern: str = "Default_Rectilinear",
    pattern_confidence: Optional[float] = None,
    infill_density_pct: Optional[float] = None,
    model_used: str = "THz-Adapted DeepLabV3+ (ResNet-18, OS=8, ASPP r=1,2,4)",
) -> Dict[str, Any]:
    """Assemble standardized 'X-ray style' polymer NDT JSON inspection report."""
    return {
        "inspection_metadata": {
            "specimen_id": specimen_id,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "frequency_ghz": round(float(frequency_ghz), 2),
            "scan_pitch_mm_per_pixel": float(pixel_pitch_mm),
            "native_dimensions_px": [int(native_shape[0]), int(native_shape[1])],
            "segmentation_model": model_used,
        },
        "global_structural_analysis": {
            "material_type": material_type,
            "material_confidence": material_confidence,
            "infill_pattern": infill_pattern,
            "pattern_confidence": pattern_confidence,
            "infill_density_pct": infill_density_pct,
            "note": "Model 1 multi-pattern/material training pending fixed-frequency CSIR multi-pattern scans.",
        },
        "subsurface_defect_analysis": {
            "status": "FAIL" if len(defects) > 0 else "PASS",
            "defects_detected_count": len(defects),
            "defects": defects,
        },
    }


def save_inspection_report(report: Dict[str, Any], out_path: str | Path) -> None:
    """Write formatted JSON inspection report to disk."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
