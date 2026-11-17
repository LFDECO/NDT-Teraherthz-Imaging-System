"""Preprocessing utilities for THz CSV scan loading, normalization, and physics-safe augmentation."""

from .loader import ScanRecord, load_all_scans, parse_scan_filename
from .normalization import (
    build_physics_tensor,
    extract_reference_mask,
    mask_hot_strips,
    robust_interior_normalize,
)
from .augmentation import PhysicsSafeAugmentor, inject_synthetic_defect

__all__ = [
    "ScanRecord",
    "load_all_scans",
    "parse_scan_filename",
    "build_physics_tensor",
    "extract_reference_mask",
    "mask_hot_strips",
    "robust_interior_normalize",
    "PhysicsSafeAugmentor",
    "inject_synthetic_defect",
]
