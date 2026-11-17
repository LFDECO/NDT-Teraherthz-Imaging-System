"""Quantitative OpenCV defect metrology and standardized JSON inspection reporting."""

from .contour_metrology import (
    build_inspection_report,
    extract_defect_metrology,
    save_inspection_report,
)

__all__ = [
    "build_inspection_report",
    "extract_defect_metrology",
    "save_inspection_report",
]
