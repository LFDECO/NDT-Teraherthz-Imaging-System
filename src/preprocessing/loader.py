"""CSV scan loader and filename metadata parser for CSIR-CSIO Terahertz scans."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class ScanRecord:
    """Structured metadata and raw numerical array for a single THz C-scan."""

    stem: str
    filename: str
    csv_path: str
    png_path: Optional[str]
    cohort: str
    specimen_id: str
    material: str
    frequency_ghz: float
    infill_pct: float
    pattern: str
    defect_code: str
    has_known_defect: Optional[bool]
    array: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.array.shape[0]), int(self.array.shape[1])


def parse_scan_filename(filename: str) -> dict:
    """Parse CSIR-CSIO scan filenames into normalized physical metadata.

    Handles both naming conventions observed in ``Polymer sample data/``:
      1. Cohort A defect geometry: ``PLA_1C_450THz_100p.csv``
         Note: ``135THz``..``450THz`` in these filenames physically denote GHz.
      2. Cohort B material/infill plates: ``PETG_0.375THz_45p.csv``
    """
    stem = Path(filename).stem

    # Cohort A pattern: <MAT>_<DEFECT>_<FREQ_GHZ>THz_<INFILL>p
    m_a = re.match(r"^([A-Za-z]+)_([A-Za-z0-9]+)_(\d+(?:\.\d+)?)THz_(\d+)p$", stem)
    if m_a:
        material, defect_code, freq_val_str, infill_str = m_a.groups()
        freq_val = float(freq_val_str)
        # Values >= 10 labeled 'THz' in filenames (e.g., 135THz, 450THz) are GHz.
        freq_ghz = freq_val if freq_val >= 10.0 else freq_val * 1000.0
        has_defect = defect_code.upper() != "ND"
        return {
            "stem": stem,
            "cohort": "cohort_a_defect_geometry",
            "specimen_id": f"{material.upper()}_{defect_code.upper()}_{int(infill_str)}P",
            "material": material.upper(),
            "frequency_ghz": freq_ghz,
            "infill_pct": float(infill_str),
            "pattern": "Default_Rectilinear",
            "defect_code": defect_code.upper(),
            "has_known_defect": has_defect,
        }

    # Cohort B pattern: <MAT>_<FREQ_THZ>THz_<INFILL>p
    m_b = re.match(r"^([A-Za-z]+)_(\d+(?:\.\d+)?)THz_(\d+)p$", stem)
    if m_b:
        material, freq_val_str, infill_str = m_b.groups()
        freq_val = float(freq_val_str)
        freq_ghz = freq_val * 1000.0 if freq_val < 10.0 else freq_val
        return {
            "stem": stem,
            "cohort": "cohort_b_material_infill",
            "specimen_id": f"{material.upper()}_{int(infill_str)}P_{int(round(freq_ghz))}GHZ",
            "material": material.upper(),
            "frequency_ghz": freq_ghz,
            "infill_pct": float(infill_str),
            "pattern": "Default_Unknown",
            "defect_code": "PLATE_ARRAY",
            "has_known_defect": None,  # Plate contains multi-band features/circular voids
        }

    raise ValueError(f"Unrecognized THz CSV filename schema: {filename}")


def load_all_scans(raw_dir: str | Path) -> List[ScanRecord]:
    """Load all ``*.csv`` THz C-scans from ``raw_dir`` sorted alphabetically."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw THz data directory not found: {raw_path}")

    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {raw_path}")

    records: List[ScanRecord] = []
    for csv_file in csv_files:
        meta = parse_scan_filename(csv_file.name)
        arr = np.loadtxt(csv_file, delimiter=",", dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {csv_file}, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values detected in {csv_file}")

        png_candidate = csv_file.with_suffix(".png")
        png_path = str(png_candidate) if png_candidate.exists() else None

        records.append(
            ScanRecord(
                stem=meta["stem"],
                filename=csv_file.name,
                csv_path=str(csv_file),
                png_path=png_path,
                cohort=meta["cohort"],
                specimen_id=meta["specimen_id"],
                material=meta["material"],
                frequency_ghz=meta["frequency_ghz"],
                infill_pct=meta["infill_pct"],
                pattern=meta["pattern"],
                defect_code=meta["defect_code"],
                has_known_defect=meta["has_known_defect"],
                array=arr,
            )
        )
    return records
