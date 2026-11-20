"""Step 1 Verification: Audit all 15 THz CSV scans, build manifest/masks, and plot conditioning."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing import (
    build_physics_tensor,
    extract_reference_mask,
    load_all_scans,
    robust_interior_normalize,
)
from src.visualization import plot_dataset_15scans_overview, plot_preprocessing_and_hotstrip


def main() -> None:
    cfg_path = REPO_ROOT / "configs" / "default_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    raw_dir = REPO_ROOT / cfg["data"]["raw_dir"]
    manifest_path = REPO_ROOT / cfg["data"]["manifest_path"]
    masks_dir = REPO_ROOT / cfg["data"]["masks_dir"]
    figures_dir = REPO_ROOT / "outputs" / "figures"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_scans(raw_dir)
    print(f"[INFO] Loaded {len(records)} raw THz CSV scans from {raw_dir}")

    border_px = int(cfg["preprocessing"]["hot_strip_border_px"])
    dog_s = float(cfg["preprocessing"]["dog_sigma_small"])
    dog_l = float(cfg["preprocessing"]["dog_sigma_large"])

    manifest_rows = []
    preview_items = []
    preview_targets = {
        "PLA_ND_300THz_100p",
        "PLA_1C_450THz_100p",
        "PLA_1S_325THz_100p",
        "PLA_3C_135THz_100p",
        "PLA_0.4THz_100p",
    }

    for rec in records:
        _, stats = robust_interior_normalize(rec.array, border_px=border_px, use_wiener=True)
        tensor = build_physics_tensor(
            rec.array,
            border_px=border_px,
            dog_sigma_small=dog_s,
            dog_sigma_large=dog_l,
        )
        mask = extract_reference_mask(rec.array, defect_code=rec.defect_code)

        mask_npy = masks_dir / f"{rec.stem}_mask.npy"
        mask_png = masks_dir / f"{rec.stem}_mask.png"
        np.save(mask_npy, mask)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_png)

        pos_pct = float(mask.mean() * 100.0)
        manifest_rows.append(
            {
                "stem": rec.stem,
                "filename": rec.filename,
                "cohort": rec.cohort,
                "specimen_id": rec.specimen_id,
                "material": rec.material,
                "frequency_ghz": round(rec.frequency_ghz, 2),
                "infill_pct": rec.infill_pct,
                "pattern": rec.pattern,
                "defect_code": rec.defect_code,
                "rows": rec.shape[0],
                "cols": rec.shape[1],
                "raw_min": round(float(rec.array.min()), 5),
                "raw_max": round(float(rec.array.max()), 5),
                "interior_median": round(stats["interior_median"], 5),
                "interior_iqr": round(stats["interior_iqr"], 5),
                "ref_mask_fg_pct": round(pos_pct, 2),
                "raw_csv_path": Path(rec.csv_path).relative_to(REPO_ROOT).as_posix(),
                "mask_npy_path": mask_npy.relative_to(REPO_ROOT).as_posix(),
            }
        )

        if rec.stem in preview_targets:
            preview_items.append(
                {
                    "stem": rec.stem,
                    "label": f"{rec.stem}\n({rec.shape[0]}x{rec.shape[1]}, {rec.frequency_ghz:.0f}GHz)",
                    "raw": rec.array,
                    "tensor": tensor,
                    "mask": mask,
                }
            )

        print(
            f"  - {rec.filename:28s} | {rec.shape[0]:2d}x{rec.shape[1]:2d} | "
            f"{rec.frequency_ghz:5.0f} GHz | med={stats['interior_median']:+.4f} | "
            f"fg_mask={pos_pct:5.2f}%"
        )

    with open(manifest_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[INFO] Wrote dataset manifest -> {manifest_path}")

    fig1_path = figures_dir / "01_dataset_15scans_overview.png"
    plot_dataset_15scans_overview(records, fig1_path)
    print(f"[INFO] Saved Figure 01 -> {fig1_path}")

    fig2_path = figures_dir / "02_preprocessing_and_hotstrip_correction.png"
    plot_preprocessing_and_hotstrip(preview_items, fig2_path)
    print(f"[INFO] Saved Figure 02 -> {fig2_path}")


if __name__ == "__main__":
    main()
