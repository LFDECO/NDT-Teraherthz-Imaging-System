"""Build synchronized CSIR meeting PDF, native editable PPTX, and side-by-side speaker guides."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization import (
    build_csir_meeting_pdf,
    build_csir_meeting_pptx,
    build_side_by_side_speaker_guide_docx,
    build_side_by_side_speaker_guide_md,
)


def build_presentation_bundle(
    repo_root: Path = REPO_ROOT,
    copy_to_downloads: bool = False,
) -> Dict[str, Path]:
    """Generate PDF, editable PPTX, DOCX side-by-side guide, and MD guide from live outputs."""
    figures_dir = repo_root / "outputs" / "figures"
    reports_dir = repo_root / "outputs" / "reports"
    manifest_path = repo_root / "data" / "rgb_defect_samples" / "rgb_manifest.csv"

    with open(reports_dir / "real_rgb_deeplab_metrics.json", "r", encoding="utf-8") as fp:
        rgb_metrics = json.load(fp)
    with open(reports_dir / "deeplab_vs_unet_verification_metrics.json", "r", encoding="utf-8") as fp:
        thz_metrics = json.load(fp)
    with open(manifest_path, "r", encoding="utf-8") as fp:
        rgb_manifest_rows = list(csv.DictReader(fp))

    pdf_path = repo_root / "outputs" / "CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pdf"
    pptx_path = repo_root / "outputs" / "CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pptx"
    docx_path = repo_root / "outputs" / "CSIR_Presentation_Side_By_Side_Speaker_Guide.docx"
    md_path = repo_root / "docs" / "CSIR_PRESENTATION_SPEAKER_GUIDE.md"

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        slide_png_dir = Path(tmp_dir_str)
        build_csir_meeting_pdf(
            out_pdf_path=pdf_path,
            figures_dir=figures_dir,
            rgb_metrics=rgb_metrics,
            thz_metrics=thz_metrics,
            rgb_manifest_rows=rgb_manifest_rows,
            slide_png_dir=slide_png_dir,
        )
        print(f"[INFO] Generated 10-slide presentation PDF -> {pdf_path}", flush=True)

        slide_png_paths = sorted(slide_png_dir.glob("slide_*.png"))
        build_csir_meeting_pptx(
            out_pptx_path=pptx_path,
            figures_dir=figures_dir,
            rgb_metrics=rgb_metrics,
            thz_metrics=thz_metrics,
            rgb_manifest_rows=rgb_manifest_rows,
        )
        print(f"[INFO] Generated 10-slide editable PPTX -> {pptx_path}", flush=True)

        build_side_by_side_speaker_guide_docx(
            out_docx_path=docx_path,
            slide_png_paths=slide_png_paths,
            rgb_metrics=rgb_metrics,
            thz_metrics=thz_metrics,
        )
        print(f"[INFO] Generated side-by-side speaker DOCX -> {docx_path}", flush=True)

    build_side_by_side_speaker_guide_md(
        out_md_path=md_path,
        rgb_metrics=rgb_metrics,
        thz_metrics=thz_metrics,
    )
    print(f"[INFO] Generated side-by-side speaker MD -> {md_path}", flush=True)

    if copy_to_downloads:
        downloads_dir = Path.home() / "Downloads"
        if downloads_dir.exists():
            for src_p in (pdf_path, pptx_path, docx_path):
                dst_p = downloads_dir / src_p.name
                shutil.copy2(src_p, dst_p)
                print(f"[INFO] Copied to Downloads -> {dst_p}", flush=True)

    return {
        "pdf": pdf_path,
        "pptx": pptx_path,
        "docx": docx_path,
        "md": md_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble CSIR PDF, PPTX, DOCX, and Markdown speaker bundle.")
    parser.add_argument(
        "--copy-to-downloads",
        action="store_true",
        help="Also copy the generated PDF, PPTX, and DOCX files to ~/Downloads.",
    )
    args = parser.parse_args()
    build_presentation_bundle(copy_to_downloads=args.copy_to_downloads)


if __name__ == "__main__":
    main()
