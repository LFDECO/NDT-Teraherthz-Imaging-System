"""Automated verification suite for THz-AI Polymer NDT pipeline and folder governance."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrology import build_inspection_report, extract_defect_metrology
from src.models import (
    DeepLabV3Plus,
    FocalDiceLoss,
    Model1Characterizer,
    MultiTaskCharacterizerLoss,
    SmallUNet,
)
from src.preprocessing import build_physics_tensor, load_all_scans, robust_interior_normalize


class TestTHzPipelineAndGovernance(unittest.TestCase):
    """End-to-end checks on data hygiene, models, metrology, outputs, and folder READMEs."""

    def test_01_all_15_csvs_load_and_normalize(self) -> None:
        records = load_all_scans(REPO_ROOT / "Polymer sample data")
        self.assertEqual(len(records), 15)
        for rec in records:
            self.assertGreaterEqual(rec.frequency_ghz, 100.0)
            self.assertLessEqual(rec.frequency_ghz, 600.0)
            norm, stats = robust_interior_normalize(rec.array)
            self.assertEqual(norm.shape, rec.shape)
            self.assertTrue(np.all(np.isfinite(norm)))
            self.assertGreaterEqual(float(norm.min()), 0.0)
            self.assertLessEqual(float(norm.max()), 1.0)
            tensor = build_physics_tensor(rec.array)
            self.assertEqual(tensor.shape, (3, rec.shape[0], rec.shape[1]))
            self.assertTrue("interior_median" in stats)

    def test_02_deeplab_rgb_and_thz_modes(self) -> None:
        # RGB Demo Mode (OS=16, r=6,12,18)
        dl_rgb = DeepLabV3Plus(in_channels=3, num_classes=1, backbone="resnet34", output_stride=16, aspp_rates=(6, 12, 18))
        x_rgb = torch.randn(2, 3, 256, 256)
        inter_rgb = dl_rgb.forward_with_intermediates(x_rgb)
        self.assertEqual(inter_rgb["logits"].shape, (2, 1, 256, 256))
        self.assertEqual(inter_rgb["encoder_high_feat"].shape[-2:], (16, 16))
        self.assertEqual(inter_rgb["low_level_feat"].shape[-2:], (64, 64))

        # THz NDT Mode (OS=8, r=1,2,4)
        dl_thz = DeepLabV3Plus(in_channels=3, num_classes=1, backbone="resnet18", output_stride=8, aspp_rates=(1, 2, 4))
        x_thz = torch.randn(2, 3, 128, 128)
        y_thz = torch.randint(0, 2, (2, 1, 128, 128)).float()
        inter_thz = dl_thz.forward_with_intermediates(x_thz)
        self.assertEqual(inter_thz["logits"].shape, (2, 1, 128, 128))
        self.assertEqual(inter_thz["encoder_high_feat"].shape[-2:], (16, 16))
        self.assertEqual(inter_thz["low_level_feat"].shape[-2:], (32, 32))

        loss = FocalDiceLoss()(inter_thz["logits"], y_thz)
        loss.backward()

        # SmallUNet comparator check
        unet = SmallUNet(in_channels=3, num_classes=1)
        self.assertEqual(unet(x_thz).shape, (2, 1, 128, 128))

    def test_03_model1_characterizer_forward_and_loss(self) -> None:
        m1 = Model1Characterizer(in_channels=3, num_materials=4, num_patterns=4, backbone="resnet34")
        x = torch.randn(4, 3, 128, 128)
        out = m1(x)
        self.assertEqual(out["material_logits"].shape, (4, 4))
        self.assertEqual(out["pattern_logits"].shape, (4, 4))
        self.assertEqual(out["infill_pred"].shape, (4, 1))
        loss = MultiTaskCharacterizerLoss()(
            out,
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0.2, 0.4, 0.75, 1.0]),
        )
        loss.backward()

    def test_04_opencv_metrology_outputs_valid_mm_and_json(self) -> None:
        prob = np.zeros((46, 46), dtype=np.float32)
        yy, xx = np.ogrid[:46, :46]
        prob[((yy - 22) ** 2 + (xx - 24) ** 2) <= 3.0**2] = 0.92
        binary, defects = extract_defect_metrology(prob, native_shape=(46, 46), pixel_pitch_mm=1.0)
        self.assertEqual(binary.shape, (46, 46))
        self.assertEqual(len(defects), 1)
        self.assertAlmostEqual(defects[0]["centroid_mm"]["x"], 24.0, delta=0.5)
        self.assertAlmostEqual(defects[0]["centroid_mm"]["y"], 22.0, delta=0.5)
        report = build_inspection_report("TEST_SPEC", (46, 46), 450.0, 1.0, defects)
        self.assertEqual(report["subsurface_defect_analysis"]["status"], "FAIL")

    def test_05_generated_outputs_exist_and_nonempty(self) -> None:
        expected_figs = [
            "01_dataset_15scans_overview.png",
            "02_preprocessing_and_hotstrip_correction.png",
            "03_deeplabv3plus_rgb_architecture_demo.png",
            "04_thz_deeplabv3plus_internal_activations.png",
            "05_thz_defect_localization_all_samples.png",
            "06_model1_material_and_infill_separability.png",
            "07_real_rgb_deeplabv3plus_architecture_flow.png",
            "08_real_rgb_heldout_test_samples_results.png",
        ]
        for fname in expected_figs:
            p = REPO_ROOT / "outputs" / "figures" / fname
            self.assertTrue(p.exists(), f"Missing expected figure: {p}")
            self.assertGreater(p.stat().st_size, 10_000, f"Empty figure: {p}")

        metrics_path = REPO_ROOT / "outputs" / "reports" / "deeplab_vs_unet_verification_metrics.json"
        self.assertTrue(metrics_path.exists())
        with open(metrics_path, "r", encoding="utf-8") as fp:
            m = json.load(fp)
        self.assertIn("thz_loso_verification", m)

        rgb_metrics_path = REPO_ROOT / "outputs" / "reports" / "real_rgb_deeplab_metrics.json"
        self.assertTrue(rgb_metrics_path.exists())
        with open(rgb_metrics_path, "r", encoding="utf-8") as fp:
            rm = json.load(fp)
        self.assertEqual(rm["summary"]["total_test_samples"], 8)

        metadata_manifest = REPO_ROOT / "data" / "metadata_manifest.csv"
        self.assertTrue(metadata_manifest.exists())
        metadata_text = metadata_manifest.read_text(encoding="utf-8")
        self.assertNotIn("\\", metadata_text, "data/metadata_manifest.csv must use POSIX forward slashes")
        metadata_lines = [l for l in metadata_text.splitlines() if l.strip()]
        self.assertEqual(len(metadata_lines), 16)  # 1 header + 15 scans

        rgb_manifest = REPO_ROOT / "data" / "rgb_defect_samples" / "rgb_manifest.csv"
        self.assertTrue(rgb_manifest.exists())
        rgb_manifest_text = rgb_manifest.read_text(encoding="utf-8")
        self.assertNotIn("\\", rgb_manifest_text, "rgb_manifest.csv must use POSIX forward slashes")
        manifest_lines = [l for l in rgb_manifest_text.splitlines() if l.strip()]
        self.assertEqual(len(manifest_lines), 25)  # 1 header + 24 samples
        with open(rgb_manifest, "r", encoding="utf-8") as fp:
            rgb_rows = list(csv.DictReader(fp))
        self.assertEqual(
            {r["sample_id"] for r in rgb_rows if r["split"] == "test"},
            {m["sample_id"] for m in rm["per_sample_test_metrics"]},
        )

        pdf_path = REPO_ROOT / "outputs" / "CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pdf"
        if (REPO_ROOT / "docs").exists() or pdf_path.exists():
            self.assertTrue(pdf_path.exists(), f"Missing presentation PDF: {pdf_path}")
            self.assertGreater(pdf_path.stat().st_size, 500_000, f"PDF unexpectedly small: {pdf_path}")
            with open(pdf_path, "rb") as fp:
                self.assertTrue(fp.read(5) == b"%PDF-", "Invalid PDF header")

            pptx_path = REPO_ROOT / "outputs" / "CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pptx"
            self.assertTrue(pptx_path.exists(), f"Missing presentation PPTX: {pptx_path}")
            self.assertGreater(pptx_path.stat().st_size, 500_000, f"PPTX unexpectedly small: {pptx_path}")
            from pptx import Presentation

            prs = Presentation(pptx_path)
            self.assertEqual(len(prs.slides), 10)
            for idx, slide in enumerate(prs.slides, start=1):
                notes_txt = slide.notes_slide.notes_text_frame.text.strip()
                self.assertGreater(len(notes_txt), 50, f"Slide {idx} missing speaker notes in PPTX")

            docx_path = REPO_ROOT / "outputs" / "CSIR_Presentation_Side_By_Side_Speaker_Guide.docx"
            self.assertTrue(docx_path.exists(), f"Missing speaker DOCX: {docx_path}")
            self.assertGreater(docx_path.stat().st_size, 500_000, f"DOCX unexpectedly small: {docx_path}")
            from docx import Document

            doc = Document(docx_path)
            self.assertGreaterEqual(len(doc.tables), 12)

            guide_md = REPO_ROOT / "docs" / "CSIR_PRESENTATION_SPEAKER_GUIDE.md"
            self.assertTrue(guide_md.exists(), f"Missing speaker Markdown guide: {guide_md}")
            guide_text = guide_md.read_text(encoding="utf-8")
            for s_num in range(1, 11):
                self.assertIn(f"Slide {s_num} / 10", guide_text)

        nd_report = REPO_ROOT / "outputs" / "reports" / "inspection_report_PLA_ND_300THz_100p.json"
        self.assertTrue(nd_report.exists())

    def test_06_every_folder_has_governance_readme(self) -> None:
        if not (REPO_ROOT / "docs").exists():
            self.skipTest("Local subfolder governance READMEs are excluded from public GitHub clones via .gitignore")
        required_markers = [
            "Folder Responsibility",
            "Verification",
            "Why This Approach",
            "What Could Be Done Better",
            "Changelog",
        ]
        skip_dirs = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
        for root, dirs, _ in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            folder = Path(root)
            if folder == REPO_ROOT:
                continue
            readme = folder / "README.md"
            self.assertTrue(readme.exists(), f"Folder missing mandatory README.md: {folder}")
            text = readme.read_text(encoding="utf-8")
            for marker in required_markers:
                self.assertIn(marker, text, f"README in {folder} missing required section '{marker}'")

    def test_07_gitignore_and_public_privacy_hygiene(self) -> None:
        if not (REPO_ROOT / ".git").exists() or shutil.which("git") is None:
            self.skipTest("Git repository metadata or git CLI unavailable")

        ignored_samples = [
            ".env",
            "docs/PRD.md",
            "docs/CSIR_PRESENTATION_SPEAKER_GUIDE.md",
            "outputs/README.md",
            "outputs/CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pdf",
            "outputs/CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pptx",
            "outputs/CSIR_Presentation_Side_By_Side_Speaker_Guide.docx",
            "outputs/reports/rgb_demo_plot_cache.npz",
            "outputs/reports/thz_demo_plot_cache.npz",
            "outputs/~$CSIR_Presentation_Side_By_Side_Speaker_Guide.docx",
        ]
        for rel in ignored_samples:
            res = subprocess.run(
                ["git", "check-ignore", "-q", rel],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                res.returncode,
                0,
                f"Expected path to be ignored by .gitignore: {rel}",
            )

        ls_res = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        )
        pushable_rels = [
            p for p in ls_res.stdout.decode("utf-8", errors="replace").split("\0") if p
        ]
        self.assertGreater(len(pushable_rels), 20)
        pushable_set = set(pushable_rels)
        for code_dir in ("src", "scripts", "tests"):
            for py_path in (REPO_ROOT / code_dir).rglob("*.py"):
                if "__pycache__" in py_path.parts:
                    continue
                py_rel = py_path.relative_to(REPO_ROOT).as_posix()
                self.assertIn(py_rel, pushable_set, f"Python source file unexpectedly ignored by .gitignore: {py_rel}")

        blocked_exts = {
            ".pdf",
            ".pptx",
            ".docx",
            ".npz",
            ".pt",
            ".pth",
            ".ckpt",
            ".safetensors",
            ".onnx",
        }
        text_exts = {".py", ".yaml", ".yml", ".json", ".csv", ".txt", ".example", ".md"}
        forbidden_tokens = (
            "C:" + "\\Users\\",
            "/" + "Users/",
            "ghp" + "_",
            "AI" + "za",
            "sk-" + "proj-",
        )

        for rel in pushable_rels:
            fpath = REPO_ROOT / rel
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            self.assertNotIn(ext, blocked_exts, f"Blocked artifact would be pushed to GitHub: {rel}")
            if ext == ".md":
                self.assertEqual(rel, "README.md", f"Subfolder or internal markdown not ignored: {rel}")
            self.assertLessEqual(
                fpath.stat().st_size,
                2_000_000,
                f"Non-ignored file exceeds 2 MB threshold: {rel}",
            )
            if ext in text_exts or fpath.name in {"Dockerfile", ".env.example", ".gitattributes", ".gitignore"}:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
                for tok in forbidden_tokens:
                    self.assertNotIn(tok, content, f"Forbidden path/secret pattern found in {rel}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
