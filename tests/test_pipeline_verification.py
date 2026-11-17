"""Automated verification suite for THz-AI Polymer NDT pipeline and folder governance."""

from __future__ import annotations

import json
import os
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

        nd_report = REPO_ROOT / "outputs" / "reports" / "inspection_report_PLA_ND_300THz_100p.json"
        self.assertTrue(nd_report.exists())

    def test_06_every_folder_has_governance_readme(self) -> None:
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
            readme = folder / "README.md"
            self.assertTrue(readme.exists(), f"Folder missing mandatory README.md: {folder}")
            text = readme.read_text(encoding="utf-8")
            for marker in required_markers:
                self.assertIn(marker, text, f"README in {folder} missing required section '{marker}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
