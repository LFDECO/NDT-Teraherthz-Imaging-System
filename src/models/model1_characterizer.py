"""Model 1: Global Structural Characterizer (Material, Printing Pattern, Infill %)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models


class Model1Characterizer(nn.Module):
    """Multi-head CNN characterizer decoupled from pixel-wise defect segmentation.

    Outputs:
      - Head A: Material class logits (4 classes: PLA, PETG, ABS, TPU)
      - Head B: Printing pattern logits (4 classes: Rectilinear, Grid, Gyroid, Concentric)
      - Head C: Continuous infill fraction in [0.0, 1.0] (scaled to 10%-100%)
    """

    MATERIAL_CLASSES = ("PLA", "PETG", "ABS", "TPU")
    PATTERN_CLASSES = ("Rectilinear", "Grid", "Gyroid", "Concentric")

    def __init__(
        self,
        in_channels: int = 3,
        num_materials: int = 4,
        num_patterns: int = 4,
        backbone: str = "resnet34",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone == "resnet18":
            net = tv_models.resnet18(weights=None)
        else:
            net = tv_models.resnet34(weights=None)

        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net

        self.shared_proj = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_material = nn.Linear(256, num_materials)
        self.head_pattern = nn.Linear(256, num_patterns)
        self.head_infill = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        emb = self.shared_proj(feat)
        return {
            "embedding": emb,
            "material_logits": self.head_material(emb),
            "pattern_logits": self.head_pattern(emb),
            "infill_pred": self.head_infill(emb),
        }


def extract_physics_features(arr: np.ndarray, border_px: int = 6) -> dict[str, float]:
    """Extract interpretable physical attenuation & texture statistics from raw THz CSV.

    Used for small-sample Cohort B feasibility audits (N=10-40) before full CNN
    cross-validation.
    """
    h, w = arr.shape
    b = min(border_px, max(2, min(h, w) // 8))
    core = arr[b : h - b, b : w - b].astype(np.float64)

    p10, p25, p50, p75, p90 = np.percentile(core, [10.0, 25.0, 50.0, 75.0, 90.0])
    # Horizontal vs vertical texture energy (relevant once multi-pattern scans arrive)
    dx = np.abs(np.diff(core, axis=1)).mean()
    dy = np.abs(np.diff(core, axis=0)).mean()
    anisotropy = float((dx - dy) / (dx + dy + 1e-6))

    return {
        "core_mean": float(np.mean(core)),
        "core_std": float(np.std(core)),
        "core_median": float(p50),
        "core_iqr": float(p75 - p25),
        "core_p10": float(p10),
        "core_p90": float(p90),
        "positive_fraction": float((core > 0).mean()),
        "directional_anisotropy": anisotropy,
    }
