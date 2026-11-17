"""Physics-calibrated loss functions for sparse THz defects and multi-head characterization."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    """Compound Focal + Soft Dice loss for imbalanced subsurface defect segmentation.

    Defects in 40x40 mm THz scans typically occupy < 5-10% of image pixels.
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0,
        alpha: float = 0.75,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 1, H, W), targets: (B, 1, H, W) in [0, 1]
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal = (alpha_t * ((1.0 - pt) ** self.gamma) * bce).mean()

        # Soft Dice over batch
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denom + self.smooth)).mean()

        return self.focal_weight * focal + self.dice_weight * dice


class MultiTaskCharacterizerLoss(nn.Module):
    """Weighted multi-task loss for Model 1 (Material + Pattern + Infill %)."""

    def __init__(
        self,
        w_material: float = 1.0,
        w_pattern: float = 1.0,
        w_infill: float = 2.0,
    ) -> None:
        super().__init__()
        self.w_material = w_material
        self.w_pattern = w_pattern
        self.w_infill = w_infill
        self.ce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        material_target: torch.Tensor,
        pattern_target: torch.Tensor,
        infill_target: torch.Tensor,
    ) -> torch.Tensor:
        loss_mat = self.ce(outputs["material_logits"], material_target.long())
        loss_pat = self.ce(outputs["pattern_logits"], pattern_target.long())
        loss_inf = self.smooth_l1(outputs["infill_pred"].squeeze(-1), infill_target.float())
        return self.w_material * loss_mat + self.w_pattern * loss_pat + self.w_infill * loss_inf
