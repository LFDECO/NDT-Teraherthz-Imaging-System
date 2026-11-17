"""Physics-consistent augmentations and synthetic beam-blurred defect injection."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def inject_synthetic_defect(
    raw_scan: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    border_px: int = 9,
    shape_type: str = "circle",
    polarity: str = "dip",
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject a realistic beam-blurred blind/buried defect into a clean THz scan.

    Uses the measured 3-4 mm Rayleigh beam PSF (Gaussian sigma ~1.8 px on 1 mm/px grid)
    to generate soft diffraction-blurred boundaries instead of unrealistic sharp disks.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = raw_scan.shape
    b = min(border_px, max(5, min(h, w) // 5))
    cy = int(rng.integers(b + 3, max(b + 4, h - b - 3)))
    cx = int(rng.integers(b + 3, max(b + 4, w - b - 3)))
    radius = float(rng.uniform(1.8, 3.6))

    yy, xx = np.ogrid[:h, :w]
    if shape_type == "square":
        ideal_mask = ((np.abs(yy - cy) <= radius) & (np.abs(xx - cx) <= radius)).astype(np.float32)
    else:
        ideal_mask = (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2).astype(np.float32)

    # Apply THz beam-waist diffraction blur
    blurred_profile = gaussian_filter(ideal_mask, sigma=1.65)
    if blurred_profile.max() > 0:
        blurred_profile /= blurred_profile.max()

    core = raw_scan[b : h - b, b : w - b]
    iqr = max(float(np.percentile(core, 75.0) - np.percentile(core, 25.0)), float(np.std(core)), 0.02)
    amplitude = float(rng.uniform(1.8, 3.2)) * iqr

    out = raw_scan.astype(np.float32).copy()
    if polarity == "dip":
        out = out - amplitude * blurred_profile
    else:
        out = out + amplitude * blurred_profile

    target_mask = (blurred_profile >= 0.45).astype(np.float32)
    return out, target_mask


class PhysicsSafeAugmentor:
    """Applies only transformations that preserve physical millimeter calibration.

    Allowed:
      - Orthogonal 90-degree rotations (invariance to linear THz emitter polarization)
      - Horizontal and vertical flips
      - Sub-pixel / +-1 px shift (matches observed cross-frequency misregistration)
      - Multiplicative brightness gain in [0.90, 1.10] (matches observed 5-11% nuisance shift)
      - Additive Gaussian detector noise
    """

    def __init__(self, seed: int = 42, shift_px: int = 1, noise_std: float = 0.025) -> None:
        self.rng = np.random.default_rng(seed)
        self.shift_px = shift_px
        self.noise_std = noise_std

    def __call__(self, image_chw: np.ndarray, mask_hw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = image_chw.copy()
        msk = mask_hw.copy()

        # 1. Orthogonal 90-degree rotation
        k = int(self.rng.integers(0, 4))
        if k > 0:
            img = np.rot90(img, k=k, axes=(1, 2)).copy()
            msk = np.rot90(msk, k=k, axes=(0, 1)).copy()

        # 2. Horizontal / vertical flips
        if self.rng.random() < 0.5:
            img = np.flip(img, axis=2).copy()
            msk = np.flip(msk, axis=1).copy()
        if self.rng.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=0).copy()

        # 3. +-1 px shift matching THz stage misregistration
        if self.shift_px > 0:
            dy = int(self.rng.integers(-self.shift_px, self.shift_px + 1))
            dx = int(self.rng.integers(-self.shift_px, self.shift_px + 1))
            if dy != 0 or dx != 0:
                img = np.roll(img, shift=(dy, dx), axis=(1, 2))
                msk = np.roll(msk, shift=(dy, dx), axis=(0, 1))

        # 4. Brightness gain jitter on intensity channel + mild detector noise
        gain = float(self.rng.uniform(0.90, 1.10))
        img[0] = np.clip(img[0] * gain, 0.0, 1.0)
        if self.noise_std > 0:
            noise = self.rng.normal(0.0, self.noise_std, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        return img.astype(np.float32), (msk >= 0.5).astype(np.float32)
