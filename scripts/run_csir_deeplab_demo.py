"""CSIR Meeting Demo: DeepLabV3+ RGB Explainer + THz-Adapted LOSO Defect Localization."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrology import build_inspection_report, extract_defect_metrology, save_inspection_report
from src.models import DeepLabV3Plus, FocalDiceLoss, SmallUNet
from src.preprocessing import (
    PhysicsSafeAugmentor,
    build_physics_tensor,
    extract_reference_mask,
    inject_synthetic_defect,
    load_all_scans,
)
from src.visualization import (
    plot_deeplabv3plus_rgb_demo,
    plot_thz_deeplabv3plus_activations,
    plot_thz_localization_gallery,
)

_DEFECT_LABEL_MAP = {
    "1C": "1 Circular Void",
    "1S": "1 Square Void",
    "3C": "3x3 Circular Array",
    "3S": "3x3 Square Array",
    "ND": "No Defect Control",
}


def set_deterministic(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_rgb_demo_polymer_specimen(size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a high-resolution RGB polymer inspection specimen with blind & buried defects."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # Polymer plate texture (subtle FDM diagonal infill raster + warm PLA amber-ivory tint)
    infill_grid = 0.035 * np.sin(0.55 * (xx + yy)) + 0.025 * np.sin(0.55 * (xx - yy))
    base = np.zeros((size, size, 3), dtype=np.float32)
    base[..., 0] = 0.78 + infill_grid - 0.04 * (yy / size)
    base[..., 1] = 0.68 + infill_grid - 0.03 * (xx / size)
    base[..., 2] = 0.56 + infill_grid * 0.7

    # Dark outer stage frame around 40x40 mm specimen
    border = 18
    in_plate = (yy >= border) & (yy < size - border) & (xx >= border) & (xx < size - border)
    base[~in_plate] = np.array([0.14, 0.16, 0.20], dtype=np.float32)

    mask = np.zeros((size, size), dtype=np.float32)

    # Defect 1: Circular blind void (dark shadowed cavity)
    c1 = ((yy - 88) ** 2 + (xx - 92) ** 2) <= 18.0**2
    base[c1] *= np.array([0.38, 0.36, 0.42], dtype=np.float32)
    mask[c1] = 1.0

    # Defect 2: Rectangular blind pocket
    c2 = (np.abs(yy - 165) <= 15) & (np.abs(xx - 168) <= 18)
    base[c2] *= np.array([0.42, 0.40, 0.45], dtype=np.float32)
    mask[c2] = 1.0

    # Defect 3: Buried Layer-3 cold-joint delamination seam (subsurface diffuse streak)
    seam_dist = np.abs((yy - 122) - 0.35 * (xx - 150))
    c3 = (seam_dist <= 6.5) & (xx >= 120) & (xx <= 210)
    blur_seam = cv2.GaussianBlur(c3.astype(np.float32), (11, 11), 3.0)
    base[..., 0] = np.clip(base[..., 0] - 0.28 * blur_seam, 0.0, 1.0)
    base[..., 1] = np.clip(base[..., 1] - 0.30 * blur_seam, 0.0, 1.0)
    base[..., 2] = np.clip(base[..., 2] - 0.22 * blur_seam, 0.0, 1.0)
    mask[blur_seam >= 0.45] = 1.0

    rgb = np.clip(base, 0.0, 1.0)
    return rgb, mask


def run_part_a_rgb_demo(cfg: dict, out_fig_path: Path, device: torch.device) -> Tuple[dict, Dict[str, Any]]:
    """Train & run DeepLabV3+ in RGB Demo Mode and export architecture walkthrough figure."""
    rgb_cfg = cfg["model2_deeplab_rgb_demo"]
    size = int(rgb_cfg["input_size"][0])
    rgb_img, gt_mask = make_rgb_demo_polymer_specimen(size=size)

    model = DeepLabV3Plus(
        in_channels=int(rgb_cfg["in_channels"]),
        num_classes=int(rgb_cfg["num_classes"]),
        backbone=str(rgb_cfg["backbone"]),
        output_stride=int(rgb_cfg["output_stride"]),
        aspp_rates=tuple(rgb_cfg["aspp_rates"]),
    ).to(device)
    criterion = FocalDiceLoss(focal_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # Create mini augmented batch of the specimen so BatchNorm & ASPP converge cleanly
    rng = np.random.default_rng(7)
    aug_imgs, aug_msks = [], []
    base_chw = np.transpose(rgb_img, (2, 0, 1))
    for i in range(8):
        k = i % 4
        im_r = np.rot90(base_chw, k=k, axes=(1, 2)).copy()
        ms_r = np.rot90(gt_mask, k=k, axes=(0, 1)).copy()
        gain = float(rng.uniform(0.92, 1.08))
        noise = rng.normal(0.0, 0.015, size=im_r.shape).astype(np.float32)
        im_r = np.clip(im_r * gain + noise, 0.0, 1.0)
        aug_imgs.append(im_r)
        aug_msks.append(ms_r)

    x_batch = torch.from_numpy(np.stack(aug_imgs)).float().to(device)
    y_batch = torch.from_numpy(np.stack(aug_msks)).unsqueeze(1).float().to(device)

    model.train()
    for _ in range(36):
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        x_eval = torch.from_numpy(base_chw).unsqueeze(0).float().to(device)
        intermediates = model.forward_with_intermediates(x_eval)
        probs = intermediates["probs"][0, 0].cpu().numpy()

    pred_bin = (probs >= 0.5).astype(np.float32)
    inter = float((pred_bin * gt_mask).sum())
    union = float(((pred_bin + gt_mask) > 0).sum())
    iou = inter / max(union, 1e-6)
    dice = (2.0 * inter) / max(float(pred_bin.sum() + gt_mask.sum()), 1e-6)

    plot_deeplabv3plus_rgb_demo(rgb_img, gt_mask, intermediates, out_fig_path)
    print(f"[INFO] RGB DeepLabV3+ Demo -> IoU={iou:.3f}, Dice={dice:.3f} | Saved {out_fig_path}")
    rgb_cache = {
        "rgb_img": rgb_img,
        "gt_mask": gt_mask,
        "low_level_feat": intermediates["low_level_feat"].detach().cpu().numpy(),
        "encoder_high_feat": intermediates["encoder_high_feat"].detach().cpu().numpy(),
        "decoder_fused": intermediates["decoder_fused"].detach().cpu().numpy(),
        "probs": intermediates["probs"].detach().cpu().numpy(),
        "aspp_keys": list(intermediates["aspp_branches"].keys()),
        "aspp_branches": {k: v.detach().cpu().numpy() for k, v in intermediates["aspp_branches"].items()},
    }
    return {"rgb_demo_iou": round(iou, 4), "rgb_demo_dice": round(dice, 4)}, rgb_cache


def _upsample_pair(img_chw: np.ndarray, mask_hw: np.ndarray, size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    t_img = torch.from_numpy(img_chw).unsqueeze(0).float()
    t_msk = torch.from_numpy(mask_hw).unsqueeze(0).unsqueeze(0).float()
    up_img = F.interpolate(t_img, size=(size, size), mode="bicubic", align_corners=False)
    up_img = torch.clamp(up_img, 0.0, 1.0).squeeze(0).numpy()
    up_msk = F.interpolate(t_msk, size=(size, size), mode="bilinear", align_corners=False)
    up_msk = (up_msk.squeeze(0).squeeze(0).numpy() >= 0.5).astype(np.float32)
    return up_img, up_msk


def _build_fold_training_tensors(
    train_records: list,
    border_px: int,
    upsample_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create physics-safe augmented training samples strictly from training-fold scans."""
    augmentor = PhysicsSafeAugmentor(seed=seed, shift_px=1, noise_std=0.02)
    rng = np.random.default_rng(seed)

    xs, ys = [], []
    for rec in train_records:
        raw = rec.array
        base_tensor = build_physics_tensor(raw, border_px=border_px)
        base_mask = extract_reference_mask(raw, defect_code=rec.defect_code)

        # Original + 5 physics-safe augmented variants per real training scan
        u_img, u_msk = _upsample_pair(base_tensor, base_mask, size=upsample_size)
        xs.append(u_img)
        ys.append(u_msk)
        for _ in range(5):
            aug_i, aug_m = augmentor(base_tensor, base_mask)
            ui, um = _upsample_pair(aug_i, aug_m, size=upsample_size)
            xs.append(ui)
            ys.append(um)

        # If clean ND scan is in this training fold, inject both dip and peak synthetic defects
        if rec.defect_code.upper() == "ND":
            for pol in ("dip", "peak", "dip"):
                shape_t = "circle" if rng.random() < 0.6 else "square"
                syn_raw, syn_msk = inject_synthetic_defect(raw, rng=rng, shape_type=shape_t, polarity=pol)
                syn_tensor = build_physics_tensor(syn_raw, border_px=border_px)
                ui, um = _upsample_pair(syn_tensor, syn_msk, size=upsample_size)
                xs.append(ui)
                ys.append(um)
        else:
            # Also create clean negative sample from interior median background of training scan
            # so folds holding out ND still learn negative interior contrast rejection
            h, w = raw.shape
            clean_bg = np.full((h, w), float(np.median(raw[5:-5, 5:-5])), dtype=np.float32)
            clean_bg += rng.normal(0.0, 0.004, size=(h, w)).astype(np.float32)
            bg_tensor = build_physics_tensor(clean_bg, border_px=border_px)
            bg_mask = np.zeros((h, w), dtype=np.float32)
            ui, um = _upsample_pair(bg_tensor, bg_mask, size=upsample_size)
            xs.append(ui)
            ys.append(um)

    x_t = torch.from_numpy(np.stack(xs)).float()
    y_t = torch.from_numpy(np.stack(ys)).unsqueeze(1).float()
    return x_t, y_t


def _train_seg_model(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
    batch_size: int = 8,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    model.to(device)
    criterion = FocalDiceLoss(focal_weight=0.5, dice_weight=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n = x_train.size(0)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()


def _save_thz_plot_cache(
    cache_path: Path,
    rgb_cache: Dict[str, Any],
    activation_items: List[Dict[str, Any]],
    gallery_items: List[Dict[str, Any]],
) -> None:
    """Save computed arrays for rapid --replot-only regeneration without retraining."""
    payload: Dict[str, Any] = {
        "rgb_img": rgb_cache["rgb_img"],
        "rgb_gt_mask": rgb_cache["gt_mask"],
        "rgb_low_level_feat": rgb_cache["low_level_feat"],
        "rgb_encoder_high_feat": rgb_cache["encoder_high_feat"],
        "rgb_decoder_fused": rgb_cache["decoder_fused"],
        "rgb_probs": rgb_cache["probs"],
        "rgb_aspp_keys": np.array([json.dumps(rgb_cache["aspp_keys"])]),
    }
    for k, arr in rgb_cache["aspp_branches"].items():
        payload[f"rgb_aspp_{k}"] = arr

    act_meta = []
    for idx, item in enumerate(activation_items):
        inter = item["intermediates"]
        br = inter["aspp_branches"]
        act_meta.append(
            {
                "stem": item["stem"],
                "label": item["label"],
                "native_shape": list(item["native_shape"]),
            }
        )
        payload[f"act_{idx}_input_ch0"] = item["input_ch0"]
        payload[f"act_{idx}_probs"] = inter["probs"].detach().cpu().numpy()
        payload[f"act_{idx}_low"] = inter["low_level_feat"].detach().cpu().numpy()
        payload[f"act_{idx}_r1"] = br["rate_1"].detach().cpu().numpy()
        payload[f"act_{idx}_r2"] = br["rate_2"].detach().cpu().numpy()
        payload[f"act_{idx}_r4"] = br["rate_4"].detach().cpu().numpy()

    gal_meta = []
    for idx, item in enumerate(gallery_items):
        gal_meta.append(
            {
                "stem": item["stem"],
                "label": item["label"],
                "defects": item["defects"],
            }
        )
        payload[f"gal_{idx}_raw"] = item["raw"]
        payload[f"gal_{idx}_norm_ch0"] = item["norm_ch0"]
        payload[f"gal_{idx}_dog_ch1"] = item["dog_ch1"]
        payload[f"gal_{idx}_prob_native"] = item["prob_native"]
        payload[f"gal_{idx}_binary_native"] = item["binary_native"]

    payload["act_meta_json"] = np.array([json.dumps(act_meta)])
    payload["gal_meta_json"] = np.array([json.dumps(gal_meta)])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(f"[INFO] Saved THz demo plot cache -> {cache_path}")


def _replot_from_cache(cache_path: Path, figures_dir: Path) -> None:
    """Regenerate Figures 03, 04, and 05 in seconds using cached inference arrays."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Plot cache not found: {cache_path}. Run without --replot-only first.")

    data = np.load(cache_path, allow_pickle=False)
    aspp_keys = json.loads(str(data["rgb_aspp_keys"][0]))
    rgb_inter = {
        "low_level_feat": torch.from_numpy(data["rgb_low_level_feat"]),
        "encoder_high_feat": torch.from_numpy(data["rgb_encoder_high_feat"]),
        "decoder_fused": torch.from_numpy(data["rgb_decoder_fused"]),
        "probs": torch.from_numpy(data["rgb_probs"]),
        "aspp_branches": {k: torch.from_numpy(data[f"rgb_aspp_{k}"]) for k in aspp_keys},
    }
    fig3_path = figures_dir / "03_deeplabv3plus_rgb_architecture_demo.png"
    plot_deeplabv3plus_rgb_demo(data["rgb_img"], data["rgb_gt_mask"], rgb_inter, fig3_path)
    print(f"[INFO] Replotted Figure 03 -> {fig3_path}")

    act_meta = json.loads(str(data["act_meta_json"][0]))
    activation_items = []
    for idx, meta in enumerate(act_meta):
        activation_items.append(
            {
                "stem": meta["stem"],
                "native_shape": tuple(meta["native_shape"]),
                "label": meta["label"],
                "input_ch0": data[f"act_{idx}_input_ch0"],
                "intermediates": {
                    "probs": torch.from_numpy(data[f"act_{idx}_probs"]),
                    "low_level_feat": torch.from_numpy(data[f"act_{idx}_low"]),
                    "aspp_branches": {
                        "rate_1": torch.from_numpy(data[f"act_{idx}_r1"]),
                        "rate_2": torch.from_numpy(data[f"act_{idx}_r2"]),
                        "rate_4": torch.from_numpy(data[f"act_{idx}_r4"]),
                    },
                },
            }
        )
    fig4_path = figures_dir / "04_thz_deeplabv3plus_internal_activations.png"
    plot_thz_deeplabv3plus_activations(activation_items, fig4_path)
    print(f"[INFO] Replotted Figure 04 -> {fig4_path}")

    gal_meta = json.loads(str(data["gal_meta_json"][0]))
    gallery_items = []
    for idx, meta in enumerate(gal_meta):
        gallery_items.append(
            {
                "stem": meta["stem"],
                "label": meta["label"],
                "defects": meta["defects"],
                "raw": data[f"gal_{idx}_raw"],
                "norm_ch0": data[f"gal_{idx}_norm_ch0"],
                "dog_ch1": data[f"gal_{idx}_dog_ch1"],
                "prob_native": data[f"gal_{idx}_prob_native"],
                "binary_native": data[f"gal_{idx}_binary_native"],
            }
        )
    fig5_path = figures_dir / "05_thz_defect_localization_all_samples.png"
    plot_thz_localization_gallery(gallery_items, fig5_path)
    print(f"[INFO] Replotted Figure 05 -> {fig5_path}")


def run_part_b_thz_verification(cfg: dict, device: torch.device) -> Tuple[dict, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run Leave-One-Scan-Out comparison (THz-DeepLabV3+ vs SmallUNet) + activations + JSONs."""
    raw_dir = REPO_ROOT / cfg["data"]["raw_dir"]
    figures_dir = REPO_ROOT / "outputs" / "figures"
    reports_dir = REPO_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    thz_cfg = cfg["model2_deeplab_thz"]
    metro_cfg = cfg["metrology"]
    border_px = int(cfg["preprocessing"]["hot_strip_border_px"])
    up_size = int(thz_cfg["input_size"][0])
    epochs = int(thz_cfg["epochs_demo"])
    lr = float(thz_cfg["learning_rate"])

    records = load_all_scans(raw_dir)
    cohort_a = [r for r in records if r.cohort == "cohort_a_defect_geometry"]
    cohort_b_demo = [r for r in records if r.stem in {"PLA_0.4THz_100p", "ABS_0.4THz_100p"}]

    fold_metrics = []
    deeplab_predictions: Dict[str, dict] = {}
    activation_items = []

    t0 = time.time()
    for fold_idx, held_out in enumerate(cohort_a):
        train_recs = [r for r in cohort_a if r.stem != held_out.stem]
        x_tr, y_tr = _build_fold_training_tensors(
            train_recs, border_px=border_px, upsample_size=up_size, seed=100 + fold_idx
        )

        # 1. THz-Adapted DeepLabV3+
        dl_model = DeepLabV3Plus(
            in_channels=3,
            num_classes=1,
            backbone=str(thz_cfg["backbone"]),
            output_stride=int(thz_cfg["output_stride"]),
            aspp_rates=tuple(thz_cfg["aspp_rates"]),
        )
        _train_seg_model(dl_model, x_tr, y_tr, epochs=epochs, lr=lr, device=device, seed=200 + fold_idx)

        # 2. SmallUNet Baseline
        unet_model = SmallUNet(in_channels=3, num_classes=1, base_ch=24)
        _train_seg_model(unet_model, x_tr, y_tr, epochs=epochs, lr=lr, device=device, seed=200 + fold_idx)

        # Evaluate strictly on held-out real THz scan
        test_tensor = build_physics_tensor(held_out.array, border_px=border_px)
        gt_native = extract_reference_mask(held_out.array, defect_code=held_out.defect_code)
        up_test, _ = _upsample_pair(test_tensor, gt_native, size=up_size)
        x_test_t = torch.from_numpy(up_test).unsqueeze(0).float().to(device)

        dl_model.eval()
        unet_model.eval()
        with torch.no_grad():
            dl_inter = dl_model.forward_with_intermediates(x_test_t)
            dl_prob_up = dl_inter["probs"][0, 0].cpu().numpy()
            unet_prob_up = torch.sigmoid(unet_model(x_test_t))[0, 0].cpu().numpy()

        dl_prob_nat = cv2.resize(dl_prob_up, (held_out.shape[1], held_out.shape[0]), interpolation=cv2.INTER_LINEAR)
        unet_prob_nat = cv2.resize(unet_prob_up, (held_out.shape[1], held_out.shape[0]), interpolation=cv2.INTER_LINEAR)

        dl_bin, dl_defects = extract_defect_metrology(
            dl_prob_nat,
            native_shape=held_out.shape,
            pixel_pitch_mm=float(cfg["data"]["cohort_a_pixel_pitch_mm"]),
            prob_threshold=float(metro_cfg["prob_threshold"]),
            morph_kernel_size=int(metro_cfg["morph_kernel_size"]),
            min_area_mm2=float(metro_cfg["min_area_mm2"]),
            beam_width_mm=float(metro_cfg["beam_width_mm"]),
        )
        unet_bin, _ = extract_defect_metrology(
            unet_prob_nat,
            native_shape=held_out.shape,
            pixel_pitch_mm=float(cfg["data"]["cohort_a_pixel_pitch_mm"]),
            prob_threshold=float(metro_cfg["prob_threshold"]),
            morph_kernel_size=int(metro_cfg["morph_kernel_size"]),
            min_area_mm2=float(metro_cfg["min_area_mm2"]),
            beam_width_mm=float(metro_cfg["beam_width_mm"]),
        )

        # Compute metrics vs reference mask
        gt_flat = (gt_native > 0.5).ravel()
        if gt_flat.sum() > 0:
            dl_ap = float(average_precision_score(gt_flat, dl_prob_nat.ravel()))
            unet_ap = float(average_precision_score(gt_flat, unet_prob_nat.ravel()))
            dl_dice = float((2.0 * (dl_bin * gt_native).sum()) / max(dl_bin.sum() + gt_native.sum(), 1e-6))
            unet_dice = float((2.0 * (unet_bin * gt_native).sum()) / max(unet_bin.sum() + gt_native.sum(), 1e-6))
        else:
            dl_ap, unet_ap = None, None
            dl_dice = 1.0 if dl_bin.sum() == 0 else 0.0
            unet_dice = 1.0 if unet_bin.sum() == 0 else 0.0

        fold_row = {
            "held_out_scan": held_out.stem,
            "defect_code": held_out.defect_code,
            "frequency_ghz": held_out.frequency_ghz,
            "deeplabv3plus_ap": round(dl_ap, 4) if dl_ap is not None else None,
            "unet_ap": round(unet_ap, 4) if unet_ap is not None else None,
            "deeplabv3plus_dice": round(dl_dice, 4),
            "unet_dice": round(unet_dice, 4),
            "deeplabv3plus_false_alarm_px": int(dl_bin.sum()) if gt_flat.sum() == 0 else None,
            "unet_false_alarm_px": int(unet_bin.sum()) if gt_flat.sum() == 0 else None,
            "deeplab_defects_detected": len(dl_defects),
        }
        fold_metrics.append(fold_row)
        print(
            f"  [LOSO Fold {held_out.stem:22s}] DeepLab AP={str(fold_row['deeplabv3plus_ap']):>6s} "
            f"Dice={fold_row['deeplabv3plus_dice']:.3f} | UNet AP={str(fold_row['unet_ap']):>6s} "
            f"Dice={fold_row['unet_dice']:.3f} | Defects={len(dl_defects)}"
        )

        # Save JSON inspection report
        report = build_inspection_report(
            specimen_id=held_out.specimen_id,
            native_shape=held_out.shape,
            frequency_ghz=held_out.frequency_ghz,
            pixel_pitch_mm=float(cfg["data"]["cohort_a_pixel_pitch_mm"]),
            defects=dl_defects,
            material_type=held_out.material,
            infill_pattern=held_out.pattern,
            infill_density_pct=held_out.infill_pct,
        )
        save_inspection_report(report, reports_dir / f"inspection_report_{held_out.stem}.json")

        defect_desc = _DEFECT_LABEL_MAP.get(held_out.defect_code.upper(), held_out.defect_code)
        deeplab_predictions[held_out.stem] = {
            "stem": held_out.stem,
            "label": f"{held_out.stem}\n({defect_desc}, {held_out.frequency_ghz:.0f}GHz)",
            "raw": held_out.array,
            "norm_ch0": test_tensor[0],
            "dog_ch1": test_tensor[1],
            "prob_native": dl_prob_nat,
            "binary_native": dl_bin,
            "defects": dl_defects,
        }

        if held_out.stem in {"PLA_1C_450THz_100p", "PLA_3S_200THz_100p"}:
            activation_items.append(
                {
                    "stem": held_out.stem,
                    "native_shape": held_out.shape,
                    "label": f"{held_out.stem}\n({defect_desc}, {held_out.frequency_ghz:.0f}GHz)",
                    "input_ch0": up_test[0],
                    "intermediates": dl_inter,
                }
            )

    # Full Cohort-A model for cross-cohort Cohort B plate showcase
    x_full, y_full = _build_fold_training_tensors(cohort_a, border_px=border_px, upsample_size=up_size, seed=999)
    full_dl = DeepLabV3Plus(
        in_channels=3,
        num_classes=1,
        backbone=str(thz_cfg["backbone"]),
        output_stride=int(thz_cfg["output_stride"]),
        aspp_rates=tuple(thz_cfg["aspp_rates"]),
    )
    _train_seg_model(full_dl, x_full, y_full, epochs=epochs, lr=lr, device=device, seed=999)
    full_dl.eval()

    gallery_items = [deeplab_predictions[r.stem] for r in cohort_a]
    for plate in cohort_b_demo:
        p_tensor = build_physics_tensor(plate.array, border_px=border_px)
        p_up, _ = _upsample_pair(p_tensor, np.zeros(plate.shape, dtype=np.float32), size=up_size)
        with torch.no_grad():
            p_prob_up = torch.sigmoid(full_dl(torch.from_numpy(p_up).unsqueeze(0).float().to(device)))[0, 0].cpu().numpy()
        p_prob_nat = cv2.resize(p_prob_up, (plate.shape[1], plate.shape[0]), interpolation=cv2.INTER_LINEAR)
        p_bin, p_defects = extract_defect_metrology(
            p_prob_nat,
            native_shape=plate.shape,
            pixel_pitch_mm=float(cfg["data"]["cohort_b_assumed_pitch_mm"]),
            prob_threshold=float(metro_cfg["prob_threshold"]),
            morph_kernel_size=int(metro_cfg["morph_kernel_size"]),
            min_area_mm2=float(metro_cfg["min_area_mm2"]),
            beam_width_mm=float(metro_cfg["beam_width_mm"]),
        )
        gallery_items.append(
            {
                "stem": plate.stem,
                "label": f"{plate.stem}\n(Multi-Defect Plate: {plate.material} {int(plate.infill_pct)}%)",
                "raw": plate.array,
                "norm_ch0": p_tensor[0],
                "dog_ch1": p_tensor[1],
                "prob_native": p_prob_nat,
                "binary_native": p_bin,
                "defects": p_defects,
            }
        )

    fig4_path = figures_dir / "04_thz_deeplabv3plus_internal_activations.png"
    plot_thz_deeplabv3plus_activations(activation_items, fig4_path)
    print(f"[INFO] Saved Figure 04 -> {fig4_path}")

    fig5_path = figures_dir / "05_thz_defect_localization_all_samples.png"
    plot_thz_localization_gallery(gallery_items, fig5_path)
    print(f"[INFO] Saved Figure 05 -> {fig5_path}")

    valid_dl_ap = [m["deeplabv3plus_ap"] for m in fold_metrics if m["deeplabv3plus_ap"] is not None]
    valid_un_ap = [m["unet_ap"] for m in fold_metrics if m["unet_ap"] is not None]
    summary = {
        "evaluation_protocol": "Leave-One-Scan-Out (LOSO) across 5 Cohort A PLA scans with physics-safe training-fold augmentations",
        "caveat": "Reference masks in data/masks/ are physics-derived until CSIR-CSIO supplies physical CAD/caliper defect ground truth.",
        "runtime_seconds": round(time.time() - t0, 2),
        "mean_deeplabv3plus_ap": round(float(np.mean(valid_dl_ap)), 4),
        "mean_unet_ap": round(float(np.mean(valid_un_ap)), 4),
        "mean_deeplabv3plus_dice": round(float(np.mean([m["deeplabv3plus_dice"] for m in fold_metrics])), 4),
        "mean_unet_dice": round(float(np.mean([m["unet_dice"] for m in fold_metrics])), 4),
        "per_fold_metrics": fold_metrics,
    }
    return summary, activation_items, gallery_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSIR DeepLabV3+ RGB & THz verification.")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate Figures 03, 04, and 05 from cached inference arrays without retraining.",
    )
    args = parser.parse_args()

    figures_dir = REPO_ROOT / "outputs" / "figures"
    reports_dir = REPO_ROOT / "outputs" / "reports"
    cache_path = reports_dir / "thz_demo_plot_cache.npz"

    if args.replot_only:
        _replot_from_cache(cache_path, figures_dir)
        return

    set_deterministic(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running DeepLabV3+ CSIR demo on device: {device}")
    with open(REPO_ROOT / "configs" / "default_config.yaml", "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    fig3_path = figures_dir / "03_deeplabv3plus_rgb_architecture_demo.png"
    rgb_summary, rgb_cache = run_part_a_rgb_demo(cfg, fig3_path, device=device)

    thz_summary, activation_items, gallery_items = run_part_b_thz_verification(cfg, device=device)
    _save_thz_plot_cache(cache_path, rgb_cache, activation_items, gallery_items)

    combined = {"rgb_demo": rgb_summary, "thz_loso_verification": thz_summary}
    metrics_path = reports_dir / "deeplab_vs_unet_verification_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(combined, fp, indent=2)
    print(f"[INFO] Saved benchmark JSON -> {metrics_path}")


if __name__ == "__main__":
    main()
