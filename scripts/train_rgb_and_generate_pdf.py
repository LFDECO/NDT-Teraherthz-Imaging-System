"""Train DeepLabV3+ on 24 curated real RGB industrial defect images and build presentation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import DeepLabV3Plus, FocalDiceLoss
from src.visualization import (
    plot_real_rgb_deeplab_flow,
    plot_real_rgb_test_gallery,
)


def build_rgb_feature_tensor(rgb_u8: np.ndarray, size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Resize industrial RGB image to ``size x size`` and build 3-channel feature tensor.

    Channels:
      - Ch 0: CLAHE-normalized reflectance in [0, 1]
      - Ch 1: Fixed-scale Absolute Local DoG anomaly map in [0, 1] (clean tiles stay low)
      - Ch 2: Fixed-scale Sobel gradient magnitude in [0, 1]
    """
    rgb_res = cv2.resize(rgb_u8, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb_res, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch0 = clahe.apply(gray).astype(np.float32) / 255.0

    g_small = gaussian_filter(ch0, sigma=0.8)
    g_large = gaussian_filter(ch0, sigma=5.0)
    abs_dog = np.abs(g_small - g_large)
    # Fixed scale divisor avoids blowing up noise on defect-free MT_Free tiles
    ch1 = np.clip(abs_dog / 0.20, 0.0, 1.0).astype(np.float32)

    gx = cv2.Sobel(ch0, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ch0, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    ch2 = np.clip(grad / 1.10, 0.0, 1.0).astype(np.float32)

    tensor_chw = np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)
    rgb_disp = rgb_res.astype(np.float32) / 255.0
    return tensor_chw, rgb_disp


def resize_mask(mask_u8: np.ndarray, size: int = 128) -> np.ndarray:
    m_f = (mask_u8 > 127).astype(np.float32)
    m_res = cv2.resize(m_f, (size, size), interpolation=cv2.INTER_AREA)
    return (m_res >= 0.15).astype(np.float32)


def _save_rgb_plot_cache(
    cache_path: Path,
    flow_items: List[Dict[str, Any]],
    gallery_items: List[Dict[str, Any]],
) -> None:
    payload: Dict[str, Any] = {}
    flow_meta = []
    for idx, item in enumerate(flow_items):
        inter = item["intermediates"]
        br = inter["aspp_branches"]
        flow_meta.append({"label": item["label"]})
        payload[f"flow_{idx}_rgb"] = item["rgb"]
        payload[f"flow_{idx}_dog_ch1"] = item["dog_ch1"]
        payload[f"flow_{idx}_gt_mask"] = item["gt_mask"]
        payload[f"flow_{idx}_probs"] = inter["probs"].detach().cpu().numpy()
        payload[f"flow_{idx}_low"] = inter["low_level_feat"].detach().cpu().numpy()
        payload[f"flow_{idx}_r2"] = br["rate_2"].detach().cpu().numpy()
        payload[f"flow_{idx}_r4"] = br["rate_4"].detach().cpu().numpy()
        payload[f"flow_{idx}_r8"] = br["rate_8"].detach().cpu().numpy()
        payload[f"flow_{idx}_dec"] = inter["decoder_fused"].detach().cpu().numpy()

    gal_meta = []
    for idx, item in enumerate(gallery_items):
        gal_meta.append(
            {
                "label": item["label"],
                "centroids": item["centroids"],
                "gt_status": item["gt_status"],
                "pred_status": item["pred_status"],
                "ap": item["ap"],
                "dice": item["dice"],
            }
        )
        payload[f"gal_{idx}_rgb"] = item["rgb"]
        payload[f"gal_{idx}_gt_mask"] = item["gt_mask"]
        payload[f"gal_{idx}_probs"] = item["probs"]
        payload[f"gal_{idx}_binary"] = item["binary"]

    payload["flow_meta_json"] = np.array([json.dumps(flow_meta)])
    payload["gal_meta_json"] = np.array([json.dumps(gal_meta)])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(f"[INFO] Saved RGB demo plot cache -> {cache_path}", flush=True)


def _replot_rgb_from_cache(cache_path: Path, figures_dir: Path) -> None:
    if not cache_path.exists():
        raise FileNotFoundError(f"RGB plot cache not found: {cache_path}. Run without --replot-only first.")

    data = np.load(cache_path, allow_pickle=False)
    flow_meta = json.loads(str(data["flow_meta_json"][0]))
    flow_items = []
    for idx, meta in enumerate(flow_meta):
        flow_items.append(
            {
                "label": meta["label"],
                "rgb": data[f"flow_{idx}_rgb"],
                "dog_ch1": data[f"flow_{idx}_dog_ch1"],
                "gt_mask": data[f"flow_{idx}_gt_mask"],
                "intermediates": {
                    "probs": torch.from_numpy(data[f"flow_{idx}_probs"]),
                    "low_level_feat": torch.from_numpy(data[f"flow_{idx}_low"]),
                    "aspp_branches": {
                        "rate_2": torch.from_numpy(data[f"flow_{idx}_r2"]),
                        "rate_4": torch.from_numpy(data[f"flow_{idx}_r4"]),
                        "rate_8": torch.from_numpy(data[f"flow_{idx}_r8"]),
                    },
                    "decoder_fused": torch.from_numpy(data[f"flow_{idx}_dec"]),
                },
            }
        )
    fig7_path = figures_dir / "07_real_rgb_deeplabv3plus_architecture_flow.png"
    plot_real_rgb_deeplab_flow(flow_items, fig7_path)
    print(f"[INFO] Replotted Figure 07 -> {fig7_path}", flush=True)

    gal_meta = json.loads(str(data["gal_meta_json"][0]))
    gallery_items = []
    for idx, meta in enumerate(gal_meta):
        gallery_items.append(
            {
                "label": meta["label"],
                "rgb": data[f"gal_{idx}_rgb"],
                "gt_mask": data[f"gal_{idx}_gt_mask"],
                "probs": data[f"gal_{idx}_probs"],
                "binary": data[f"gal_{idx}_binary"],
                "centroids": [tuple(c) for c in meta["centroids"]],
                "gt_status": meta["gt_status"],
                "pred_status": meta["pred_status"],
                "ap": meta["ap"],
                "dice": meta["dice"],
            }
        )
    fig8_path = figures_dir / "08_real_rgb_heldout_test_samples_results.png"
    plot_real_rgb_test_gallery(gallery_items, fig8_path)
    print(f"[INFO] Replotted Figure 08 -> {fig8_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RGB DeepLabV3+ & build CSIR visuals.")
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate Figures 07 and 08 from cached inference arrays without retraining.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip building the presentation PDF (useful when running parallel training).",
    )
    args = parser.parse_args()

    figures_dir = REPO_ROOT / "outputs" / "figures"
    reports_dir = REPO_ROOT / "outputs" / "reports"
    cache_path = reports_dir / "rgb_demo_plot_cache.npz"
    manifest_path = REPO_ROOT / "data" / "rgb_defect_samples" / "rgb_manifest.csv"

    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest_rows = list(csv.DictReader(fp))

    if args.replot_only:
        _replot_rgb_from_cache(cache_path, figures_dir)
        if not args.skip_pdf:
            from scripts.build_csir_presentation_bundle import build_presentation_bundle

            build_presentation_bundle(REPO_ROOT)
        return

    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running Real RGB DeepLabV3+ training on device: {device}", flush=True)

    img_size = 128
    train_x, train_y = [], []
    test_records = []

    for row in manifest_rows:
        img_u8 = np.asarray(Image.open(REPO_ROOT / row["image_path"]).convert("RGB"))
        msk_u8 = np.asarray(Image.open(REPO_ROOT / row["mask_path"]).convert("L"))
        feat_chw, rgb_disp = build_rgb_feature_tensor(img_u8, size=img_size)
        msk_hw = resize_mask(msk_u8, size=img_size)

        if row["split"] == "train":
            # Deterministic physics-consistent geometric augmentations (5x)
            variants = [
                (feat_chw, msk_hw),
                (np.flip(feat_chw, axis=2).copy(), np.flip(msk_hw, axis=1).copy()),
                (np.flip(feat_chw, axis=1).copy(), np.flip(msk_hw, axis=0).copy()),
                (np.rot90(feat_chw, k=1, axes=(1, 2)).copy(), np.rot90(msk_hw, k=1, axes=(0, 1)).copy()),
                (np.rot90(feat_chw, k=2, axes=(1, 2)).copy(), np.rot90(msk_hw, k=2, axes=(0, 1)).copy()),
            ]
            # Balance clean MT_Free samples (2x) against 3 defective classes
            repeat_factor = 2 if int(row["has_defect"]) == 0 else 1
            for _ in range(repeat_factor):
                for vx, vy in variants:
                    train_x.append(vx)
                    train_y.append(vy)
        else:
            test_records.append(
                {
                    "row": row,
                    "feat_chw": feat_chw,
                    "rgb_disp": rgb_disp,
                    "gt_mask": msk_hw,
                }
            )

    x_tr = torch.from_numpy(np.stack(train_x)).float()
    y_tr = torch.from_numpy(np.stack(train_y)).unsqueeze(1).float()
    print(f"[INFO] Train tensors: {x_tr.shape} | Held-out test samples: {len(test_records)}", flush=True)

    model = DeepLabV3Plus(
        in_channels=3,
        num_classes=1,
        backbone="resnet18",
        output_stride=8,
        aspp_rates=(2, 4, 8),
    ).to(device)
    criterion = FocalDiceLoss(focal_weight=0.50, dice_weight=0.50, gamma=2.0, alpha=0.65)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    epochs = 28
    batch_size = 16
    rng = np.random.default_rng(42)
    n_train = x_tr.size(0)

    model.train()
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        ep_loss = 0.0
        steps = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue
            xb = x_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item())
            steps += 1
        if (ep + 1) % 6 == 0 or ep == 0:
            print(f"  [Epoch {ep+1:2d}/{epochs}] loss={ep_loss / max(steps, 1):.4f}", flush=True)

    # Evaluate on all 8 strictly held-out test samples
    model.eval()
    flow_items = []
    gallery_items = []
    per_sample_metrics = []
    flow_targets = {"MT_Blowhole", "MT_Crack"}
    seen_flow_cats = set()

    prob_thresh = 0.55
    for rec in test_records:
        row = rec["row"]
        gt = rec["gt_mask"]
        xt = torch.from_numpy(rec["feat_chw"]).unsqueeze(0).float().to(device)
        with torch.no_grad():
            inter = model.forward_with_intermediates(xt)
            probs = inter["probs"][0, 0].cpu().numpy()

        binary_raw = (probs >= prob_thresh).astype(np.uint8)
        # Filter tiny 1-5 px isolated noise speckles at 128x128
        num_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(binary_raw)
        clean_bin = np.zeros_like(binary_raw)
        centroids = []
        for lbl in range(1, num_lbl):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= 6:
                comp = (lbls == lbl).astype(np.uint8)
                clean_bin[comp > 0] = 1
                m = cv2.moments(comp)
                if m["m00"] > 0:
                    centroids.append((float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])))

        gt_bin = (gt >= 0.5).astype(np.float32)
        gt_has_defect = int(row["has_defect"]) == 1
        pred_has_defect = len(centroids) > 0

        if gt_bin.sum() > 0:
            ap = float(average_precision_score(gt_bin.ravel(), probs.ravel()))
            inter_px = float((clean_bin * gt_bin).sum())
            union_px = float(((clean_bin + gt_bin) > 0).sum())
            dice = (2.0 * inter_px) / max(float(clean_bin.sum() + gt_bin.sum()), 1e-6)
            iou = inter_px / max(union_px, 1e-6)
        else:
            ap = None
            dice = 1.0 if clean_bin.sum() == 0 else 0.0
            iou = 1.0 if clean_bin.sum() == 0 else 0.0

        gt_status = "FAIL" if gt_has_defect else "PASS"
        pred_status = "FAIL" if pred_has_defect else "PASS"

        per_sample_metrics.append(
            {
                "sample_id": row["sample_id"],
                "category": row["category"],
                "ground_truth_status": gt_status,
                "predicted_status": pred_status,
                "identified_correctly": gt_status == pred_status,
                "ap": round(ap, 4) if ap is not None else None,
                "dice": round(dice, 4),
                "iou": round(iou, 4),
                "max_prob": round(float(probs.max()), 4),
                "predicted_components": len(centroids),
                "predicted_fg_pixels": int(clean_bin.sum()),
            }
        )
        print(
            f"  [Test] {row['sample_id']:28s} | GT={gt_status} Pred={pred_status} | "
            f"AP={str(round(ap,3)) if ap is not None else 'N/A ':>5s} | Dice={dice:.3f} | Comps={len(centroids)}",
            flush=True,
        )

        gallery_items.append(
            {
                "label": f"{row['category']}\n{row['sample_id'].split('__', 1)[1]}",
                "rgb": rec["rgb_disp"],
                "gt_mask": gt,
                "probs": probs,
                "binary": clean_bin,
                "centroids": centroids,
                "gt_status": gt_status,
                "pred_status": pred_status,
                "ap": ap,
                "dice": dice,
            }
        )

        if row["category"] in flow_targets and row["category"] not in seen_flow_cats:
            seen_flow_cats.add(row["category"])
            flow_items.append(
                {
                    "label": f"{row['category']}\n({row['sample_id'].split('__', 1)[1]})",
                    "rgb": rec["rgb_disp"],
                    "dog_ch1": rec["feat_chw"][1],
                    "gt_mask": gt,
                    "intermediates": inter,
                }
            )

    fig7_path = figures_dir / "07_real_rgb_deeplabv3plus_architecture_flow.png"
    plot_real_rgb_deeplab_flow(flow_items, fig7_path)
    print(f"[INFO] Saved Figure 07 -> {fig7_path}", flush=True)

    fig8_path = figures_dir / "08_real_rgb_heldout_test_samples_results.png"
    plot_real_rgb_test_gallery(gallery_items, fig8_path)
    print(f"[INFO] Saved Figure 08 -> {fig8_path}", flush=True)

    _save_rgb_plot_cache(cache_path, flow_items, gallery_items)

    def_aps = [m["ap"] for m in per_sample_metrics if m["ap"] is not None]
    def_dices = [m["dice"] for m in per_sample_metrics if m["ground_truth_status"] == "FAIL"]
    def_ious = [m["iou"] for m in per_sample_metrics if m["ground_truth_status"] == "FAIL"]
    clean_fp = sum(m["predicted_fg_pixels"] for m in per_sample_metrics if m["ground_truth_status"] == "PASS")
    correct_id = sum(1 for m in per_sample_metrics if m["identified_correctly"])

    rgb_metrics = {
        "dataset": "Curated Magnetic-Tile-Defect Industrial RGB Benchmark (24 samples: 16 train / 8 held-out test)",
        "model": "DeepLabV3Plus (ResNet-18, Output Stride=8, ASPP r=[2,4,8], 3-Ch CLAHE+DoG+Sobel)",
        "runtime_seconds": round(time.time() - t0, 2),
        "summary": {
            "total_test_samples": len(per_sample_metrics),
            "correct_identifications": correct_id,
            "identification_accuracy": round(correct_id / max(len(per_sample_metrics), 1), 4),
            "mean_defective_ap": round(float(np.mean(def_aps)), 4),
            "mean_defective_dice": round(float(np.mean(def_dices)), 4),
            "mean_defective_iou": round(float(np.mean(def_ious)), 4),
            "clean_false_alarm_pixels": int(clean_fp),
        },
        "per_sample_test_metrics": per_sample_metrics,
    }
    rgb_json_path = reports_dir / "real_rgb_deeplab_metrics.json"
    with open(rgb_json_path, "w", encoding="utf-8") as fp:
        json.dump(rgb_metrics, fp, indent=2)
    print(f"[INFO] Saved RGB metrics -> {rgb_json_path}", flush=True)

    if not args.skip_pdf:
        from scripts.build_csir_presentation_bundle import build_presentation_bundle

        build_presentation_bundle(REPO_ROOT)


if __name__ == "__main__":
    main()
