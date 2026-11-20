"""Single source of truth for CSIR speaker scripts, DOCX side-by-side guide & MD guide."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor


_NAVY_HEX = "0F2942"
_TEAL_HEX = "136F8A"
_LIGHT_HEX = "F4F7FA"
_TINT_HEX = "EBF3F9"
_ALT_HEX = "F8FAFC"
_BORDER_HEX = "CBD5E1"

_NAVY_RGB = RGBColor(0x0F, 0x29, 0x42)
_TEAL_RGB = RGBColor(0x13, 0x6F, 0x8A)
_SLATE_RGB = RGBColor(0x1E, 0x29, 0x3B)
_MUTED_RGB = RGBColor(0x47, 0x55, 0x69)
_WHITE_RGB = RGBColor(0xFF, 0xFF, 0xFF)


def _format_num(val: Any, fmt: str = ".3f", fallback: str = "N/A") -> str:
    if val is None:
        return fallback
    return format(float(val), fmt)


def get_csir_speaker_content(
    rgb_metrics: Dict[str, Any],
    thz_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build synchronized presentation speaker guide content from live benchmark metrics."""
    rgb_sum = rgb_metrics["summary"]
    thz_loso = thz_metrics["thz_loso_verification"]
    rgb_demo = thz_metrics["rgb_demo"]

    rgb_id_pct = rgb_sum["identification_accuracy"] * 100.0
    rgb_correct = rgb_sum["correct_identifications"]
    rgb_total = rgb_sum["total_test_samples"]
    rgb_ap = rgb_sum["mean_defective_ap"]
    rgb_dice = rgb_sum["mean_defective_dice"]
    rgb_iou = rgb_sum["mean_defective_iou"]
    rgb_fp = rgb_sum["clean_false_alarm_pixels"]

    thz_dl_ap = thz_loso["mean_deeplabv3plus_ap"]
    thz_un_ap = thz_loso["mean_unet_ap"]
    thz_dl_dice = thz_loso["mean_deeplabv3plus_dice"]
    thz_un_dice = thz_loso["mean_unet_dice"]

    fold_map = {f["held_out_scan"]: f for f in thz_loso["per_fold_metrics"]}
    f_1c = fold_map.get("PLA_1C_450THz_100p", {})
    f_1s = fold_map.get("PLA_1S_325THz_100p", {})
    f_3c = fold_map.get("PLA_3C_135THz_100p", {})
    f_3s = fold_map.get("PLA_3S_200THz_100p", {})
    f_nd = fold_map.get("PLA_ND_300THz_100p", {})

    cheat_sheet = {
        "meeting_title": "SRM University & CSIR-CSIO Terahertz NDT Diagnostic Review",
        "presenters": "SRM University Research Team -> CSIR-CSIO Research Mentors",
        "total_duration": "~13 minutes (10 slides, ~60-90 sec/slide) + Q&A",
        "dataset_breakdown": [
            ["No Defect (Clean Control)", "1 scan (PLA_ND_300THz_100p)", "PLA | 100% infill | 300 GHz | 40x40 mm (46x46 px)", "False-alarm / PASS control"],
            ["Single Defect (1 Void)", "2 scans (PLA_1C, PLA_1S)", "PLA | 100% infill | 450 & 325 GHz | 40x40 mm", "1 circular void (1C) & 1 square void (1S)"],
            ["Uniform 3x3 Arrays", "2 scans (PLA_3C, PLA_3S)", "PLA | 100% infill | 135 & 200 GHz | 39x39 & 38x38 mm", "3x3 circular (3C) & 3x3 square (3S) arrays"],
            ["Multi-Defect Mixed Plates", "10 scans (Cohort B)", "4 materials (PLA, PETG, ABS, TPU) | Same pattern | 15, 45, 75, 100%", "Multi-band anomalies, circular voids, edge features"],
        ],
        "headline_metrics": [
            f"Real Industrial RGB Test (N=8 unseen): {rgb_id_pct:.1f}% ({rgb_correct}/{rgb_total}) ID accuracy | Mean AP = {rgb_ap:.3f} | Mean Dice = {rgb_dice:.3f} | Mean IoU = {rgb_iou:.3f}",
            f"Real CSIR THz LOSO (N=5 Cohort A): THz-DeepLabV3+ Mean AP = {thz_dl_ap:.3f} vs SmallUNet {thz_un_ap:.3f} | PLA_1S AP = {_format_num(f_1s.get('deeplabv3plus_ap'))}, Dice = {_format_num(f_1s.get('deeplabv3plus_dice'))} | PLA_ND = {f_nd.get('deeplabv3plus_false_alarm_px', 0)} false-alarm px",
            f"Synthetic RGB Polymer Walkthrough: IoU = {rgb_demo['rgb_demo_iou']:.3f} | Dice = {rgb_demo['rgb_demo_dice']:.3f}",
            "Visual Standard Applied: CSIR physical Distance (mm) scales, explicit colorbars, and interior-anchored high-saturation jet (p4-p96) across all THz plots.",
        ],
    }

    slides: List[Dict[str, Any]] = [
        {
            "slide_num": 1,
            "title": "Terahertz Non-Destructive Testing (NDT) via DeepLabV3+",
            "duration_sec": 75,
            "slide_goal": "Frame the standalone AI scope, physical THz constraints, why we split into two decoupled models, and what the 15-scan dataset contains.",
            "point_to_on_screen": [
                "Top-left box: 'Standalone AI models & OpenCV metrology' (no printer/hardware driver coupling).",
                "Bottom-left box: Model 1 (bulk characterization) vs Model 2 DeepLabV3+ (pixel defect localization).",
                "Right flowchart: Trace Layer 1 conditioning -> parallel Model 1 & Model 2 -> Layer 5 OpenCV mm metrology -> JSON report.",
            ],
            "read_off_script": [
                "Good morning everyone. Today we are presenting our standalone DeepLabV3+ architecture and defect localization results across both an industrial RGB benchmark and the 15 real Terahertz C-scans from CSIR-CSIO.",
                "First, to align with our agreed scoping boundary, this pipeline is strictly standalone software—signal conditioning, predictive AI models, and deterministic OpenCV metrology—without modifying any printer or scanner hardware drivers. Physically, our 0.135 to 0.450 THz reflection scans have a 3 to 4 mm diffraction beam waist, which blurs internal boundaries, while the 15 CSV scans span 4 polymer materials—PLA, PETG, ABS, and TPU—printed with the same infill pattern across 15% to 100% infill, containing clean controls, single voids, 3x3 arrays, and multi-defect plates.",
                "Because we only have 15 scans today, training one single multi-task CNN for both bulk polymer properties and fine pixel masks creates severe gradient conflict. So as shown on the right, after Layer 1 cleans border hot-strips and extracts physics channels, we decouple into Model 1 for bulk characterization and Model 2—our THz-adapted DeepLabV3+—dedicated purely to localizing defects, feeding OpenCV metrology for millimeter PASS/FAIL reports.",
            ],
            "technical_backing": [
                "Dataset audit: 15 CSV scans (5 Cohort A PLA 100% + 10 Cohort B multi-material/multi-infill multi-defect plates).",
                "Conditioning tensor: 3 channels [Ch0 Robust Interior Norm, Ch1 |Local DoG|, Ch2 Sobel Edge] (src/preprocessing/normalization.py).",
                "Outputs: PASS/FAIL status, centroid (X, Y) mm, apparent area mm^2, beam-corrected diameter d_true = sqrt(max(0.5^2, d_app^2 - d_beam^2)).",
            ],
            "transition_line": "Now let us look inside DeepLabV3+ on Slide 2 to see how it works and the critical change we had to make so it does not break on small 46x46 THz grids.",
        },
        {
            "slide_num": 2,
            "title": "DeepLabV3+ Architecture & Critical Adaptation for THz C-Scans",
            "duration_sec": 90,
            "slide_goal": "Explain DeepLabV3+ dataflow (ResNet + ASPP + 1/4 skip decoder) and prove mathematically why textbook Stride-16 fails on raw THz scans.",
            "point_to_on_screen": [
                "Top diagram: Follow Input -> ResNet-18 -> parallel ASPP branches + 1/4 low-level skip -> Decoder Fusion -> Pixel Mask.",
                "Bottom-left red box: Highlight floor(46 / 16) = 2x2 or 3x3 bottleneck collapse.",
                "Bottom-right comparison table: Contrast Textbook RGB (OS=16, r=6,12,18, ResNet-101) vs THz-Adapted (OS=8, r=1,2,4, ResNet-18).",
            ],
            "read_off_script": [
                "Slide 2 explains why DeepLabV3+ is so effective for defect localization, and why we could not just plug in textbook RGB DeepLabV3+ out of the box.",
                "In the top diagram, DeepLabV3+ has two complementary paths. The upper path extracts semantic context through Atrous Spatial Pyramid Pooling—ASPP—running parallel dilated convolutions at multiple receptive-field scales plus global pooling. Meanwhile, the lower 1/4-resolution skip connection brings sharp spatial edges directly from Layer 1 of ResNet-18 into the decoder so defect boundaries do not get washed out.",
                "However, look at the red box on the bottom left: our raw CSIR THz grids are only 38x38 to 46x46 pixels. Standard DeepLabV3+ uses ResNet-101 with Output Stride 16 and ASPP dilation rates 6, 12, and 18. Dividing 46 by 16 collapses the bottleneck feature map to just 2x2 or 3x3 pixels—where dilation 12 and 18 sample completely outside the image borders! Instead, as summarized in the table on the right, we bicubic-upsample inputs to 128x128, switch to a lightweight dilated ResNet-18 at Output Stride 8 giving a 16x16 bottleneck, and scale ASPP rates to 1, 2, and 4—matching the 3 to 4 mm beam waist and 3x3 array spacing.",
            ],
            "technical_backing": [
                "Implemented in src/models/model2_deeplabv3plus.py with DilatedBasicBlock supporting arbitrary OS=8 or OS=16.",
                "Low-level 1/4 skip projects 64 channels -> 48 channels at 32x32 resolution (per Chen et al. 2018).",
                "Parameter reduction: ~11M params (ResNet-18 backbone) vs ~44M (ResNet-101), preventing small-data memorization.",
            ],
            "transition_line": "On Slide 3, we summarize why this DeepLabV3+ setup beats alternative segmentation paradigms and how we train it safely on sparse defects.",
        },
        {
            "slide_num": 3,
            "title": "Architectural Rationale: Why This Design Over All Alternatives",
            "duration_sec": 75,
            "slide_goal": "Justify DeepLabV3+ over YOLO, PatchCore, U-Net, ViTs, and monolithic CNNs, plus Focal+Dice loss and physics-safe augmentations.",
            "point_to_on_screen": [
                "Top comparison matrix: Contrast Primary Pipeline against YOLO, PatchCore, SmallUNet, and SegFormer/Swin.",
                "Bottom-left teal box: Explain Focal + Soft Dice for 0.2%-2.0% defect pixel sparsity.",
                "Bottom-right orange box: Highlight allowed 90-deg rotations/flips/+/-1 px shifts vs forbidden elastic warps.",
            ],
            "read_off_script": [
                "Before looking at our test images, Slide 3 answers why we chose DeepLabV3+ over every other candidate architecture.",
                "In the table at the top, bounding-box detectors like YOLOv8-seg struggle when a subsurface delamination is blurry without crisp rectangular boxes. Unsupervised methods like PatchCore give uncalibrated anomaly scores without closed contours needed for millimeter area metrology. Standard U-Net is our direct ablation baseline—it detects central blobs well, but lacks parallel multi-rate ASPP kernels to separate background infill texture from defects. Heavy Vision Transformers like SegFormer or Swin overfit rapidly when fewer than 200 scans exist.",
                "At the bottom left, because real defects occupy only 0.2% to 2% of pixels—for example, 0.19% in cracks and 1.47% in PLA_1S—standard cross-entropy collapses to predicting all background. So we train with a compound Focal plus Soft Dice loss. And on the bottom right, we strictly forbid elastic warps or arbitrary shearing that distort millimeter calibration; we only allow 90-degree rotations, flips, plus-or-minus 1-pixel shifts, and mild gain jitter under strict held-out splits.",
            ],
            "technical_backing": [
                "Loss: FocalDiceLoss(focal_weight=0.5, dice_weight=0.5, gamma=2.0) in src/models/losses.py.",
                "Augmentation: PhysicsSafeAugmentor(shift_px=1, noise_std=0.02) in src/preprocessing/augmentation.py.",
                "SmallUNet baseline retained in src/models/model2_unet_baseline.py for head-to-head THz LOSO verification.",
            ],
            "transition_line": "To prove DeepLabV3+ generalizes on real pixel ground-truth defects before moving to small THz sample counts, let us inspect our first benchmark on Slide 4.",
        },
        {
            "slide_num": 4,
            "title": "Benchmark 1: Curated Real Industrial RGB Defect Dataset (24 Samples)",
            "duration_sec": 70,
            "slide_goal": "Introduce the 24-image Magnetic-Tile industrial RGB benchmark and its strict 16-train / 8-unseen-test split.",
            "point_to_on_screen": [
                "Top-left box: Note 16 training / 8 strictly held-out test samples across 4 real defect categories.",
                "Bottom-left table: Map MT_Blowhole, MT_Crack, MT_Break, and MT_Free directly to Polymer NDT defect analogies.",
                "Right 2x4 grid: Show the raw industrial test image next to its pixel-accurate ground-truth mask.",
            ],
            "read_off_script": [
                "Because our current CSIR Cohort A batch only has 5 defect-geometry scans without CAD caliper masks yet, we first validated our exact DeepLabV3+ implementation on 24 real industrial defect images from the public Magnetic-Tile-Defect benchmark.",
                "We split these 24 samples strictly into 16 training images—4 per category—and 8 completely unseen held-out test images, 2 per category. As listed in the table on the bottom left, these four categories directly mirror our THz polymer NDT challenge: MT_Blowhole represents tiny localized circular voids occupying just 0.26% to 0.48% of pixels; MT_Crack tests hairline linear seams down to 0.19%; MT_Break covers larger pocket and edge flaws; and MT_Free serves as our clean defect-free control.",
                "On the right, you can see one held-out test example per category alongside its exact binary ground-truth mask. Notice how faint the defects are against the brushed metallic background texture—just like subsurface voids inside FDM infill texture.",
            ],
            "technical_backing": [
                "Manifest: data/rgb_defect_samples/rgb_manifest.csv (24 images + verified binary PNG masks).",
                "Input conditioning: 3 channels [CLAHE Gray, Fixed-Scale |Local DoG|, Fixed-Scale Sobel] resized to 128x128.",
                "Fixed-scale DoG denominator (0.20) ensures defect-free MT_Free textures are not artificially amplified.",
            ],
            "transition_line": "Let us peek inside the trained DeepLabV3+ network on Slide 5 to watch how it isolates a blowhole and a hairline crack layer by layer.",
        },
        {
            "slide_num": 5,
            "title": "How DeepLabV3+ Works Inside: Real RGB Layer-by-Layer Flow",
            "duration_sec": 75,
            "slide_goal": "Walk visually through Figure 07 showing how input, DoG, 1/4 low-level, ASPP r=2/4/8, and Decoder Fusion localize unseen RGB defects.",
            "point_to_on_screen": [
                "Columns 1 & 2: Raw unseen tile image -> Fixed-scale Local DoG highlights anomalies.",
                "Column 3 (Low-Level 1/4): Show how vertical brushed texture is visible alongside crisp edge details at 32x32.",
                "Columns 4-6 (ASPP r=2,4,8) -> Columns 7-8: Decoder fusion suppresses background and snaps green prediction contour onto cyan GT.",
            ],
            "read_off_script": [
                "Slide 5 shows Figure 07—a live layer-by-layer trace through our trained DeepLabV3+ model on two unseen held-out test images: an MT_Blowhole on the top row and an MT_Crack on the bottom row.",
                "Starting from the left, Column 1 is the raw grayscale-textured input and Column 2 is our Local DoG feature map, which makes surface discontinuities stand out while keeping uniform areas calmer. Column 3 shows the 1/4-resolution ResNet-18 low-level feature map at 32x32—notice it still retains fine edge geometry. Columns 4, 5, and 6 visualize the parallel ASPP dilated branches at rates 2, 4, and 8, capturing local spot context through wider illumination context.",
                "Look at Column 7—Decoder Fusion—where ASPP context merges back with the 1/4 low-level features: all the brushed background texture is suppressed and only the true defect location lights up brightly. In Column 8, with our saturated overlay and pixel scale, the green predicted contour lands directly on top of the dashed cyan ground-truth void on the right edge and the crack at the top.",
            ],
            "technical_backing": [
                "Generated via plot_real_rgb_deeplab_flow() in src/visualization/csir_demo_plots.py (Figure 07).",
                "Architecture: DeepLabV3Plus(in_channels=3, num_classes=1, backbone='resnet18', output_stride=8, aspp_rates=(2,4,8)).",
                "Saturated overlay uses alpha=0 above threshold p>=0.15 so brushed background remains clearly inspectable.",
            ],
            "transition_line": "Slide 6 presents the quantitative metrics and visual overlays across all 8 unseen RGB test samples.",
        },
        {
            "slide_num": 6,
            "title": "Real RGB Held-Out Test Results: Identification & Localization",
            "duration_sec": 80,
            "slide_goal": "Report all 8 unseen test evaluations honestly, including 87.5% identification accuracy and the single clean-tile edge false alarm.",
            "point_to_on_screen": [
                "Left super-grid (Figure 08): Point out the 6 defective rows (green contour matching cyan GT) and the bottom-right clean MT_Free PASS sample.",
                "Top-right summary card: 87.5% (7/8) ID accuracy, 0.443 mean defective AP, 0.367 mean Dice.",
                "Right table: Point out MT_Free__exp1_num_351039 (the one dark vignette false alarm) vs MT_Free__exp1_num_855 (0 px error, Dice=1.000).",
            ],
            "read_off_script": [
                "Slide 6 evaluates every single one of the 8 strictly unseen test samples—6 defective and 2 defect-free controls.",
                "Looking at the summary box and table on the right, DeepLabV3+ correctly identified PASS versus FAIL status on 7 out of 8 unseen samples, achieving 87.5% identification accuracy, a mean defective Average Precision of 0.443, and a mean Dice score of 0.367—despite training on only 16 real images. On the left gallery, across MT_Blowhole, MT_Crack, and MT_Break, the high-saturation probability maps and green localization contours with white centroid crosshairs pinpoint where the flaws are, even on tiny 0.2% cracks.",
                "To be completely transparent about limitations: on one clean control sample, MT_Free__exp1_num_351039, a dark shadow vignette near the top rim triggered a 127-pixel false positive, while the second clean control passed with zero false-alarm pixels and 1.000 Dice. That directly motivates adding outer boundary masking as a future improvement before deployment.",
            ],
            "technical_backing": [
                "Metrics stored in outputs/reports/real_rgb_deeplab_metrics.json (runtime logged, deterministic seed=42).",
                "Trained 28 epochs with AdamW (lr=2.5e-3), FocalDiceLoss(alpha=0.65), and 2x MT_Free clean-tile balancing.",
                "Connected-component filter suppresses isolated speckles <6 px before PASS/FAIL thresholding at p>=0.55.",
            ],
            "transition_line": "Having verified DeepLabV3+ on real pixel-annotated industrial images, we now turn directly to our 15 real CSIR-CSIO Terahertz scans on Slide 7.",
        },
        {
            "slide_num": 7,
            "title": "Benchmark 2: CSIR-CSIO Terahertz Polymer Dataset (15 Scans)",
            "duration_sec": 90,
            "slide_goal": "Present the full 15-scan empirical audit (4 materials, same pattern, 15-100% infill, varied defect cases) and demonstrate the new CSIR mm scales & high-saturation contrast.",
            "point_to_on_screen": [
                "Left 3x5 gallery (Figure 01): Point out Row 1 (Cohort A: No defect, 1C, 1S, 3C, 3S) and Rows 2-3 (10 Cohort B multi-defect plates across PLA, PETG, ABS, TPU and 15-100% infill).",
                "Axes & colorbars: Show Distance (mm) axes (40x40, 39x39, 38x38, 55x55, 60x60 mm) and interior-saturated jet colorbars (p4-p96).",
                "Right panel (Figure 02): Show how masking hot-strip columns and computing |Local DoG| exposes both dips (1C/1S) and peaks (3C/3S).",
            ],
            "read_off_script": [
                "Slide 7 brings us to the core CSIR-CSIO Terahertz dataset—all 15 raw CSV scans in Polymer sample data. Thank you for the feedback on adding explicit millimeter scales and boosting image saturation so defects stand out prominently. We updated every figure here to use your exact physical Distance in mm axes—40x40, 39x39, 38x38, 55x55, and 60x60 mm—with explicit colorbars and an interior-anchored p4-to-p96 jet stretch.",
                "Why was that saturation fix so critical? Look at Rows 2 and 3 in Figure 01: those 10 Cohort B scans have extreme hot-strip flares along the outer border columns up to +0.77, which previously compressed the entire polymer interior into uniform dark blue. By anchoring the color scale strictly on the sample interior, you can now clearly see the multi-defect plate bands, circular anomalies, and pockets across all 4 materials—PLA, PETG, ABS, and TPU—printed with the same pattern across 15%, 45%, 75%, and 100% infill. Meanwhile Row 1 shows our 5 Cohort A PLA scans spanning no defect, single circular and square voids, and 3x3 circular and square arrays.",
                "On the right in Figure 02, our Layer 1 conditioning masks those border hot strips, denoises with a 3x3 Wiener filter, and computes Absolute Local DoG so both destructive attenuation dips like 1C/1S and constructive 3x3 reflection peaks like 3C/3S light up consistently for DeepLabV3+.",
            ],
            "technical_backing": [
                "Filename parser converts '135THz'..'450THz' labels to physical 135..450 GHz (src/preprocessing/loader.py).",
                "Cohort A has strictly positive values [0.004, 1.658]; Cohort B is baseline-subtracted [-0.116, +0.771]—handled uniformly via interior [p5, p95] norm.",
                "Cohort A includes ND (0 defects), 1C/1S (1 defect), 3C/3S (3x3 arrays); Cohort B has 10 multi-defect mixed plates.",
            ],
            "transition_line": "Now on Slide 8, let us look inside THz-Adapted DeepLabV3+ when processing held-out real THz scans.",
        },
        {
            "slide_num": 8,
            "title": "How THz-Adapted DeepLabV3+ Works on Real CSIR Scans",
            "duration_sec": 75,
            "slide_goal": "Explain Figure 04 showing THz-DeepLabV3+ (OS=8, ASPP r=1,2,4) internal feature activations on unseen PLA_1C and PLA_3S scans.",
            "point_to_on_screen": [
                "Top row (Held-out PLA_1C at 450 GHz, 40x40 mm): Trace center circular dip from Ch0 -> 32x32 Low-Level -> ASPP r=1 -> saturated green prediction.",
                "Bottom row (Held-out PLA_3S at 200 GHz, 38x38 mm): Trace periodic 3x3 grid spots captured by ASPP r=2 and r=4.",
                "Point to Distance (mm) scales and tight colorbars across every column.",
            ],
            "read_off_script": [
                "Slide 8 shows Figure 04—the internal feature activations of our THz-Adapted DeepLabV3+ on two strictly held-out CSIR scans that the model never saw during training: single circular void PLA_1C on top, and the 3x3 square array PLA_3S on the bottom.",
                "Notice how our architecture changes pay off physically. Upsampling the 46x46 scan to 128x128 with ResNet-18 Output Stride 8 preserves a crisp 32x32 low-level edge map in Column 2 and a 16x16 bottleneck across Columns 3 to 5. In Column 3, ASPP dilation rate 1 acts like a localized matched filter around the 3 to 4 mm beam spot—you can see it isolating the central cavity in PLA_1C. Meanwhile, in Columns 4 and 5, rates 2 and 4 capture wider neighborhood context spanning the periodic row and column spacing of the 3x3 array in PLA_3S.",
                "In Column 6, with high-saturation probability overlays calibrated in physical millimeters, the predicted contours lock tightly onto the central circular void of PLA_1C and the multiple spot peaks of PLA_3S.",
            ],
            "technical_backing": [
                "Model: DeepLabV3Plus(in_channels=3, num_classes=1, backbone='resnet18', output_stride=8, aspp_rates=(1,2,4)).",
                "Evaluated strictly inside Leave-One-Scan-Out (LOSO) folds where the displayed scan was completely held out.",
                "Generated by plot_thz_deeplabv3plus_activations() in src/visualization/csir_demo_plots.py.",
            ],
            "transition_line": "Slide 9 shows the full Leave-One-Scan-Out benchmark against SmallUNet plus millimeter OpenCV metrology across all samples.",
        },
        {
            "slide_num": 9,
            "title": "CSIR THz Test Samples: Held-Out LOSO Localization & Metrology",
            "duration_sec": 90,
            "slide_goal": "Present complete LOSO comparison (THz-DeepLabV3+ vs SmallUNet), millimeter metrology outputs, cross-cohort inference on multi-defect plates, and honest caveats.",
            "point_to_on_screen": [
                "Left gallery (Figure 05): Scan down rows—1C, 1S (single void), 3C, 3S (3x3 arrays), ND (PASS 0 defects), and bottom 2 rows (cross-cohort ABS & PLA multi-defect plates).",
                "Column 4 Metrology Overlays: Point out white centroids, yellow beam-corrected diameters (mm), and PASS/FAIL badges.",
                "Top-right LOSO table & bottom-right caveats: Highlight 0.654 AP vs SmallUNet 0.572 AP, PLA_1S 0.999 AP, PLA_ND 0 false alarms, and PLA_1C bevel trigger caveat.",
            ],
            "read_off_script": [
                "Slide 9 summarizes our end-to-end THz verification across Cohort A Leave-One-Scan-Out folds plus zero-shot inference on two Cohort B multi-defect plates.",
                "How did we validate this without data leakage? For each of the 5 Cohort A scans, we trained THz-DeepLabV3+ and a SmallUNet baseline on only the other 4 scans using physics-safe augmentations and synthetic defect injection on clean backgrounds, then tested strictly on the held-out scan. Overall, THz-DeepLabV3+ outperformed SmallUNet on mean Average Precision: 0.654 versus 0.572. Specifically, on PLA_1S it reached 0.999 AP and 0.984 Dice, pinpointing a single defect with a beam-corrected diameter of 5.3 mm; and on the clean PLA_ND scan it returned PASS with zero false-alarm pixels.",
                "Look at the bottom two rows of Figure 05 as well: even though Cohort B plates come from a completely different signed-baseline scan protocol with 4 bands and multiple defect kinds, the Cohort-A-trained DeepLabV3+ still flags localized anomalies in ABS and PLA. Honestly noting our two caveats: on PLA_1C, alongside the central void at (24.5, 21.9) mm with d=5.5 mm, the model also fired on the outer sample bevel ring; and all THz metrics currently use physics-derived reference masks until your CAD and caliper measurements arrive.",
            ],
            "technical_backing": [
                f"LOSO metrics (outputs/reports/deeplab_vs_unet_verification_metrics.json): DL Mean AP={thz_dl_ap:.3f} vs UNet={thz_un_ap:.3f}; 1S AP={_format_num(f_1s.get('deeplabv3plus_ap'))}, 3C AP={_format_num(f_3c.get('deeplabv3plus_ap'))}, 3S AP={_format_num(f_3s.get('deeplabv3plus_ap'))}, 1C AP={_format_num(f_1c.get('deeplabv3plus_ap'))}.",
                "Metrology JSONs saved per scan in outputs/reports/inspection_report_*.json.",
                "Cross-cohort Cohort B showcase uses full Cohort-A DeepLabV3+ model on ABS_0.4THz_100p & PLA_0.4THz_100p.",
            ],
            "transition_line": "Finally, Slide 10 covers our Model 1 audit across the 4 polymers and infills, and the exact scan requests we need from CSIR-CSIO for Phase 2.",
        },
        {
            "slide_num": 10,
            "title": "Model 1 Audit, Key Constraints & Future Improvements Roadmap",
            "duration_sec": 85,
            "slide_goal": "Explain why Model 1 cannot yet be trained due to the Cohort B frequency confounder and single printing pattern, present the improvement roadmap, and align on CSIR next-batch scans.",
            "point_to_on_screen": [
                "Top-left chart (Figure 06 left): Show 400 GHz 100% infill bars overlapping (PLA +0.0581, ABS +0.0600, TPU +0.0619) while PETG is red at 450 GHz (+0.0300).",
                "Top-left right chart (Figure 06 right): Point to PETG 15% at 425 GHz (-0.0863) vs PLA at 375 GHz.",
                "Right boxes: Walk through the 4 algorithmic upgrades and the 4 priority scan requests for CSIR-CSIO.",
            ],
            "read_off_script": [
                "Our final slide audits Model 1—our Global Structural Characterizer for material, infill density, and printing pattern—across the 10 Cohort B scans.",
                "As you know, those scans cover 4 polymer materials across 15% to 100% infill, all printed in one shared printing pattern. Why haven't we trained Model 1 yet? Look at Figure 06 on the left: at 400 GHz and 100% infill, PLA (+0.0581), ABS (+0.0600), and TPU (+0.0619) overlap well within their interquartile spread. Meanwhile PETG appears much lower—+0.0300 at 100% and -0.0863 at 15%—only because PETG was acquired at 450 and 425 GHz instead of 400 and 375 GHz! Because frequency and material are confounded, and only one printing pattern exists so far, training Model 1 right now would memorize acquisition frequency rather than polymer physics.",
                "So to conclude, on the right we outline our roadmap: first, RGB-to-THz transfer learning, outer plate-masking to suppress the PLA_1C bevel false alarm, dual-frequency phase fusion, and scaling up on our SRM GPU Lab Docker environment. In return, to unlock Phase 2, we kindly request from CSIR-CSIO: fixed-frequency and fixed-gain scans with an empty-holder reference, true CAD/caliper defect measurements for calibration, paired Layer-3 buried defects with clean twins, and the partner AM lab's multi-pattern prints. Thank you, and we are happy to take your questions.",
            ],
            "technical_backing": [
                "Model 1 multi-head ResNet-34 smoke-tested cleanly in scripts/verify_model1_separability.py (outputs/reports/model1_separability_summary.json).",
                "Containerized GPU setup ready in docker/Dockerfile + configs/default_config.yaml for SRM GPU Lab scaling.",
                "Phase 1 target: ~45 scans for robust DeepLabV3+ defect metrology; Phase 2 target: ~100-130 scans for full 4-task benchmark.",
            ],
            "transition_line": "End of deck — open floor for CSIR-CSIO mentor feedback and next-batch DoE confirmation.",
        },
    ]

    for sl in slides:
        bullets = " | ".join(sl["point_to_on_screen"])
        script_txt = "\n\n".join(sl["read_off_script"])
        tech_txt = "\n".join(f"- {t}" for t in sl["technical_backing"])
        sl["pptx_notes"] = (
            f"SLIDE {sl['slide_num']} ({sl['duration_sec']}s) — {sl['title']}\n"
            f"GOAL: {sl['slide_goal']}\n\n"
            f"POINT TO ON SCREEN:\n- " + "\n- ".join(sl["point_to_on_screen"]) + "\n\n"
            f"READ-OFF SCRIPT:\n{script_txt}\n\n"
            f"TECHNICAL BACKING:\n{tech_txt}\n\n"
            f"TRANSITION: {sl['transition_line']}"
        )
        _ = bullets

    qas = [
        [
            "Why did you show results on an industrial RGB dataset first before THz?",
            "Cohort A currently has only 5 THz defect-geometry scans without external CAD ground-truth masks. Testing the exact same ResNet-18 OS=8 DeepLabV3+ on 24 real industrial RGB images (16 train / 8 held-out test) with human-annotated masks proves objectively that our pipeline detects tiny 0.2% voids and hairline cracks at 87.5% ID accuracy before scaling THz scans.",
        ],
        [
            "Why DeepLabV3+ instead of U-Net, YOLOv8-seg, PatchCore, or Vision Transformers?",
            "YOLOv8-seg needs tight rectangular bounding boxes, which break on 3-4 mm diffraction-blurred seams. PatchCore outputs uncalibrated anomaly maps without crisp closed contours for mm^2 metrology. ViTs overfit under N<200 scans. Standard U-Net lacks parallel dilated ASPP branches, which gave DeepLabV3+ higher mean AP (0.654 vs 0.572) in our THz LOSO verification.",
        ],
        [
            "Why did you add mm scales and change the color saturation across the figures?",
            "Per CSIR guidance, plotting raw min-max caused hot border flares (+0.59 to +0.77) in Cohort B to compress interior polymer defects into flat dark blue. Anchoring the jet color scale strictly on interior [p4, p96] percentiles, mapping pixels to physical Distance (mm) matching MATLAB scans, and using transparent-background probability overlays makes every defect pop immediately.",
        ],
        [
            "Why did PLA_1C trigger along the outer bevel edge in addition to the central hole?",
            "At 450 GHz, PLA_1C has a strong outer thickness/bevel transition gradient. Because Cohort A only contains 5 scans, the LOSO fold holding out PLA_1C lacked enough bevel-gradient negatives and treated the steep edge drop-off as a secondary anomaly—though it still cleanly localized the true center void (24.5, 21.9 mm, d=5.5 mm, conf=0.959). Next step: add outer plate-boundary masking before defect segmentation.",
        ],
        [
            "Are the THz AP and Dice scores evaluated against true physical CAD measurements?",
            "Not yet—we explicitly flag this caveat on Slide 9. Because caliper/CAD coordinates for 1C, 1S, 3C, 3S and Cohort B plates have not been supplied yet, current THz LOSO metrics are measured against physics-derived reference masks extracted from the CSV intensity anomalies. Once CSIR shares the ground-truth table, we will recalibrate both Dice/IoU and the beam-deconvolution diameter formula.",
        ],
        [
            "Why is Model 1 (material, infill %, printing pattern) not trained yet despite having 4 polymers and 4 infills?",
            "Two empirical reasons found in our Cohort B audit: (1) PETG was scanned at 450/425 GHz while PLA/ABS/TPU were scanned at 400/375 GHz, confounding material attenuation with frequency attenuation; and (2) all 15 scans currently share one default printing pattern. Training a classifier now would learn scanner frequency instead of polymer physics.",
        ],
        [
            "How will the pipeline handle plates with multiple defects of different kinds (e.g., blind vs buried)?",
            "Right now DeepLabV3+ runs with num_classes=1 (binary defect vs clean) because Cohort A has 5 scans. The architecture already supports num_classes=K multi-class segmentation (e.g., 0=Clean, 1=Blind Void, 2=Buried Delamination, 3=Bulge) and will be switched on once the Phase-1 ~45-scan batch arrives with labeled blind and Layer-3 buried twins, optionally fusing dual-frequency and ToF phase channels.",
        ],
    ]

    return {
        "cheat_sheet": cheat_sheet,
        "slides": slides,
        "qas": qas,
    }


def _set_cell_shading(cell: Any, fill_hex: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def _set_cell_margins(cell: Any, top: int = 80, bottom: int = 80, left: int = 110, right: int = 110) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def _set_table_borders(table: Any, color_hex: str = _BORDER_HEX, sz: str = "4") -> None:
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'<w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'<w:right w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f'<w:insideV w:val="single" w:sz="{sz}" w:space="0" w:color="{color_hex}"/>'
        f"</w:tblBorders>"
    )
    tbl_pr.append(borders)


def _prevent_row_split(row: Any) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def _add_run(
    para: Any,
    text: str,
    size_pt: float = 10.0,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor = _SLATE_RGB,
) -> None:
    run = para.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = color


def _populate_styled_table(
    doc: Document,
    headers: List[str],
    rows: List[List[str]],
    col_widths_in: List[float],
    font_size: float = 9.0,
) -> None:
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    _set_table_borders(tbl)

    hdr_row = tbl.rows[0]
    _prevent_row_split(hdr_row)
    tr_pr = hdr_row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:tblHeader"))

    for c_idx, title in enumerate(headers):
        cell = hdr_row.cells[c_idx]
        cell.width = Inches(col_widths_in[c_idx])
        _set_cell_shading(cell, _NAVY_HEX)
        _set_cell_margins(cell, top=65, bottom=65, left=90, right=90)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        _add_run(p, title, size_pt=font_size, bold=True, color=_WHITE_RGB)

    for r_idx, row_vals in enumerate(rows):
        row = tbl.rows[r_idx + 1]
        _prevent_row_split(row)
        bg = "FFFFFF" if r_idx % 2 == 0 else _ALT_HEX
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = Inches(col_widths_in[c_idx])
            _set_cell_shading(cell, bg)
            _set_cell_margins(cell, top=55, bottom=55, left=90, right=90)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            _add_run(p, val, size_pt=font_size, bold=(c_idx == 0), color=_SLATE_RGB)


def build_side_by_side_speaker_guide_docx(
    out_docx_path: str | Path,
    slide_png_paths: List[Path],
    rgb_metrics: Dict[str, Any],
    thz_metrics: Dict[str, Any],
) -> None:
    """Create Landscape 12-page DOCX where every slide thumbnail sits side-by-side with its read-off script."""
    content = get_csir_speaker_content(rgb_metrics, thz_metrics)
    cheat = content["cheat_sheet"]
    slides = content["slides"]
    qas = content["qas"]

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width = Inches(11.0)
    sec.page_height = Inches(8.5)
    sec.top_margin = Inches(0.45)
    sec.bottom_margin = Inches(0.45)
    sec.left_margin = Inches(0.50)
    sec.right_margin = Inches(0.50)

    # Configure header / footer
    hdr_p = sec.header.paragraphs[0]
    hdr_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _add_run(hdr_p, "SRM University & CSIR-CSIO | Side-by-Side Presentation Speaker Guide", size_pt=8.0, color=_MUTED_RGB)

    # ------------------------------------------------------------------
    # PAGE 1: Meeting Cheat Sheet, 15-Scan Taxonomy & Pacing
    # ------------------------------------------------------------------
    banner_tbl = doc.add_table(rows=1, cols=1)
    banner_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    banner_tbl.autofit = False
    b_cell = banner_tbl.rows[0].cells[0]
    b_cell.width = Inches(10.0)
    _set_cell_shading(b_cell, _NAVY_HEX)
    _set_cell_margins(b_cell, top=110, bottom=110, left=150, right=150)
    bp1 = b_cell.paragraphs[0]
    bp1.paragraph_format.space_after = Pt(2)
    _add_run(bp1, cheat["meeting_title"], size_pt=16.0, bold=True, color=_WHITE_RGB)
    bp2 = b_cell.add_paragraph()
    bp2.paragraph_format.space_after = Pt(0)
    _add_run(bp2, f"{cheat['presenters']}  |  Pacing: {cheat['total_duration']}", size_pt=9.5, color=RGBColor(0xBF, 0xD7, 0xED))

    h1 = doc.add_paragraph()
    h1.paragraph_format.space_before = Pt(8)
    h1.paragraph_format.space_after = Pt(3)
    _add_run(h1, "1. Complete 15-Scan THz Dataset Inventory (Read-Off Reference)", size_pt=11.0, bold=True, color=_NAVY_RGB)
    _populate_styled_table(
        doc,
        ["Defect Category", "Scan Count & IDs", "Materials / Pattern / Infill / Size", "Role in Verification"],
        cheat["dataset_breakdown"],
        [2.0, 2.2, 3.3, 2.5],
        font_size=8.6,
    )

    h2 = doc.add_paragraph()
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(3)
    _add_run(h2, "2. Headline Numbers to Cite Live", size_pt=11.0, bold=True, color=_NAVY_RGB)
    for line in cheat["headline_metrics"]:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.left_indent = Inches(0.15)
        _add_run(p, f"• {line}", size_pt=9.2, color=_SLATE_RGB)

    h3 = doc.add_paragraph()
    h3.paragraph_format.space_before = Pt(7)
    h3.paragraph_format.space_after = Pt(3)
    _add_run(h3, "3. 10-Slide Flow & Pacing Map (1 Slide per Page Following This Sheet)", size_pt=11.0, bold=True, color=_NAVY_RGB)
    pacing_rows = [
        [f"Slide {s['slide_num']}", f"{s['duration_sec']}s", s["title"], s["slide_goal"]]
        for s in slides
    ]
    _populate_styled_table(
        doc,
        ["Slide", "Time", "Slide Title", "Core Message"],
        pacing_rows,
        [0.75, 0.55, 3.85, 4.85],
        font_size=8.2,
    )

    # ------------------------------------------------------------------
    # PAGES 2-11: Side-by-Side Slide Thumbnail + Word-for-Word Script
    # ------------------------------------------------------------------
    for idx, sl in enumerate(slides):
        doc.add_page_break()

        s_tbl = doc.add_table(rows=2, cols=2)
        s_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        s_tbl.autofit = False
        _set_table_borders(s_tbl)

        # Row 0: Merged Navy Header Bar
        hdr_row = s_tbl.rows[0]
        _prevent_row_split(hdr_row)
        merged_hdr = hdr_row.cells[0].merge(hdr_row.cells[1])
        merged_hdr.width = Inches(10.0)
        _set_cell_shading(merged_hdr, _NAVY_HEX)
        _set_cell_margins(merged_hdr, top=75, bottom=75, left=120, right=120)
        hp = merged_hdr.paragraphs[0]
        hp.paragraph_format.space_after = Pt(0)
        _add_run(
            hp,
            f"SLIDE {sl['slide_num']} / {len(slides)}   |   {sl['title']}   (~{sl['duration_sec']} sec)",
            size_pt=12.0,
            bold=True,
            color=_WHITE_RGB,
        )

        # Row 1: Side-by-Side Content
        body_row = s_tbl.rows[1]
        _prevent_row_split(body_row)
        left_cell = body_row.cells[0]
        right_cell = body_row.cells[1]
        left_cell.width = Inches(4.2)
        right_cell.width = Inches(5.8)
        left_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        right_cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
        _set_cell_shading(left_cell, _LIGHT_HEX)
        _set_cell_shading(right_cell, "FFFFFF")
        _set_cell_margins(left_cell, top=90, bottom=90, left=90, right=90)
        _set_cell_margins(right_cell, top=90, bottom=90, left=110, right=110)

        # Left cell: Slide thumbnail
        p_img = left_cell.paragraphs[0]
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_after = Pt(6)
        if idx < len(slide_png_paths) and slide_png_paths[idx].exists():
            p_img.add_run().add_picture(str(slide_png_paths[idx]), width=Inches(4.0))
        else:
            _add_run(p_img, f"[Slide {sl['slide_num']} Thumbnail Missing]", size_pt=9.0, italic=True, color=_MUTED_RGB)

        # Left cell: Point to on Screen box
        p_pt_hdr = left_cell.add_paragraph()
        p_pt_hdr.paragraph_format.space_before = Pt(4)
        p_pt_hdr.paragraph_format.space_after = Pt(3)
        _add_run(p_pt_hdr, "POINT TO ON SCREEN:", size_pt=9.5, bold=True, color=_TEAL_RGB)
        for cue in sl["point_to_on_screen"]:
            pc = left_cell.add_paragraph()
            pc.paragraph_format.space_before = Pt(1)
            pc.paragraph_format.space_after = Pt(3)
            pc.paragraph_format.left_indent = Inches(0.10)
            _add_run(pc, f"▸ {cue}", size_pt=8.8, color=_SLATE_RGB)

        # Right cell: What This Slide Shows
        p_goal = right_cell.paragraphs[0]
        p_goal.paragraph_format.space_before = Pt(0)
        p_goal.paragraph_format.space_after = Pt(5)
        _add_run(p_goal, "WHAT THIS SLIDE SHOWS: ", size_pt=9.5, bold=True, color=_TEAL_RGB)
        _add_run(p_goal, sl["slide_goal"], size_pt=9.5, bold=True, color=_NAVY_RGB)

        # Right cell: Word-for-word Read-Off Script
        p_say_hdr = right_cell.add_paragraph()
        p_say_hdr.paragraph_format.space_before = Pt(2)
        p_say_hdr.paragraph_format.space_after = Pt(3)
        _add_run(p_say_hdr, "WHAT TO SAY (READ-OFF SCRIPT):", size_pt=9.5, bold=True, color=_TEAL_RGB)

        for para_txt in sl["read_off_script"]:
            ps = right_cell.add_paragraph()
            ps.paragraph_format.space_before = Pt(0)
            ps.paragraph_format.space_after = Pt(5)
            ps.paragraph_format.line_spacing = 1.12
            _add_run(ps, para_txt, size_pt=10.2, color=_SLATE_RGB)

        # Right cell: How We Built & Computed This
        p_tech_hdr = right_cell.add_paragraph()
        p_tech_hdr.paragraph_format.space_before = Pt(3)
        p_tech_hdr.paragraph_format.space_after = Pt(2)
        _add_run(p_tech_hdr, "HOW WE BUILT & COMPUTED THIS (TECHNICAL BACKING):", size_pt=8.8, bold=True, color=_NAVY_RGB)
        for t_line in sl["technical_backing"]:
            pt = right_cell.add_paragraph()
            pt.paragraph_format.space_before = Pt(0)
            pt.paragraph_format.space_after = Pt(2)
            pt.paragraph_format.left_indent = Inches(0.10)
            _add_run(pt, f"• {t_line}", size_pt=8.4, color=_MUTED_RGB)

        # Right cell: Transition Line
        p_tr = right_cell.add_paragraph()
        p_tr.paragraph_format.space_before = Pt(5)
        p_tr.paragraph_format.space_after = Pt(0)
        _add_run(p_tr, "TRANSITION -> ", size_pt=8.8, bold=True, color=_TEAL_RGB)
        _add_run(p_tr, sl["transition_line"], size_pt=9.0, italic=True, color=_TEAL_RGB)

    # ------------------------------------------------------------------
    # PAGE 12: Anticipated CSIR-CSIO Mentor Q&As
    # ------------------------------------------------------------------
    doc.add_page_break()
    qa_banner = doc.add_paragraph()
    qa_banner.paragraph_format.space_after = Pt(6)
    _add_run(qa_banner, "Anticipated CSIR-CSIO Mentor Q&A Cheat Sheet", size_pt=14.0, bold=True, color=_NAVY_RGB)
    _populate_styled_table(
        doc,
        ["Likely Question from CSIR Mentors", "Recommended Clear Answer (Read-Off)"],
        qas,
        [3.3, 6.7],
        font_size=9.2,
    )

    out_path = Path(out_docx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)


def build_side_by_side_speaker_guide_md(
    out_md_path: str | Path,
    rgb_metrics: Dict[str, Any],
    thz_metrics: Dict[str, Any],
) -> None:
    """Generate markdown companion speaker guide synced with the DOCX and PPTX notes."""
    content = get_csir_speaker_content(rgb_metrics, thz_metrics)
    cheat = content["cheat_sheet"]
    slides = content["slides"]
    qas = content["qas"]

    lines: List[str] = [
        f"# {cheat['meeting_title']} — Side-by-Side Speaker Guide",
        "",
        f"**Presenters:** {cheat['presenters']}  ",
        f"**Total Pacing:** {cheat['total_duration']}  ",
        "**Companion Decks:** [`outputs/CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pdf`](../outputs/CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pdf) | [`outputs/CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pptx`](../outputs/CSIR_DeepLabV3Plus_Architecture_and_Verification_Presentation.pptx) | [`outputs/CSIR_Presentation_Side_By_Side_Speaker_Guide.docx`](../outputs/CSIR_Presentation_Side_By_Side_Speaker_Guide.docx)",
        "",
        "---",
        "",
        "## Pre-Meeting Cheat Sheet",
        "",
        "### 15-Scan THz Dataset Summary",
        "| Defect Category | Scans | Materials / Pattern / Infill / Size | Role in Pipeline |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for row in cheat["dataset_breakdown"]:
        safe_row = [str(cell).replace("|", "·") for cell in row]
        lines.append(f"| {safe_row[0]} | {safe_row[1]} | {safe_row[2]} | {safe_row[3]} |")

    lines.extend(["", "### Headline Verification Metrics"])
    for m in cheat["headline_metrics"]:
        lines.append(f"- {m}")

    lines.extend(["", "---", "", "## Slide-by-Slide Walkthrough & Read-Off Scripts"])
    for sl in slides:
        lines.extend(
            [
                "",
                f"### Slide {sl['slide_num']} / {len(slides)}: {sl['title']} *(~{sl['duration_sec']}s)*",
                f"**What This Slide Shows:** {sl['slide_goal']}",
                "",
                "**Point to on Screen:**",
            ]
        )
        for cue in sl["point_to_on_screen"]:
            lines.append(f"- {cue}")

        lines.extend(["", "**What to Say (Read-Off Script):**"])
        for p in sl["read_off_script"]:
            lines.extend(["", f"> {p}"])

        lines.extend(["", "**How We Built & Computed This (Technical Backing):**"])
        for t in sl["technical_backing"]:
            lines.append(f"- {t}")

        lines.extend(["", f"**Transition ->** *{sl['transition_line']}*", "", "---"])

    lines.extend(["", "## Anticipated CSIR-CSIO Mentor Q&As", ""])
    lines.extend(["| Likely Question | Recommended Clear Answer |", "| :--- | :--- |"])
    for q, a in qas:
        lines.append(f"| **{q}** | {a} |")

    out_path = Path(out_md_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
