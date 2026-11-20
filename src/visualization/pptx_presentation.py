"""Native editable 16:9 widescreen PowerPoint (.pptx) builder for CSIR-CSIO meeting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from pptx.util import Inches, Pt

from .speaker_guide import get_csir_speaker_content


NAVY = RGBColor(0x0F, 0x29, 0x42)
TEAL = RGBColor(0x13, 0x6F, 0x8A)
ACCENT = RGBColor(0xD9, 0x5F, 0x02)
GREEN = RGBColor(0x1B, 0x7F, 0x4B)
RED = RGBColor(0xB9, 0x1C, 0x1C)
BLUE_DARK = RGBColor(0x1E, 0x40, 0xAF)
TEAL_DEEP = RGBColor(0x0F, 0x76, 0x6E)
NAVY_MID = RGBColor(0x1E, 0x3A, 0x5F)
SLATE = RGBColor(0x33, 0x41, 0x55)
MUTED = RGBColor(0x64, 0x74, 0x8B)
LIGHT_BG = RGBColor(0xF4, 0xF7, 0xFA)
ALT_ROW = RGBColor(0xF8, 0xFA, 0xFC)
BORDER = RGBColor(0xCB, 0xD5, 0xE1)
SUB_TEXT = RGBColor(0xBF, 0xD7, 0xED)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _add_rect(
    slide: Any,
    left_in: float,
    top_in: float,
    w_in: float,
    h_in: float,
    fill_rgb: RGBColor,
    line_rgb: RGBColor | None = None,
    rounded: bool = False,
) -> Any:
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(
        shape_type,
        Inches(left_in),
        Inches(top_in),
        Inches(w_in),
        Inches(h_in),
    )
    if rounded:
        try:
            shp.adjustments[0] = 0.04
        except Exception:
            pass
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_rgb
    if line_rgb is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line_rgb
        shp.line.width = Pt(1.0)
    return shp


def _create_slide_shell(
    prs: Presentation,
    slide_num: int,
    total: int,
    title: str,
    subtitle: str,
    speaker_notes: str,
) -> Any:
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # White background
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE

    # Header bar + teal underline
    _add_rect(slide, 0.0, 0.0, 13.333, 0.78, NAVY)
    _add_rect(slide, 0.0, 0.78, 13.333, 0.045, TEAL)

    tb_t = slide.shapes.add_textbox(Inches(0.32), Inches(0.06), Inches(11.1), Inches(0.40))
    tf_t = tb_t.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    r_t = p_t.add_run()
    r_t.text = title
    r_t.font.name = "Calibri"
    r_t.font.size = Pt(21)
    r_t.font.bold = True
    r_t.font.color.rgb = WHITE

    tb_s = slide.shapes.add_textbox(Inches(0.32), Inches(0.45), Inches(11.1), Inches(0.28))
    tf_s = tb_s.text_frame
    tf_s.word_wrap = True
    tf_s.margin_left = tf_s.margin_top = tf_s.margin_right = tf_s.margin_bottom = 0
    p_s = tf_s.paragraphs[0]
    r_s = p_s.add_run()
    r_s.text = subtitle
    r_s.font.name = "Calibri"
    r_s.font.size = Pt(11.5)
    r_s.font.color.rgb = SUB_TEXT

    # Slide pill
    pill = _add_rect(slide, 11.75, 0.19, 1.28, 0.38, TEAL, rounded=True)
    tf_p = pill.text_frame
    tf_p.vertical_anchor = MSO_ANCHOR.MIDDLE
    p_p = tf_p.paragraphs[0]
    p_p.alignment = PP_ALIGN.CENTER
    r_p = p_p.add_run()
    r_p.text = f"Slide {slide_num} / {total}"
    r_p.font.name = "Calibri"
    r_p.font.size = Pt(11)
    r_p.font.bold = True
    r_p.font.color.rgb = WHITE

    # Footer bar
    _add_rect(slide, 0.0, 7.20, 13.333, 0.30, LIGHT_BG)
    tb_fl = slide.shapes.add_textbox(Inches(0.32), Inches(7.24), Inches(7.5), Inches(0.22))
    tf_fl = tb_fl.text_frame
    tf_fl.margin_left = tf_fl.margin_top = tf_fl.margin_right = tf_fl.margin_bottom = 0
    r_fl = tf_fl.paragraphs[0].add_run()
    r_fl.text = "THz-AI Polymer NDT Diagnostics  |  DeepLabV3+ Architecture, Real RGB Verification & CSIR THz Findings"
    r_fl.font.name = "Calibri"
    r_fl.font.size = Pt(9)
    r_fl.font.color.rgb = MUTED

    tb_fr = slide.shapes.add_textbox(Inches(8.5), Inches(7.24), Inches(4.52), Inches(0.22))
    tf_fr = tb_fr.text_frame
    tf_fr.margin_left = tf_fr.margin_top = tf_fr.margin_right = tf_fr.margin_bottom = 0
    p_fr = tf_fr.paragraphs[0]
    p_fr.alignment = PP_ALIGN.RIGHT
    r_fr = p_fr.add_run()
    r_fr.text = "SRM University & CSIR-CSIO Research Collaboration"
    r_fr.font.name = "Calibri"
    r_fr.font.size = Pt(9)
    r_fr.font.color.rgb = MUTED

    # Speaker notes
    notes_frame = slide.notes_slide.notes_text_frame
    notes_frame.text = speaker_notes
    return slide


def _add_card_box(
    slide: Any,
    left_in: float,
    top_in: float,
    w_in: float,
    h_in: float,
    title: str,
    body_lines: List[str],
    header_rgb: RGBColor = NAVY,
    fill_rgb: RGBColor = LIGHT_BG,
    title_pt: float = 12.0,
    body_pt: float = 10.0,
) -> None:
    _add_rect(slide, left_in, top_in, w_in, h_in, fill_rgb, line_rgb=BORDER, rounded=False)
    hdr_h = min(0.38, h_in * 0.28)
    hdr = _add_rect(slide, left_in, top_in, w_in, hdr_h, header_rgb, rounded=False)
    tf_h = hdr.text_frame
    tf_h.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf_h.margin_left = Inches(0.12)
    tf_h.margin_top = tf_h.margin_bottom = 0
    p_h = tf_h.paragraphs[0]
    r_h = p_h.add_run()
    r_h.text = title
    r_h.font.name = "Calibri"
    r_h.font.size = Pt(title_pt)
    r_h.font.bold = True
    r_h.font.color.rgb = WHITE

    if not body_lines:
        return

    tb = slide.shapes.add_textbox(
        Inches(left_in + 0.12),
        Inches(top_in + hdr_h + 0.07),
        Inches(w_in - 0.24),
        Inches(max(0.2, h_in - hdr_h - 0.12)),
    )
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for idx, line in enumerate(body_lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_before = Pt(0)
        p.space_after = Pt(4.0)
        p.line_spacing = 1.08
        r = p.add_run()
        r.text = line
        r.font.name = "Calibri"
        r.font.size = Pt(body_pt)
        r.font.color.rgb = SLATE


def _add_flow_block(
    slide: Any,
    left_in: float,
    top_in: float,
    w_in: float,
    h_in: float,
    title: str,
    subtitle: str,
    fill_rgb: RGBColor,
    title_pt: float = 10.5,
    sub_pt: float = 8.8,
) -> None:
    shp = _add_rect(slide, left_in, top_in, w_in, h_in, fill_rgb, rounded=True)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.04)

    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    p1.space_after = Pt(2)
    r1 = p1.add_run()
    r1.text = title
    r1.font.name = "Calibri"
    r1.font.size = Pt(title_pt)
    r1.font.bold = True
    r1.font.color.rgb = WHITE

    if subtitle:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.space_after = Pt(0)
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.name = "Calibri"
        r2.font.size = Pt(sub_pt)
        r2.font.color.rgb = RGBColor(0xF1, 0xF5, 0xF9)


def _add_arrow(slide: Any, x1_in: float, y1_in: float, x2_in: float, y2_in: float) -> None:
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1_in),
        Inches(y1_in),
        Inches(x2_in),
        Inches(y2_in),
    )
    conn.line.color.rgb = NAVY
    conn.line.width = Pt(1.75)
    ln = conn.line._get_or_add_ln()
    ln.append(parse_xml(f'<a:tailEnd {nsdecls("a")} type="triangle" w="med" len="med"/>'))


def _add_picture_fit(
    slide: Any,
    img_path: Path,
    left_in: float,
    top_in: float,
    max_w_in: float,
    max_h_in: float,
) -> None:
    """Embed picture centered within (max_w_in, max_h_in) preserving true aspect ratio."""
    with Image.open(img_path) as im:
        px_w, px_h = im.size
    aspect = px_w / max(px_h, 1)
    box_aspect = max_w_in / max(max_h_in, 1e-5)

    if aspect >= box_aspect:
        draw_w = max_w_in
        draw_h = max_w_in / max(aspect, 1e-5)
    else:
        draw_h = max_h_in
        draw_w = max_h_in * aspect

    pos_x = left_in + (max_w_in - draw_w) / 2.0
    pos_y = top_in + (max_h_in - draw_h) / 2.0
    slide.shapes.add_picture(
        str(img_path),
        Inches(pos_x),
        Inches(pos_y),
        width=Inches(draw_w),
        height=Inches(draw_h),
    )


def _add_styled_table(
    slide: Any,
    left_in: float,
    top_in: float,
    w_in: float,
    h_in: float,
    headers: List[str],
    rows: List[List[str]],
    col_fracs: List[float],
    font_pt: float = 9.0,
) -> None:
    tbl_shape = slide.shapes.add_table(
        1 + len(rows),
        len(headers),
        Inches(left_in),
        Inches(top_in),
        Inches(w_in),
        Inches(h_in),
    )
    tbl = tbl_shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    for c_idx, frac in enumerate(col_fracs):
        tbl.columns[c_idx].width = Inches(w_in * frac)
    row_h = Inches(h_in / max(1 + len(rows), 1))
    for r_idx in range(1 + len(rows)):
        tbl.rows[r_idx].height = row_h

    for c_idx, title in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.06)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = title
        r.font.name = "Calibri"
        r.font.size = Pt(font_pt)
        r.font.bold = True
        r.font.color.rgb = WHITE

    for r_idx, row_vals in enumerate(rows):
        bg = WHITE if r_idx % 2 == 0 else ALT_ROW
        for c_idx, val in enumerate(row_vals):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.025)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            r = p.add_run()
            r.text = val
            r.font.name = "Calibri"
            r.font.size = Pt(font_pt)
            r.font.bold = (c_idx == 0)
            r.font.color.rgb = SLATE


def build_csir_meeting_pptx(
    out_pptx_path: str | Path,
    figures_dir: Path,
    rgb_metrics: Dict[str, Any],
    thz_metrics: Dict[str, Any],
    rgb_manifest_rows: List[Dict[str, str]],
) -> None:
    """Build native editable 10-slide widescreen PowerPoint (.pptx) with speaker notes."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.500)

    speaker_content = get_csir_speaker_content(rgb_metrics, thz_metrics)
    notes_by_num = {s["slide_num"]: s["pptx_notes"] for s in speaker_content["slides"]}
    total = 10

    # ------------------------------------------------------------------
    # SLIDE 1: Title, Scope & Decoupled Two-Model Architecture
    # ------------------------------------------------------------------
    s1 = _create_slide_shell(
        prs,
        1,
        total,
        "Terahertz Non-Destructive Testing (NDT) via DeepLabV3+",
        "Executive Summary, Optical Physics Constraints & Decoupled Two-Model Diagnostic Pipeline",
        notes_by_num[1],
    )
    _add_card_box(
        s1,
        0.30,
        0.95,
        5.85,
        2.95,
        "1. Project Objective, Scope & 15-Scan Inventory",
        [
            "• Goal: Automated 'X-ray style' NDT inspection of 3D-printed FDM polymers.",
            "• Outputs: (1) Material, (2) Infill %, (3) Pattern, (4) Defect Mask + Size/XY.",
            "• Scope: Standalone AI models & OpenCV metrology (no printer/stage drivers).",
            "• Physics: 0.135–0.450 THz reflection; 3–4 mm beam blurs voids; Δn ≈ 0.55.",
            "• 15 Scans: 4 polymers (PLA, PETG, ABS, TPU), 1 shared pattern, 15%–100% infill; spans No-Defect (ND), Single-Void (1C/1S), 3x3 Arrays (3C/3S), and Multi-Defect mixed plates (Cohort B).",
        ],
        header_rgb=NAVY,
        body_pt=10.2,
    )
    _add_card_box(
        s1,
        0.30,
        4.08,
        5.85,
        2.98,
        "2. Why We Decouple Into Two Specialized Models",
        [
            "• Small-Data Bottleneck: Only 15 THz scans exist (high scan cost & ~1000 hr photomixer life) -> monolithic multi-task CNNs suffer gradient conflict.",
            "• Model 1 (Global Characterizer): ResNet-34 multi-head network for bulk Material, Printing Pattern, and Infill % regression.",
            "• Model 2 (Defect Segmentor): THz-Adapted DeepLabV3+ focused solely on pixel-level localization of blind and buried discontinuities.",
            "• Today's Deliverable: Prove DeepLabV3+ on real multi-sample RGB defect data (24 images) AND verify localization on all 15 real THz CSV scans.",
        ],
        header_rgb=TEAL,
        body_pt=10.2,
    )

    _add_card_box(
        s1,
        6.40,
        0.95,
        6.62,
        6.11,
        "End-to-End 6-Layer Diagnostic Architecture",
        [],
        header_rgb=NAVY,
        fill_rgb=WHITE,
    )
    _add_flow_block(
        s1,
        6.68,
        1.52,
        6.06,
        0.82,
        "Layer 1: Signal Conditioning & Physics Stack",
        "Mask hot-strip columns -> Wiener filter -> [Norm Int, |Local DoG|, Sobel]",
        NAVY_MID,
    )
    _add_flow_block(
        s1,
        6.68,
        2.66,
        2.88,
        1.08,
        "Layer 3: Model 1\nGlobal Characterizer",
        "ResNet-34 Multi-Head:\nMat | Pattern | Infill %",
        TEAL,
    )
    _add_flow_block(
        s1,
        9.86,
        2.66,
        2.88,
        1.08,
        "Layer 4: Model 2\nDeepLabV3+ Segmentor",
        "ResNet-18 (OS=8) + ASPP\nFocal+Dice Pixel Mask",
        ACCENT,
    )
    _add_flow_block(
        s1,
        7.40,
        4.10,
        5.34,
        1.15,
        "Layer 5: Deterministic OpenCV Metrology",
        "Threshold (p>=0.5) -> Morph Open -> Connected Components & Moments\nCentroid (X,Y) mm | Area mm^2 | Beam-Corrected Diam sqrt(d_app^2 - d_beam^2)",
        GREEN,
    )
    _add_flow_block(
        s1,
        6.68,
        5.58,
        6.06,
        1.22,
        "Standardized 'X-Ray Style' Diagnostic Output (JSON + Overlay)",
        "• Sample Status: PASS / FAIL  • Defect Count & Centroids (mm)\n• Material / Infill / Pattern Metadata  • Containerized Docker GPU Training",
        SLATE,
    )
    _add_arrow(s1, 8.12, 2.34, 8.12, 2.66)
    _add_arrow(s1, 11.30, 2.34, 11.30, 2.66)
    _add_arrow(s1, 11.30, 3.74, 11.30, 4.10)
    _add_arrow(s1, 7.02, 3.74, 7.02, 5.58)
    _add_arrow(s1, 10.07, 5.25, 10.07, 5.58)

    # ------------------------------------------------------------------
    # SLIDE 2: DeepLabV3+ Architecture & Standard vs THz Adaptation
    # ------------------------------------------------------------------
    s2 = _create_slide_shell(
        prs,
        2,
        total,
        "DeepLabV3+ Architecture & Critical Adaptation for THz C-Scans",
        "How Atrous Spatial Pyramid Pooling (ASPP) + Low-Level Decoder Fusion Works — and Why Stride-16 Fails on 46x46 THz Grids",
        notes_by_num[2],
    )
    _add_rect(s2, 0.30, 0.95, 12.72, 2.82, ALT_ROW, line_rgb=BORDER, rounded=True)
    tb2 = s2.shapes.add_textbox(Inches(0.48), Inches(1.02), Inches(12.2), Inches(0.32))
    r2 = tb2.text_frame.paragraphs[0].add_run()
    r2.text = "DeepLabV3+ Internal Dataflow (Encoder -> Multi-Rate ASPP -> 1/4 Low-Level Skip Decoder)"
    r2.font.name = "Calibri"
    r2.font.size = Pt(13)
    r2.font.bold = True
    r2.font.color.rgb = NAVY

    _add_flow_block(s2, 0.48, 1.80, 1.65, 1.15, "Input Image\n3-Ch Tensor", "RGB: 128x128\nTHz: 128x128", NAVY, 10.0, 8.5)
    _add_flow_block(s2, 2.45, 1.80, 1.90, 1.15, "ResNet-18\nBackbone", "Stage1: 1/4 (32x32)\nStage4: 1/8 (16x16)", BLUE_DARK, 10.0, 8.5)
    _add_flow_block(s2, 4.72, 1.48, 2.40, 0.85, "ASPP Parallel Branches", "1x1 | r=d1 | r=d2 | r=d3 | Pool", TEAL, 10.0, 8.5)
    _add_flow_block(s2, 4.72, 2.66, 2.40, 0.88, "Low-Level Skip (1/4)", "1x1 Conv: 64 -> 48 ch\nPreserves crisp edges", TEAL_DEEP, 10.0, 8.5)
    _add_flow_block(s2, 7.52, 1.80, 2.15, 1.15, "Decoder Fusion", "Upsample ASPP 2x/4x\nConcat + 3x3 Refine", ACCENT, 10.0, 8.5)
    _add_flow_block(s2, 10.05, 1.80, 2.72, 1.15, "Pixel Defect Mask", "Upsample -> 128x128\nSigmoid + Focal/Dice", GREEN, 10.0, 8.5)

    _add_arrow(s2, 2.13, 2.38, 2.45, 2.38)
    _add_arrow(s2, 4.35, 2.15, 4.72, 1.90)
    _add_arrow(s2, 3.40, 2.95, 4.72, 3.10)
    _add_arrow(s2, 7.12, 1.90, 7.52, 2.22)
    _add_arrow(s2, 7.12, 3.10, 7.52, 2.60)
    _add_arrow(s2, 9.67, 2.38, 10.05, 2.38)

    _add_card_box(
        s2,
        0.30,
        3.95,
        5.90,
        3.10,
        "Why Textbook DeepLabV3+ Breaks on Raw THz",
        [
            "• Native THz scans are only 38x38 to 46x46 pixels (1 mm/px pitch).",
            "• Textbook DeepLabV3+ uses ResNet-101 with Output Stride = 16: floor(46 / 16) = 2x2 or 3x3 bottleneck feature grid!",
            "• ASPP dilation rates r = (6, 12, 18) on a 3x3 grid sample 90%+ outside image borders (pure zero-padding) — destroying localization.",
            "• ResNet-101 (~44M params) instantly memorizes N=5..15 scans.",
            "• Our Fix: Bicubic upsample -> 128x128 + ResNet-18 (OS=8 -> 16x16) + physics-scaled ASPP r=(1, 2, 4) spanning 7.5 to 22.5 mm.",
        ],
        header_rgb=RED,
        body_pt=10.0,
    )
    _add_styled_table(
        s2,
        6.45,
        3.95,
        6.57,
        3.10,
        ["Component", "Textbook RGB DeepLabV3+", "Our THz-Adapted DeepLabV3+"],
        [
            ["Input Grid", "512 x 512 RGB", "38–66 px -> 128 x 128 (3-Ch Physics)"],
            ["Backbone", "ResNet-101 (~44M params)", "ResNet-18 (~11M params, dilated)"],
            ["Output Stride", "OS = 16 (3x3 on THz!)", "OS = 8 (16x16 high-level map)"],
            ["Low-Level Skip", "1/4 res (64 ch -> 48 ch)", "32 x 32 res (preserves 3-4 mm edges)"],
            ["ASPP Rates", "r = (6, 12, 18)", "r = (1, 2, 4) tuned to beam & 3x3 arrays"],
            ["Loss Function", "Pixel Cross-Entropy", "Focal + Soft Dice (imbalanced <5% voids)"],
        ],
        [0.24, 0.36, 0.40],
        font_pt=9.5,
    )

    # ------------------------------------------------------------------
    # SLIDE 3: Architectural Rationale & Why Over All Alternatives
    # ------------------------------------------------------------------
    s3 = _create_slide_shell(
        prs,
        3,
        total,
        "Architectural Rationale: Why This Design Over All Alternatives",
        "Trade-Off Analysis Across Segmentation Paradigms, Loss Formulation & Physics-Safe Augmentation Policy",
        notes_by_num[3],
    )
    _add_styled_table(
        s3,
        0.30,
        0.95,
        12.72,
        3.45,
        ["Candidate Architecture", "Role in Project", "Strengths", "Why Rejected (or Restricted) for Primary THz Defect Metrology"],
        [
            ["Decoupled Model 1 +\nTHz-DeepLabV3+ (Ours)", "PRIMARY\nPIPELINE", "Isolates global vs pixel gradients; ASPP + 1/4 fusion traces blurred flaws", "Requires stride-8 & dilation scaling (solved in our THz adaptation)"],
            ["Monolithic Multi-Task Network", "Rejected", "Single forward pass for all 4 outputs", "Severe gradient conflict on N=15..60 scans destabilizes infill % & pixel masks"],
            ["YOLOv8-seg / Mask R-CNN", "Rejected", "Fast bounding-box instance segmentation", "Anchor boxes need sharp edges; fails on 3-4 mm diffraction blur & seams"],
            ["PatchCore / PaDiM\n(Unsupervised)", "Screening\nReference", "Needs only defect-free nominal scans", "Uncalibrated anomaly heatmaps without crisp closed contours for mm^2 area"],
            ["Standard U-Net (SmallUNet)", "Ablation\nBaseline", "Lightweight; strong on central blobs", "Lacks parallel multi-rate ASPP kernels to separate infill from wide seams"],
            ["Vision Transformers\n(SegFormer / Swin)", "Rejected\n(For Now)", "Strong long-range global attention", "Lacks CNN translation equivariance; overfits severely under N < 200 scans"],
        ],
        [0.22, 0.11, 0.31, 0.36],
        font_pt=9.2,
    )
    _add_card_box(
        s3,
        0.30,
        4.60,
        6.18,
        2.45,
        "Loss Function: Why Compound Focal + Soft Dice",
        [
            "• Real defects occupy only 0.2%–2.0% of pixels (e.g., MT_Crack = 0.19%, PLA_1S = 1.47%). Plain BCE collapses to predicting all-zero background.",
            "• Focal Loss (gamma=2.0, alpha=0.80) down-weights easy background pixels and focuses gradients on uncertain diffraction-blurred void boundaries.",
            "• Soft Dice Loss directly optimizes region overlap (IoU/F1) across sparse small defects and thin linear cracks.",
        ],
        header_rgb=TEAL,
        body_pt=10.0,
    )
    _add_card_box(
        s3,
        6.84,
        4.60,
        6.18,
        2.45,
        "Physics-Safe Augmentations (Zero Geometric Distortion)",
        [
            "• Allowed: 90/180/270 deg rotations (linear THz polarization invariance), H/V flips, +-1 px shifts (matches cross-frequency offset), +-10% gain jitter, and Gaussian-blurred synthetic defect injection.",
            "• Forbidden: Elastic warps, arbitrary-angle shear, or color jitter (destroys millimeter calibration & physical attenuation relationships).",
            "• Evaluation Guard: Strict train/test & Leave-One-Scan-Out splits.",
        ],
        header_rgb=ACCENT,
        body_pt=10.0,
    )

    # ------------------------------------------------------------------
    # SLIDE 4: Benchmark 1 — Real Industrial RGB Defect Dataset (24 Samples)
    # ------------------------------------------------------------------
    s4 = _create_slide_shell(
        prs,
        4,
        total,
        "Benchmark 1: Curated Real Industrial RGB Defect Dataset (24 Samples)",
        "Demonstrating DeepLabV3+ Defect Identification & Pixel Localization on Real Ground-Truth Industrial Images",
        notes_by_num[4],
    )
    _add_card_box(
        s4,
        0.30,
        0.95,
        4.75,
        2.62,
        "Why Add Real RGB Benchmark",
        [
            "• Only 5 Cohort-A THz defect scans exist without external CAD masks.",
            "• We curated 24 real industrial defect images + verified pixel masks from the public Magnetic-Tile-Defect dataset.",
            "• Strict Split: 16 Train (4/class) and 8 Held-Out Test (2/class, unseen).",
        ],
        header_rgb=NAVY,
        body_pt=10.2,
    )
    _add_styled_table(
        s4,
        0.30,
        3.75,
        4.75,
        3.30,
        ["Category", "Train / Test", "Defect Area %", "Polymer NDT Analogy"],
        [
            ["MT_Blowhole", "4 / 2", "0.26% – 0.48%", "Blind circular voids (1C)"],
            ["MT_Crack", "4 / 2", "0.19% – 1.84%", "Layer-3 cold-joint seams"],
            ["MT_Break", "4 / 2", "0.29% – 21.9%", "Pocket / edge flaws (1S)"],
            ["MT_Free", "4 / 2", "0.00% (Clean)", "Nominal control (PLA_ND)"],
        ],
        [0.25, 0.19, 0.24, 0.32],
        font_pt=9.0,
    )

    tb4 = s4.shapes.add_textbox(Inches(5.30), Inches(0.95), Inches(7.72), Inches(0.32))
    p4 = tb4.text_frame.paragraphs[0]
    p4.alignment = PP_ALIGN.CENTER
    r4 = p4.add_run()
    r4.text = "Held-Out Real Test Exemplars & Ground-Truth Masks (1 per Category)"
    r4.font.name = "Calibri"
    r4.font.size = Pt(12.5)
    r4.font.bold = True
    r4.font.color.rgb = NAVY

    cat_examples = []
    seen_cats = set()
    for row in rgb_manifest_rows:
        c = row["category"]
        if row["split"] == "test" and c not in seen_cats:
            seen_cats.add(c)
            cat_examples.append(row)

    for idx, row in enumerate(cat_examples[:4]):
        bx = 5.30 + (idx % 2) * 3.92
        by = 1.38 if idx < 2 else 4.25
        # Image + GT Mask pair
        for sub_idx, (path_key, lbl) in enumerate(
            [
                ("image_path", f"{row['category']} ({row['native_height']}x{row['native_width']})"),
                ("mask_path", f"GT Mask (FG={row['defect_area_pct']}%)"),
            ]
        ):
            sx = bx + sub_idx * 1.92
            tb_lbl = s4.shapes.add_textbox(Inches(sx), Inches(by), Inches(1.82), Inches(0.28))
            p_lbl = tb_lbl.text_frame.paragraphs[0]
            p_lbl.alignment = PP_ALIGN.CENTER
            r_lbl = p_lbl.add_run()
            r_lbl.text = lbl
            r_lbl.font.name = "Calibri"
            r_lbl.font.size = Pt(8.8)
            r_lbl.font.bold = True
            r_lbl.font.color.rgb = SLATE
            _add_picture_fit(
                s4,
                Path(row[path_key]) if Path(row[path_key]).exists() else figures_dir.parents[1] / row[path_key],
                sx,
                by + 0.30,
                1.82,
                2.38,
            )

    # ------------------------------------------------------------------
    # SLIDE 5: Real RGB Layer-by-Layer DeepLabV3+ Flow (Figure 07)
    # ------------------------------------------------------------------
    s5 = _create_slide_shell(
        prs,
        5,
        total,
        "How DeepLabV3+ Works Inside: Real RGB Layer-by-Layer Flow",
        "Live Feature Activations on Unseen Held-Out Test Samples (MT_Blowhole & MT_Crack) Through ResNet-18, ASPP & Decoder",
        notes_by_num[5],
    )
    _add_picture_fit(s5, figures_dir / "07_real_rgb_deeplabv3plus_architecture_flow.png", 0.25, 0.90, 12.83, 5.20)
    _add_card_box(
        s5,
        0.30,
        6.18,
        12.72,
        0.92,
        "Key Architectural Takeaway",
        [
            "Low-Level 1/4 (32x32) preserves fine defect edges despite vertical brushed texture; ASPP branches (r=2,4,8 + Global Pool) separate localized pits & hairline cracks from global illumination; Decoder fusion sharpens contours.",
        ],
        header_rgb=TEAL,
        title_pt=10.5,
        body_pt=9.8,
    )

    # ------------------------------------------------------------------
    # SLIDE 6: Real RGB Held-Out Test Results Gallery & Metrics (Figure 08)
    # ------------------------------------------------------------------
    s6 = _create_slide_shell(
        prs,
        6,
        total,
        "Real RGB Held-Out Test Results: Identification & Localization",
        "Evaluation on All 8 Strictly Unseen Industrial Test Samples (6 Defective + 2 Nominal Defect-Free Controls)",
        notes_by_num[6],
    )
    _add_picture_fit(s6, figures_dir / "08_real_rgb_heldout_test_samples_results.png", 0.20, 0.90, 9.25, 6.20)

    summary_rgb = rgb_metrics["summary"]
    per_sample_rows = [
        [
            r["sample_id"].replace("MT_", ""),
            r["ground_truth_status"],
            r["predicted_status"],
            f"{r['ap']:.3f}" if r["ap"] is not None else "N/A",
            f"{r['dice']:.3f}",
        ]
        for r in rgb_metrics["per_sample_test_metrics"]
    ]
    _add_card_box(
        s6,
        9.55,
        0.95,
        3.50,
        2.12,
        "Held-Out Test Summary (N = 8)",
        [
            f"• Identification Acc: {summary_rgb['identification_accuracy']*100:.1f}% ({summary_rgb['correct_identifications']}/8)",
            f"• Mean Defective AP: {summary_rgb['mean_defective_ap']:.3f}",
            f"• Mean Defective Dice: {summary_rgb['mean_defective_dice']:.3f}",
            f"• Mean Defective IoU: {summary_rgb['mean_defective_iou']:.3f}",
            f"• Clean Control False Alarms: {summary_rgb['clean_false_alarm_pixels']} px",
        ],
        header_rgb=GREEN,
        title_pt=10.8,
        body_pt=9.4,
    )
    _add_styled_table(
        s6,
        9.55,
        3.18,
        3.50,
        3.90,
        ["Test Sample", "GT", "Pred", "AP", "Dice"],
        per_sample_rows,
        [0.44, 0.12, 0.14, 0.15, 0.15],
        font_pt=8.0,
    )

    # ------------------------------------------------------------------
    # SLIDE 7: Benchmark 2 — 15 CSIR THz Scans & Conditioning (Figs 01 & 02)
    # ------------------------------------------------------------------
    s7 = _create_slide_shell(
        prs,
        7,
        total,
        "Benchmark 2: CSIR-CSIO Terahertz Polymer Dataset (15 Scans)",
        "CSIR Physical Distance (mm) Scales & High-Saturation Contrast Across All 4 Materials, Infills & Defect Cases",
        notes_by_num[7],
    )
    _add_picture_fit(s7, figures_dir / "01_dataset_15scans_overview.png", 0.22, 0.90, 7.22, 4.72)
    _add_picture_fit(s7, figures_dir / "02_preprocessing_and_hotstrip_correction.png", 7.52, 0.90, 5.58, 4.72)
    _add_card_box(
        s7,
        0.30,
        5.70,
        12.72,
        1.40,
        "15-Scan Taxonomy & CSIR Scale / Saturation Upgrade",
        [
            "• Defect Cases: 1 No-Defect control (ND), 2 Single-Void scans (1C/1S), 2 Uniform 3x3 Arrays (3C/3S), and 10 Multi-Defect Mixed Plates (Cohort B).",
            "• Material & Infill Coverage: 4 polymers (PLA, PETG, ABS, TPU), 1 shared printing pattern, across 15%, 45%, 75%, and 100% infill (135–450 GHz).",
            "• CSIR Visual Standard: Physical Distance (mm) axes + interior-anchored (p4–p96) jet stretch removes hot-strip compression so defects pop clearly.",
        ],
        header_rgb=NAVY,
        title_pt=10.8,
        body_pt=9.6,
    )

    # ------------------------------------------------------------------
    # SLIDE 8: THz-Adapted DeepLabV3+ Internal Activations (Figure 04)
    # ------------------------------------------------------------------
    s8 = _create_slide_shell(
        prs,
        8,
        total,
        "How THz-Adapted DeepLabV3+ Works on Real CSIR Scans",
        "Internal Feature Maps (mm Scales & Saturated Overlays) on Held-Out PLA_1C (450 GHz) & PLA_3S (200 GHz)",
        notes_by_num[8],
    )
    _add_picture_fit(s8, figures_dir / "04_thz_deeplabv3plus_internal_activations.png", 0.25, 0.90, 12.83, 5.05)
    _add_card_box(
        s8,
        0.30,
        6.05,
        12.72,
        1.05,
        "What These Real THz Activations Show",
        [
            "Calibrated in Distance (mm) with high-saturation overlays: ResNet-18 (OS=8) yields 32x32 edges and 16x16 ASPP grid—r=1 isolates single 3–4 mm cavities (PLA_1C), while r=2 & r=4 trace periodic 3x3 arrays (PLA_3S).",
        ],
        header_rgb=ACCENT,
        title_pt=10.8,
        body_pt=9.8,
    )

    # ------------------------------------------------------------------
    # SLIDE 9: CSIR THz Test Samples — LOSO Localization & Metrology (Fig 05)
    # ------------------------------------------------------------------
    s9 = _create_slide_shell(
        prs,
        9,
        total,
        "CSIR THz Test Samples: Held-Out LOSO Localization & Metrology",
        "Leave-One-Scan-Out (THz-DeepLabV3+ vs SmallUNet) + Multi-Defect Plate Inference & mm Metrology",
        notes_by_num[9],
    )
    _add_picture_fit(s9, figures_dir / "05_thz_defect_localization_all_samples.png", 0.20, 0.90, 8.15, 6.20)

    thz_loso = thz_metrics["thz_loso_verification"]
    thz_rows = []
    for f in thz_loso["per_fold_metrics"]:
        thz_rows.append(
            [
                f["held_out_scan"].replace("_100p", ""),
                f"{f['deeplabv3plus_ap']:.3f}" if f["deeplabv3plus_ap"] is not None else "N/A (ND)",
                f"{f['unet_ap']:.3f}" if f["unet_ap"] is not None else "N/A (ND)",
                f"{f['deeplabv3plus_dice']:.3f}",
                str(f["deeplab_defects_detected"]),
            ]
        )
    thz_rows.append(
        [
            "MEAN (4 Defect)",
            f"{thz_loso['mean_deeplabv3plus_ap']:.3f}",
            f"{thz_loso['mean_unet_ap']:.3f}",
            f"{thz_loso['mean_deeplabv3plus_dice']:.3f}",
            "—",
        ]
    )
    _add_styled_table(
        s9,
        8.45,
        0.95,
        4.60,
        2.95,
        ["Held-Out Scan", "DL AP", "UNet AP", "DL Dice", "Defects"],
        thz_rows,
        [0.40, 0.14, 0.16, 0.15, 0.15],
        font_pt=8.4,
    )
    _add_card_box(
        s9,
        8.45,
        4.05,
        4.60,
        3.03,
        "Metrology Findings & Honest Caveats",
        [
            "• PLA_1S (325 GHz): AP=0.999, Dice=0.984 (1 defect, beam-corrected diameter d=5.3 mm at center).",
            "• PLA_ND (300 GHz): PASS with 0 false-alarm pixels.",
            "• PLA_1C (450 GHz): Localized center void (24.5, 21.9 mm, d=5.5 mm, conf=0.959) + triggered on sample bevel edge.",
            "• Multi-Defect Cohort B: Zero-shot inference highlights mixed plate anomalies on ABS & PLA under mm scales.",
            "• Caveat: Masks are physics-derived until CAD truth arrives.",
        ],
        header_rgb=NAVY,
        title_pt=10.8,
        body_pt=9.3,
    )

    # ------------------------------------------------------------------
    # SLIDE 10: Model 1 Audit, Constraints & Future Improvements
    # ------------------------------------------------------------------
    s10 = _create_slide_shell(
        prs,
        10,
        total,
        "Model 1 Audit, Key Constraints & Future Improvements Roadmap",
        "4-Material & Multi-Infill Audit (Same Pattern) + Actionable Next Steps for CSIR-CSIO & SRM GPU Lab",
        notes_by_num[10],
    )
    _add_picture_fit(s10, figures_dir / "06_model1_material_and_infill_separability.png", 0.25, 0.90, 7.20, 4.45)
    _add_card_box(
        s10,
        0.30,
        5.42,
        7.15,
        1.65,
        "Model 1 Frequency Confounder Discovered in Cohort B",
        [
            "• At 400 GHz & 100% infill, PLA (+0.0581), ABS (+0.0600), and TPU (+0.0619) overlap within IQR.",
            "• PETG appears lower (+0.0300 at 100%, -0.0863 at 15%) ONLY because it ran at 450/425 GHz vs 400/375 GHz. All 15 scans also share 1 pattern -> fixed-freq multi-pattern scans needed.",
        ],
        header_rgb=RED,
        title_pt=10.6,
        body_pt=9.5,
    )
    _add_card_box(
        s10,
        7.65,
        0.95,
        5.38,
        2.98,
        "Future Model & Pipeline Improvements",
        [
            "1. Transfer Learning: Pretrain ResNet-18 DeepLabV3+ on industrial defect RGB data -> fine-tune on THz scans.",
            "2. Plate-Mask Extraction: Segment outer polymer boundary first to suppress bevel edge false alarms (seen in PLA_1C).",
            "3. Multi-Class & Phase Fusion: Separate blind/buried defect kinds via 350+500 GHz magnitude + ToF phase fusion.",
            "4. SRM GPU Lab Scaling: Run 5-fold GroupKFold sweeps in our CUDA Docker container (docker/Dockerfile) at N>=45.",
        ],
        header_rgb=TEAL,
        title_pt=10.8,
        body_pt=9.6,
    )
    _add_card_box(
        s10,
        7.65,
        4.08,
        5.38,
        2.98,
        "Priority Requests for CSIR-CSIO Next Batch",
        [
            "1. Fixed Acquisition Settings: Same frequency (e.g., 400 or 500 GHz) + fixed gain + empty-holder reference scan.",
            "2. Ground-Truth Metrology Table: True CAD/caliper defect diameter, depth (mm), and center (X, Y) for calibration.",
            "3. Confirmed Buried & Mixed Defects: Layer-3 power-cut prints paired with clean twins (~25-45 scans).",
            "4. Pattern Diversity: Grid, Gyroid, Concentric prints at 20%-60% infill to enable Model 1 Pattern classification.",
        ],
        header_rgb=GREEN,
        title_pt=10.8,
        body_pt=9.6,
    )

    out_path = Path(out_pptx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
