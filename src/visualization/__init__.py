"""Visualization utilities for CSIR-CSIO DeepLabV3+ and THz dataset presentations."""

from .csir_demo_plots import (
    plot_dataset_15scans_overview,
    plot_deeplabv3plus_rgb_demo,
    plot_model1_separability,
    plot_preprocessing_and_hotstrip,
    plot_real_rgb_deeplab_flow,
    plot_real_rgb_test_gallery,
    plot_thz_deeplabv3plus_activations,
    plot_thz_localization_gallery,
)
from .pdf_presentation import build_csir_meeting_pdf
from .pptx_presentation import build_csir_meeting_pptx
from .speaker_guide import (
    build_side_by_side_speaker_guide_docx,
    build_side_by_side_speaker_guide_md,
    get_csir_speaker_content,
)

__all__ = [
    "plot_dataset_15scans_overview",
    "plot_deeplabv3plus_rgb_demo",
    "plot_model1_separability",
    "plot_preprocessing_and_hotstrip",
    "plot_real_rgb_deeplab_flow",
    "plot_real_rgb_test_gallery",
    "plot_thz_deeplabv3plus_activations",
    "plot_thz_localization_gallery",
    "build_csir_meeting_pdf",
    "build_csir_meeting_pptx",
    "build_side_by_side_speaker_guide_docx",
    "build_side_by_side_speaker_guide_md",
    "get_csir_speaker_content",
]
