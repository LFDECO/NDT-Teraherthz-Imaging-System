"""Model 2: DeepLabV3+ supporting both standard RGB Demo Mode and THz-Adapted NDT Mode."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedBasicBlock(nn.Module):
    """ResNet BasicBlock supporting arbitrary dilation > 1 without torchvision restrictions."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class DilatedResNetEncoder(nn.Module):
    """Configurable ResNet-18/34 encoder with output_stride 8 or 16 for DeepLabV3+."""

    def __init__(
        self,
        in_channels: int = 3,
        backbone: str = "resnet18",
        output_stride: int = 8,
    ) -> None:
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError(f"output_stride must be 8 or 16, got {output_stride}")

        layers_cfg = [2, 2, 2, 2] if backbone == "resnet18" else [3, 4, 6, 3]
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 1 -> 1/4 spatial resolution (low-level features for decoder fusion)
        self.layer1 = self._make_layer(64, layers_cfg[0], stride=1, dilation=1)
        # Stage 2 -> 1/8 spatial resolution
        self.layer2 = self._make_layer(128, layers_cfg[1], stride=2, dilation=1)

        if output_stride == 8:
            # Keep 1/8 resolution across stages 3 & 4 via dilated convolutions
            self.layer3 = self._make_layer(256, layers_cfg[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(512, layers_cfg[3], stride=1, dilation=4)
        else:
            # Output stride 16: downsample at stage 3, dilate at stage 4
            self.layer3 = self._make_layer(256, layers_cfg[2], stride=2, dilation=1)
            self.layer4 = self._make_layer(512, layers_cfg[3], stride=1, dilation=2)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers: List[nn.Module] = [
            DilatedBasicBlock(
                self.inplanes,
                planes,
                stride=stride,
                dilation=1 if stride == 2 else dilation,
                downsample=downsample,
            )
        ]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(DilatedBasicBlock(self.inplanes, planes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        stem = self.relu(self.bn1(self.conv1(x)))  # 1/2
        p = self.maxpool(stem)                      # 1/4
        low_level = self.layer1(p)                  # 1/4 (64 ch)
        s2 = self.layer2(low_level)                 # 1/8 (128 ch)
        s3 = self.layer3(s2)                        # 1/8 or 1/16 (256 ch)
        high_level = self.layer4(s3)                # 1/8 or 1/16 (512 ch)
        return {
            "stem": stem,
            "low_level": low_level,
            "stage2": s2,
            "high_level": high_level,
        }


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (1x1 + 3 dilated branches + global image pool)."""

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 128,
        rates: Tuple[int, int, int] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.rates = rates

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch_d1 = self._make_atrous(in_channels, out_channels, rates[0])
        self.branch_d2 = self._make_atrous(in_channels, out_channels, rates[1])
        self.branch_d3 = self._make_atrous(in_channels, out_channels, rates[2])

        self.branch_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    @staticmethod
    def _make_atrous(in_ch: int, out_ch: int, rate: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h, w = x.shape[-2:]
        b0 = self.branch_1x1(x)
        b1 = self.branch_d1(x)
        b2 = self.branch_d2(x)
        b3 = self.branch_d3(x)
        b_pool = F.interpolate(self.branch_pool(x), size=(h, w), mode="bilinear", align_corners=False)

        concat = torch.cat([b0, b1, b2, b3, b_pool], dim=1)
        fused = self.project(concat)
        branches = {
            "1x1": b0,
            f"rate_{self.rates[0]}": b1,
            f"rate_{self.rates[1]}": b2,
            f"rate_{self.rates[2]}": b3,
            "image_pool": b_pool,
        }
        return fused, branches


class DeepLabV3Plus(nn.Module):
    """Dual-mode DeepLabV3+ for RGB demonstration and THz NDT defect segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: str = "resnet18",
        output_stride: int = 8,
        aspp_rates: Iterable[int] = (1, 2, 4),
        aspp_out_channels: int = 128,
        low_level_proj_channels: int = 48,
    ) -> None:
        super().__init__()
        rates_tuple = tuple(int(r) for r in aspp_rates)
        if len(rates_tuple) != 3:
            raise ValueError(f"Expected 3 ASPP dilation rates, got {rates_tuple}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.output_stride = output_stride
        self.aspp_rates = rates_tuple

        self.encoder = DilatedResNetEncoder(
            in_channels=in_channels,
            backbone=backbone,
            output_stride=output_stride,
        )
        self.aspp = ASPP(in_channels=512, out_channels=aspp_out_channels, rates=rates_tuple)

        # Decoder low-level 1x1 projection (64 -> 48 channels per Chen et al. 2018)
        self.low_level_reduce = nn.Sequential(
            nn.Conv2d(64, low_level_proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_level_proj_channels),
            nn.ReLU(inplace=True),
        )

        dec_in = aspp_out_channels + low_level_proj_channels
        self.decoder_refine = nn.Sequential(
            nn.Conv2d(dec_in, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward_with_intermediates(self, x: torch.Tensor) -> Dict[str, object]:
        input_hw = x.shape[-2:]
        enc = self.encoder(x)
        aspp_fused, aspp_branches = self.aspp(enc["high_level"])

        low_proj = self.low_level_reduce(enc["low_level"])
        aspp_up = F.interpolate(
            aspp_fused,
            size=low_proj.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder_cat = torch.cat([aspp_up, low_proj], dim=1)
        decoder_fused = self.decoder_refine(decoder_cat)
        logits_low = self.classifier(decoder_fused)
        logits = F.interpolate(logits_low, size=input_hw, mode="bilinear", align_corners=False)

        probs = torch.sigmoid(logits) if self.num_classes == 1 else torch.softmax(logits, dim=1)
        return {
            "stem": enc["stem"],
            "low_level_feat": enc["low_level"],
            "low_level_proj": low_proj,
            "encoder_high_feat": enc["high_level"],
            "aspp_branches": aspp_branches,
            "aspp_fused": aspp_fused,
            "decoder_fused": decoder_fused,
            "logits": logits,
            "probs": probs,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(x)["logits"]  # type: ignore[return-value]
