"""Bone age < 11y classifier: ImageNet-pretrained backbone + linear head.

The whole model (backbone + head) is fine-tuned end-to-end (train.py),
like the regression model.
"""

import torch
import torch.nn as nn
import timm
from torchvision.models import (
    efficientnet_b3, EfficientNet_B3_Weights,
    vit_b_16, ViT_B_16_Weights,
)

BACKBONES = ["efficientnet_b3", "vit_b_16", "convnextv2_tiny"]


class _TimmPooled(nn.Module):
    """timm model: forward_features -> global avg pool -> flatten."""
    def __init__(self, name):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=0)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return torch.flatten(self.pool(self.model.forward_features(x)), 1)


def build_backbone(name):
    """Returns (ImageNet-pretrained backbone, feature dim)."""
    if name == "efficientnet_b3":
        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        num_ftrs = backbone.classifier[1].in_features  # 1536
        backbone.classifier = nn.Identity()
    elif name == "vit_b_16":
        backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        num_ftrs = backbone.heads.head.in_features  # 768
        backbone.heads = nn.Identity()
    elif name == "convnextv2_tiny":
        backbone = _TimmPooled("convnextv2_tiny")
        num_ftrs = backbone.model.num_features  # 768
    else:
        raise ValueError(f"Unknown backbone: {name!r}")
    return backbone, num_ftrs


class LinearHead(nn.Module):
    """logit = w . [features, sex] + b"""
    def __init__(self, num_ftrs):
        super().__init__()
        self.fc = nn.Linear(num_ftrs + 1, 1)

    def forward(self, feat, sex):
        return self.fc(torch.cat((feat, sex), dim=1))


class BoneAgeClassifier(nn.Module):
    """End-to-end model: image + sex -> logit of P(bone age < 11y)."""
    def __init__(self, name="efficientnet_b3"):
        super().__init__()
        self.backbone, num_ftrs = build_backbone(name)
        self.head = LinearHead(num_ftrs)

    def forward(self, img, sex):
        return self.head(self.backbone(img), sex)
