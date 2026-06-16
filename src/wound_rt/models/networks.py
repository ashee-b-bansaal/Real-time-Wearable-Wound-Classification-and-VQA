from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class Stage1BinaryClassifier(nn.Module):
    def __init__(self, model_name: str = "mobilenet_v3_small") -> None:
        super().__init__()
        self.model_name = model_name
        self.model = build_stage1_model(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)


def build_stage1_model(model_name: str) -> nn.Module:
    name = model_name.lower()
    if name == "mobilenet_v3_small":
        backbone = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 1)
        return backbone
    if name == "resnet50":
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        return backbone
    if name == "efficientnet_b0":
        backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 1)
        return backbone
    if name == "efficientnet_b1":
        backbone = tvm.efficientnet_b1(weights=tvm.EfficientNet_B1_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 1)
        return backbone
    if name == "efficientnet_b5":
        backbone = tvm.efficientnet_b5(weights=tvm.EfficientNet_B5_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 1)
        return backbone
    raise ValueError(
        f"Unsupported stage1 model '{model_name}'. "
        "Use one of: mobilenet_v3_small, resnet50, efficientnet_b0, efficientnet_b1, efficientnet_b5."
    )


class Stage2MultiHeadClassifier(nn.Module):
    def __init__(self, num_classes: dict[str, int], head_dropout: float = 0.3) -> None:
        super().__init__()
        backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=head_dropout)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_features, n_cls) for name, n_cls in num_classes.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return {name: head(feat) for name, head in self.heads.items()}
