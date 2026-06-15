from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class Stage1BinaryClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 1)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)


class Stage2MultiHeadClassifier(nn.Module):
    def __init__(self, num_classes: dict[str, int]) -> None:
        super().__init__()
        backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_features, n_cls) for name, n_cls in num_classes.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {name: head(feat) for name, head in self.heads.items()}
