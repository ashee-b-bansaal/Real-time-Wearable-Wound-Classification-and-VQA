from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from wound_rt.models.datasets import META_COLUMNS


RPI_STAGE2_BACKBONES = [
    "resnext50_32x4d",
    "efficientformer_l1",
    "efficientnet_l",
    "efficientnet_lite0",
    "efficientnet_lite1",
    "efficientnet_lite2",
    "efficientnet_lite3",
    "efficientnet_lite4",
    "efficientnet_m",
    "efficientnet_s",
]


def _require_timm():
    try:
        import timm  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Backbone requires `timm`. Install with `pip install timm` for Raspberry/Hailo training."
        ) from e
    return timm


def _build_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    name = backbone_name.lower()
    if name == "resnext50_32x4d":
        backbone = tvm.resnext50_32x4d(
            weights=tvm.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, in_features

    timm_name_by_alias = {
        "efficientformer_l1": "efficientformer_l1",
        "efficientnet_lite0": "efficientnet_lite0",
        "efficientnet_lite1": "efficientnet_lite1",
        "efficientnet_lite2": "efficientnet_lite2",
        "efficientnet_lite3": "efficientnet_lite3",
        "efficientnet_lite4": "efficientnet_lite4",
        # Practical mappings for common stage2 backbones requested by user.
        "efficientnet_s": "tf_efficientnetv2_s",
        "efficientnet_m": "tf_efficientnetv2_m",
        "efficientnet_l": "tf_efficientnetv2_l",
    }
    if name not in timm_name_by_alias:
        raise ValueError(
            f"Unsupported Raspberry Stage2 backbone '{backbone_name}'. "
            f"Use one of: {', '.join(RPI_STAGE2_BACKBONES)}"
        )
    timm = _require_timm()
    timm_name = timm_name_by_alias[name]
    backbone = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )
    in_features = int(backbone.num_features)
    return backbone, in_features


class RaspberryStage2MultiHeadClassifier(nn.Module):
    def __init__(
        self,
        num_classes: dict[str, int],
        backbone_name: str = "efficientformer_l1",
        head_dropout: float = 0.2,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        backbone, in_features = _build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
        )
        self.backbone = backbone
        self.dropout = nn.Dropout(p=head_dropout)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_features, n_cls) for name, n_cls in num_classes.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        feat = self.dropout(feat)
        return {name: head(feat) for name, head in self.heads.items()}


class RaspberryStage2TupleWrapper(nn.Module):
    def __init__(self, model: RaspberryStage2MultiHeadClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(x)
        return tuple(out[h] for h in META_COLUMNS)

