from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from wound_rt.models.datasets import META_COLUMNS
from wound_rt.models.networks import Stage1BinaryClassifier, Stage2MultiHeadClassifier


def _require_coremltools():
    try:
        import coremltools as ct  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "coremltools is required for iPhone export. Install with `pip install coremltools`."
        ) from e
    return ct


def resolve_stage1_model_name(stage1_model_path: Path, explicit_model_name: str | None) -> str:
    if explicit_model_name:
        return explicit_model_name
    meta_path = stage1_model_path.parent / "stage1_model_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return str(meta.get("model_name", "mobilenet_v3_small"))
    return "mobilenet_v3_small"


class Stage2TupleWrapper(nn.Module):
    def __init__(self, model: Stage2MultiHeadClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(x)
        return tuple(out[head] for head in META_COLUMNS)


def export_stage1_coreml(
    *,
    stage1_model_path: Path,
    output_path: Path,
    image_size: int = 224,
    stage1_model_name: str | None = None,
) -> Path:
    ct = _require_coremltools()
    model_name = resolve_stage1_model_name(stage1_model_path, stage1_model_name)
    device = torch.device("cpu")
    model = Stage1BinaryClassifier(model_name=model_name).to(device)
    model.load_state_dict(torch.load(stage1_model_path, map_location=device))
    model.eval()
    example = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_image", shape=example.shape)],
        outputs=[ct.TensorType(name="stage1_logit")],
        minimum_deployment_target=ct.target.iOS17,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return output_path


def export_stage2_coreml(
    *,
    stage2_model_path: Path,
    label_encoder_json_path: Path,
    output_path: Path,
    image_size: int = 224,
) -> Path:
    ct = _require_coremltools()
    with label_encoder_json_path.open("r", encoding="utf-8") as f:
        enc = json.load(f)
    num_classes = {k: len(v) for k, v in enc.items()}
    device = torch.device("cpu")
    stage2 = Stage2MultiHeadClassifier(num_classes=num_classes).to(device)
    stage2.load_state_dict(torch.load(stage2_model_path, map_location=device))
    stage2.eval()
    wrapped = Stage2TupleWrapper(stage2).to(device).eval()
    example = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    traced = torch.jit.trace(wrapped, example)
    output_specs = [ct.TensorType(name=f"{head}_logits") for head in META_COLUMNS]
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_image", shape=example.shape)],
        outputs=output_specs,
        minimum_deployment_target=ct.target.iOS17,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return output_path


def export_mobile_config(
    *,
    label_encoder_json_path: Path,
    output_path: Path,
    stage1_threshold: float = 0.45,
    stable_frames_required: int = 6,
    nebulon_delay_sec: float = 2.0,
    image_size: int = 224,
) -> Path:
    with label_encoder_json_path.open("r", encoding="utf-8") as f:
        enc = json.load(f)
    id_to_label = {
        head: {int(idx): label for label, idx in mapping.items()}
        for head, mapping in enc.items()
    }
    config = {
        "version": 1,
        "image_size": image_size,
        "preprocess": {
            "input_format": "CHW_RGB_float32",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "stage1": {
            "classes": ["not_wound", "wound"],
            "threshold": stage1_threshold,
        },
        "stage2": {
            "heads_order": META_COLUMNS,
            "id_to_label": id_to_label,
        },
        "temporal": {
            "stable_frames_required": stable_frames_required,
            "clear_metadata_after_non_wound_frames": 5,
        },
        "nebulon": {
            "optional_remote": True,
            "delay_sec": nebulon_delay_sec,
        },
        "dimensions": {
            "marker_width_mm": 20.0,
            "marker_height_mm": 20.0,
            "marker_min_area_px": 400,
            "marker_hsv_low": [95, 80, 50],
            "marker_hsv_high": [135, 255, 255],
            "lab_a_threshold": 145,
            "small_contour_area_px": 50,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return output_path

