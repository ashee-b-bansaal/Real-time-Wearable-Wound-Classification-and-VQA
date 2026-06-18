from __future__ import annotations

import json
from pathlib import Path

import torch

from wound_rt.models.datasets import META_COLUMNS
from wound_rt.models.networks import Stage1BinaryClassifier
from wound_rt.raspberry.networks import RaspberryStage2MultiHeadClassifier, RaspberryStage2TupleWrapper


def _require_onnx_runtime_deps() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("ONNX export requires `onnx`. Install with `pip install onnx`.") from e


def _resolve_stage1_model_name(stage1_model_path: Path, explicit_model_name: str | None) -> str:
    if explicit_model_name:
        return explicit_model_name
    meta_path = stage1_model_path.parent / "stage1_model_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return str(meta.get("model_name", "mobilenet_v3_small"))
    return "mobilenet_v3_small"


def export_stage1_onnx(
    *,
    stage1_model_path: Path,
    output_path: Path,
    image_size: int,
    stage1_model_name: str | None = None,
    opset_version: int = 17,
) -> Path:
    _require_onnx_runtime_deps()
    model_name = _resolve_stage1_model_name(stage1_model_path, stage1_model_name)
    model = Stage1BinaryClassifier(model_name=model_name).eval()
    model.load_state_dict(torch.load(stage1_model_path, map_location="cpu"))
    example = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(output_path),
        input_names=["input_image"],
        output_names=["stage1_logit"],
        dynamic_axes={"input_image": {0: "batch"}, "stage1_logit": {0: "batch"}},
        opset_version=opset_version,
        do_constant_folding=True,
    )
    return output_path


def export_stage2_onnx(
    *,
    stage2_model_path: Path,
    stage2_meta_path: Path,
    output_path: Path,
    image_size: int,
    opset_version: int = 17,
) -> Path:
    _require_onnx_runtime_deps()
    with stage2_meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    backbone = str(meta["backbone"])
    num_classes = {k: int(v) for k, v in dict(meta["num_classes"]).items()}
    model = RaspberryStage2MultiHeadClassifier(
        num_classes=num_classes,
        backbone_name=backbone,
        head_dropout=0.0,
        pretrained_backbone=False,
    ).eval()
    model.load_state_dict(torch.load(stage2_model_path, map_location="cpu"))
    wrapper = RaspberryStage2TupleWrapper(model).eval()
    example = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_names = [f"{head}_logits" for head in META_COLUMNS]
    dyn_axes: dict[str, dict[int, str]] = {"input_image": {0: "batch"}}
    for out in output_names:
        dyn_axes[out] = {0: "batch"}
    torch.onnx.export(
        wrapper,
        example,
        str(output_path),
        input_names=["input_image"],
        output_names=output_names,
        dynamic_axes=dyn_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    return output_path


def write_android_runtime_config(
    *,
    label_encoder_json_path: Path,
    output_path: Path,
    stage1_threshold: float = 0.45,
    stable_frames_required: int = 3,
    stage2_infer_every_n_frames: int = 3,
) -> Path:
    with label_encoder_json_path.open("r", encoding="utf-8") as f:
        enc = json.load(f)
    id_to_label = {
        head: {int(idx): label for label, idx in mapping.items()}
        for head, mapping in enc.items()
    }
    config = {
        "version": 1,
        "input_size": 224,
        "preprocess": {
            "input_format": "NHWC_RGB_float32",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "stage1": {
            "threshold": stage1_threshold,
            "label_negative": "NOT WOUND",
            "label_positive": "WOUND",
        },
        "stage2": {
            "heads_order": META_COLUMNS,
            "id_to_label": id_to_label,
            "infer_every_n_frames": stage2_infer_every_n_frames,
        },
        "temporal": {
            "stable_frames_required": stable_frames_required,
            "clear_metadata_after_non_wound_frames": 5,
        },
        "networking": {
            "nebulon_optional": True,
            "nebulon_delay_sec": 2.0,
            "nebulon_overlay_max_lines": 5,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return output_path

