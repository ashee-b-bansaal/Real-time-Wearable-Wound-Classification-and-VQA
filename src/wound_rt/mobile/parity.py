from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from wound_rt.models.datasets import META_COLUMNS
from wound_rt.models.networks import Stage1BinaryClassifier, Stage2MultiHeadClassifier
from wound_rt.mobile.coreml_export import resolve_stage1_model_name


def _require_coremltools():
    try:
        import coremltools as ct  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("coremltools is required for parity checks.") from e
    return ct


def _normalize_rgb_chw(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


@dataclass
class ParityResult:
    samples_checked: int
    stage1_max_abs_prob_diff: float
    stage1_mean_abs_prob_diff: float
    stage2_head_label_agreement: dict[str, float]
    stage2_mean_label_agreement: float

    def to_dict(self) -> dict[str, object]:
        return {
            "samples_checked": self.samples_checked,
            "stage1_max_abs_prob_diff": self.stage1_max_abs_prob_diff,
            "stage1_mean_abs_prob_diff": self.stage1_mean_abs_prob_diff,
            "stage2_head_label_agreement": self.stage2_head_label_agreement,
            "stage2_mean_label_agreement": self.stage2_mean_label_agreement,
        }


def run_parity_check(
    *,
    images: list[Path],
    stage1_model_path: Path,
    stage1_mlmodel_path: Path,
    stage2_model_path: Path,
    stage2_mlmodel_path: Path,
    label_encoder_json_path: Path,
    image_size: int = 224,
    stage1_model_name: str | None = None,
) -> ParityResult:
    _require_coremltools()
    from coremltools.models import MLModel  # type: ignore

    with label_encoder_json_path.open("r", encoding="utf-8") as f:
        enc = json.load(f)
    num_classes = {k: len(v) for k, v in enc.items()}
    dev = torch.device("cpu")
    s1_name = resolve_stage1_model_name(stage1_model_path, stage1_model_name)
    stage1 = Stage1BinaryClassifier(model_name=s1_name).to(dev).eval()
    stage1.load_state_dict(torch.load(stage1_model_path, map_location=dev))
    stage2 = Stage2MultiHeadClassifier(num_classes=num_classes).to(dev).eval()
    stage2.load_state_dict(torch.load(stage2_model_path, map_location=dev))

    s1_ml = MLModel(str(stage1_mlmodel_path))
    s2_ml = MLModel(str(stage2_mlmodel_path))

    s1_diffs: list[float] = []
    agree_counts = {head: 0 for head in META_COLUMNS}
    usable = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        x_np = _normalize_rgb_chw(img, size=image_size)
        x_t = torch.from_numpy(x_np)
        with torch.no_grad():
            s1_pt_prob = float(torch.sigmoid(stage1(x_t)).item())
            s2_pt_logits = stage2(x_t)
        s1_pred = s1_ml.predict({"input_image": x_np})
        s2_pred = s2_ml.predict({"input_image": x_np})
        s1_cm_prob = float(1.0 / (1.0 + np.exp(-float(np.array(s1_pred["stage1_logit"]).ravel()[0]))))
        s1_diffs.append(abs(s1_pt_prob - s1_cm_prob))
        for head in META_COLUMNS:
            pt_idx = int(torch.argmax(s2_pt_logits[head], dim=1).item())
            cm_logits = np.array(s2_pred[f"{head}_logits"]).ravel()
            cm_idx = int(np.argmax(cm_logits))
            if pt_idx == cm_idx:
                agree_counts[head] += 1
        usable += 1

    if usable == 0:
        return ParityResult(
            samples_checked=0,
            stage1_max_abs_prob_diff=0.0,
            stage1_mean_abs_prob_diff=0.0,
            stage2_head_label_agreement={h: 0.0 for h in META_COLUMNS},
            stage2_mean_label_agreement=0.0,
        )
    per_head = {head: agree_counts[head] / usable for head in META_COLUMNS}
    return ParityResult(
        samples_checked=usable,
        stage1_max_abs_prob_diff=float(np.max(s1_diffs) if s1_diffs else 0.0),
        stage1_mean_abs_prob_diff=float(np.mean(s1_diffs) if s1_diffs else 0.0),
        stage2_head_label_agreement=per_head,
        stage2_mean_label_agreement=float(np.mean(list(per_head.values()))),
    )

