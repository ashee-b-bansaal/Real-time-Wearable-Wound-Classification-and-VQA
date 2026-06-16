from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from wound_rt.models.datasets import LabelEncoders, META_COLUMNS, image_transform
from wound_rt.models.networks import Stage1BinaryClassifier, Stage2MultiHeadClassifier


def _normalize_image_path(image_path: str) -> Path:
    # Make manifests portable across Windows/macOS/Linux.
    p = Path(str(image_path).strip().replace("\\", "/"))
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _resolve_stage1_model_name(stage1_model_path: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    meta_path = stage1_model_path.parent / "stage1_model_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return str(meta.get("model_name", "mobilenet_v3_small"))
    return "mobilenet_v3_small"


def benchmark_stage1(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_model_path = Path(args.stage1_model)
    stage1_model_name = _resolve_stage1_model_name(stage1_model_path, args.stage1_model_name)
    model = Stage1BinaryClassifier(model_name=stage1_model_name).to(device)
    model.load_state_dict(torch.load(stage1_model_path, map_location=device))
    model.eval()

    df = pd.read_csv(args.stage1_eval_manifest)
    df = df[df["image_path"].apply(lambda p: _normalize_image_path(str(p)).exists())]
    if df.empty:
        raise ValueError(
            "No readable stage1 eval images found. "
            "Your manifest likely has OS-mismatched paths. Rebuild manifests on this machine."
        )
    tf = image_transform(train=False, size=args.image_size)
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    latencies_ms: list[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="benchmark stage1"):
        img_path = _normalize_image_path(str(row["image_path"]))
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = tf(image=rgb)["image"].unsqueeze(0).to(device)
        t0 = time.time()
        with torch.no_grad():
            prob = float(torch.sigmoid(model(x)).item())
        latencies_ms.append((time.time() - t0) * 1000.0)
        y_true.append(int(row["label"]))
        y_prob.append(prob)
        y_pred.append(int(prob >= args.stage1_threshold))

    metrics = {
        "stage1_model_name": stage1_model_name,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mae_probability": float(
            np.mean(np.abs(np.array(y_true, dtype=np.float32) - np.array(y_prob, dtype=np.float32)))
        ),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
    }
    return metrics


def benchmark_stage2(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Path(args.label_encoder_json).open("r", encoding="utf-8") as f:
        enc_map = json.load(f)
    num_classes = {k: len(v) for k, v in enc_map.items()}
    model = Stage2MultiHeadClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.stage2_model, map_location=device))
    model.eval()

    val = pd.read_csv(args.stage2_eval_manifest)
    val = val[val["image_exists"] == True].copy()  # noqa: E712
    val = val[val["image_path"].apply(lambda p: _normalize_image_path(str(p)).exists())].copy()
    if val.empty:
        raise ValueError(
            "No readable stage2 eval images found. "
            "Your manifest likely has OS-mismatched paths. Rebuild manifests on this machine."
        )
    encoder = LabelEncoders(mapping=enc_map)
    tf = image_transform(train=False, size=args.image_size)

    ys = {k: [] for k in META_COLUMNS}
    ps = {k: [] for k in META_COLUMNS}
    latencies_ms: list[float] = []
    for _, row in tqdm(val.iterrows(), total=len(val), desc="benchmark stage2"):
        img_path = _normalize_image_path(str(row["image_path"]))
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = tf(image=rgb)["image"].unsqueeze(0).to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model(x)
        latencies_ms.append((time.time() - t0) * 1000.0)
        y = encoder.encode_row(row)
        for head in META_COLUMNS:
            ys[head].append(y[head])
            ps[head].append(int(torch.argmax(out[head], dim=1).item()))

    if any(len(ys[head]) == 0 for head in META_COLUMNS):
        raise ValueError(
            "Stage2 benchmark collected zero valid samples after image loading. "
            "Rebuild manifests on this machine and verify image paths."
        )
    head_f1 = {head: f1_score(ys[head], ps[head], average="macro", zero_division=0) for head in META_COLUMNS}
    head_mae = {
        head: float(np.mean(np.abs(np.array(ys[head], dtype=np.float32) - np.array(ps[head], dtype=np.float32))))
        for head in META_COLUMNS
    }
    head_zero_one_loss = {
        head: float(np.mean((np.array(ys[head], dtype=np.int32) != np.array(ps[head], dtype=np.int32)).astype(np.float32)))
        for head in META_COLUMNS
    }
    metrics = {
        "macro_f1": float(np.mean(list(head_f1.values())) if head_f1 else 0.0),
        "mean_metadata_mae_index": float(np.mean(list(head_mae.values())) if head_mae else 0.0),
        "mean_zero_one_loss": float(np.mean(list(head_zero_one_loss.values())) if head_zero_one_loss else 0.0),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "metadata_mae_note": "MAE is computed on encoded class indices; use macro-F1 as primary metric for categorical labels.",
        "zero_one_loss_note": "Per-head 0/1 loss: 0 if predicted class equals true class, else 1. Equivalent to (1 - accuracy).",
    }
    metrics.update({f"f1_{k}": float(v) for k, v in head_f1.items()})
    metrics.update({f"mae_{k}": float(v) for k, v in head_mae.items()})
    metrics.update({f"zero_one_loss_{k}": float(v) for k, v in head_zero_one_loss.items()})
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark stage1 and stage2 models")
    parser.add_argument("--stage1-model", default="artifacts/models/stage1/stage1_best.pt")
    parser.add_argument(
        "--stage1-model-name",
        default=None,
        help=(
            "Optional stage1 backbone (mobilenet_v3_small, resnet50, efficientnet_b0, efficientnet_b1, efficientnet_b5). "
            "If omitted, inferred from stage1_model_meta.json when present."
        ),
    )
    parser.add_argument("--stage2-model", default="artifacts/models/stage2/stage2_best.pt")
    parser.add_argument("--label-encoder-json", default="artifacts/models/stage2/label_encoders.json")
    parser.add_argument("--stage1-eval-manifest", default="artifacts/eval/stage1_eval_manifest.csv")
    parser.add_argument("--stage2-eval-manifest", default="artifacts/manifests/valid_manifest.csv")
    parser.add_argument("--stage1-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--out-json", default="artifacts/eval/benchmark_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "stage1": benchmark_stage1(args),
        "stage2": benchmark_stage2(args),
    }
    with Path(args.out_json).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
