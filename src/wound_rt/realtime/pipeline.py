from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from wound_rt.models.datasets import META_COLUMNS, image_transform
from wound_rt.models.networks import Stage1BinaryClassifier, Stage2MultiHeadClassifier
from wound_rt.realtime.dimensions import MarkerConfig, estimate_dimensions
from wound_rt.realtime.quality import QualityThresholds, compute_quality


@dataclass(frozen=True)
class RuntimeConfig:
    stage1_threshold: float = 0.5
    stable_frames_required: int = 6
    output_jsonl: Path = Path("artifacts/realtime/live_outputs.jsonl")


def _to_input(frame_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tf = image_transform(train=False, size=size)
    x = tf(image=rgb)["image"].unsqueeze(0)
    return x


def load_models(stage1_path: Path, stage2_path: Path, encoder_json: Path) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1 = Stage1BinaryClassifier().to(device)
    stage1.load_state_dict(torch.load(stage1_path, map_location=device))
    stage1.eval()
    with encoder_json.open("r", encoding="utf-8") as f:
        enc = json.load(f)
    num_classes = {k: len(v) for k, v in enc.items()}
    stage2 = Stage2MultiHeadClassifier(num_classes=num_classes).to(device)
    stage2.load_state_dict(torch.load(stage2_path, map_location=device))
    stage2.eval()
    inv_enc = {k: {int(i): label for label, i in mapping.items()} for k, mapping in enc.items()}
    return stage1, stage2, inv_enc, device


def run_realtime(args: argparse.Namespace) -> None:
    stage1, stage2, inv_enc, device = load_models(
        Path(args.stage1_model), Path(args.stage2_model), Path(args.label_encoder_json)
    )
    runtime = RuntimeConfig(
        stage1_threshold=args.stage1_threshold,
        stable_frames_required=args.stable_frames_required,
        output_jsonl=Path(args.output_jsonl),
    )
    runtime.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cam = cv2.VideoCapture(args.camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_index}")

    quality_thr = QualityThresholds(
        min_laplacian_var=args.min_laplacian_var,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_contrast_std=args.min_contrast_std,
        max_motion_mad=args.max_motion_mad,
    )
    marker_cfg = MarkerConfig(
        marker_width_mm=args.marker_width_mm,
        marker_height_mm=args.marker_height_mm,
        min_area_px=args.marker_min_area_px,
    )

    prev_gray: np.ndarray | None = None
    stable_count = 0
    fps_window = deque(maxlen=30)
    last_t = time.time()

    with runtime.output_jsonl.open("a", encoding="utf-8") as out_f:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            now = time.time()
            fps_window.append(1.0 / max(1e-6, now - last_t))
            last_t = now

            q = compute_quality(frame, prev_gray=prev_gray, thr=quality_thr)
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            x = _to_input(frame, size=args.image_size).to(device)
            with torch.no_grad():
                wound_prob = float(torch.sigmoid(stage1(x)).item())
            is_wound = wound_prob >= runtime.stage1_threshold

            if is_wound and q.pass_quality:
                stable_count += 1
            else:
                stable_count = 0

            payload: dict[str, object] = {
                "timestamp_ms": int(now * 1000),
                "wound_prob": wound_prob,
                "is_wound": is_wound,
                "quality_pass": q.pass_quality,
                "quality": {
                    "laplacian_var": q.laplacian_var,
                    "brightness": q.brightness,
                    "contrast_std": q.contrast_std,
                    "motion_mad": q.motion_mad,
                },
                "stable_count": stable_count,
                "fps_estimate": float(np.mean(fps_window)) if fps_window else 0.0,
            }

            if stable_count >= runtime.stable_frames_required:
                with torch.no_grad():
                    meta_logits = stage2(x)
                metadata: dict[str, dict[str, object]] = {}
                for head in META_COLUMNS:
                    probs = torch.softmax(meta_logits[head], dim=1).cpu().numpy()[0]
                    idx = int(np.argmax(probs))
                    metadata[head] = {
                        "label": inv_enc[head][idx],
                        "confidence": float(probs[idx]),
                    }
                dims = estimate_dimensions(frame, cfg=marker_cfg)
                payload["metadata"] = metadata
                payload["dimensions"] = {
                    "status": dims.status,
                    "area_mm2": dims.area_mm2,
                    "major_axis_mm": dims.major_axis_mm,
                    "minor_axis_mm": dims.minor_axis_mm,
                    "px_per_mm": dims.px_per_mm,
                }
                stable_count = 0

            out_f.write(json.dumps(payload) + "\n")
            out_f.flush()

            status_text = f"wound={wound_prob:.2f} quality={int(q.pass_quality)} stable={stable_count}"
            cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("wound_realtime", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cam.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime wound metadata pipeline")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--stage1-model", default="artifacts/models/stage1/stage1_best.pt")
    parser.add_argument("--stage2-model", default="artifacts/models/stage2/stage2_best.pt")
    parser.add_argument("--label-encoder-json", default="artifacts/models/stage2/label_encoders.json")
    parser.add_argument("--output-jsonl", default="artifacts/realtime/live_outputs.jsonl")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--stage1-threshold", type=float, default=0.5)
    parser.add_argument("--stable-frames-required", type=int, default=6)
    parser.add_argument("--min-laplacian-var", type=float, default=80.0)
    parser.add_argument("--min-brightness", type=float, default=35.0)
    parser.add_argument("--max-brightness", type=float, default=225.0)
    parser.add_argument("--min-contrast-std", type=float, default=20.0)
    parser.add_argument("--max-motion-mad", type=float, default=16.0)
    parser.add_argument("--marker-width-mm", type=float, default=20.0)
    parser.add_argument("--marker-height-mm", type=float, default=20.0)
    parser.add_argument("--marker-min-area-px", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    run_realtime(parse_args())


if __name__ == "__main__":
    main()
