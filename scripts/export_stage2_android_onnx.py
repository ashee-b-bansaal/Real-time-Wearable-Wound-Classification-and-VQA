from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wound_rt.android.export_onnx import export_stage2_onnx, write_android_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Raspberry Stage2 checkpoint to ONNX for Android runtime")
    parser.add_argument("--stage2-model", default="artifacts/models/raspberry/stage2_efficientformer_l1/stage2_rpi_best.pt")
    parser.add_argument("--stage2-meta", default="artifacts/models/raspberry/stage2_efficientformer_l1/stage2_rpi_model_meta.json")
    parser.add_argument("--label-encoder-json", default="artifacts/models/raspberry/stage2_efficientformer_l1/label_encoders.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-path", default="artifacts/android/Stage2.onnx")
    parser.add_argument("--runtime-config-out", default="artifacts/android/android_runtime_config.json")
    parser.add_argument("--stage1-threshold", type=float, default=0.45)
    parser.add_argument("--stable-frames-required", type=int, default=3)
    parser.add_argument("--stage2-infer-every-n-frames", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = export_stage2_onnx(
        stage2_model_path=Path(args.stage2_model),
        stage2_meta_path=Path(args.stage2_meta),
        output_path=Path(args.output_path),
        image_size=args.image_size,
        opset_version=args.opset,
    )
    cfg = write_android_runtime_config(
        label_encoder_json_path=Path(args.label_encoder_json),
        output_path=Path(args.runtime_config_out),
        stage1_threshold=args.stage1_threshold,
        stable_frames_required=args.stable_frames_required,
        stage2_infer_every_n_frames=args.stage2_infer_every_n_frames,
    )
    print(f"Saved Stage2 ONNX to {out}")
    print(f"Saved Android runtime config to {cfg}")


if __name__ == "__main__":
    main()

