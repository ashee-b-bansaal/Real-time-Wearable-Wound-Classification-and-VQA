from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wound_rt.mobile.coreml_export import export_mobile_config, export_stage2_coreml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage2 PyTorch model to CoreML")
    parser.add_argument("--stage2-model", default="artifacts/models/stage2/stage2_best.pt")
    parser.add_argument("--label-encoder-json", default="artifacts/models/stage2/label_encoders.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--stage1-threshold", type=float, default=0.45)
    parser.add_argument("--stable-frames-required", type=int, default=6)
    parser.add_argument("--nebulon-delay-sec", type=float, default=2.0)
    parser.add_argument("--output-path", default="artifacts/mobile/Stage2.mlpackage")
    parser.add_argument("--mobile-config-path", default="artifacts/mobile/mobile_config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = export_stage2_coreml(
        stage2_model_path=Path(args.stage2_model),
        label_encoder_json_path=Path(args.label_encoder_json),
        image_size=args.image_size,
        output_path=Path(args.output_path),
    )
    cfg_path = export_mobile_config(
        label_encoder_json_path=Path(args.label_encoder_json),
        output_path=Path(args.mobile_config_path),
        stage1_threshold=args.stage1_threshold,
        stable_frames_required=args.stable_frames_required,
        nebulon_delay_sec=args.nebulon_delay_sec,
        image_size=args.image_size,
    )
    print(f"Saved Stage2 CoreML model to {out}")
    print(f"Saved mobile config to {cfg_path}")


if __name__ == "__main__":
    main()

