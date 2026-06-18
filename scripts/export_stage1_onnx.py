from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wound_rt.android.export_onnx import export_stage1_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage1 checkpoint to ONNX for Android runtime")
    parser.add_argument("--stage1-model", default="artifacts/models/stage1/stage1_best.pt")
    parser.add_argument("--stage1-model-name", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--output-path", default="artifacts/android/Stage1.onnx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = export_stage1_onnx(
        stage1_model_path=Path(args.stage1_model),
        output_path=Path(args.output_path),
        image_size=args.image_size,
        stage1_model_name=args.stage1_model_name,
        opset_version=args.opset,
    )
    print(f"Saved Stage1 ONNX to {out}")


if __name__ == "__main__":
    main()

