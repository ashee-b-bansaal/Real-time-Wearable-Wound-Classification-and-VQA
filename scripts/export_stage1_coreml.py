from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wound_rt.mobile.coreml_export import export_stage1_coreml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage1 PyTorch model to CoreML")
    parser.add_argument("--stage1-model", default="artifacts/models/stage1/stage1_best.pt")
    parser.add_argument("--stage1-model-name", default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-path", default="artifacts/mobile/Stage1.mlpackage")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = export_stage1_coreml(
        stage1_model_path=Path(args.stage1_model),
        stage1_model_name=args.stage1_model_name,
        image_size=args.image_size,
        output_path=Path(args.output_path),
    )
    print(f"Saved Stage1 CoreML model to {out}")


if __name__ == "__main__":
    main()

