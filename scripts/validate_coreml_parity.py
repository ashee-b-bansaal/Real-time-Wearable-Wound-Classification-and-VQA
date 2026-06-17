from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wound_rt.mobile.parity import run_parity_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate parity between PyTorch and CoreML models")
    parser.add_argument("--manifest", default="artifacts/manifests/valid_manifest.csv")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--stage1-model", default="artifacts/models/stage1/stage1_best.pt")
    parser.add_argument("--stage1-model-name", default=None)
    parser.add_argument("--stage1-coreml", default="artifacts/mobile/Stage1.mlpackage")
    parser.add_argument("--stage2-model", default="artifacts/models/stage2/stage2_best.pt")
    parser.add_argument("--stage2-coreml", default="artifacts/mobile/Stage2.mlpackage")
    parser.add_argument("--label-encoder-json", default="artifacts/models/stage2/label_encoders.json")
    parser.add_argument("--output-path", default="artifacts/mobile/parity_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    if "image_exists" in df.columns:
        df = df[df["image_exists"] == True].copy()  # noqa: E712
    img_paths: list[Path] = []
    for p in df["image_path"].tolist():
        ip = Path(str(p))
        if ip.exists():
            img_paths.append(ip)
        if len(img_paths) >= args.max_samples:
            break

    result = run_parity_check(
        images=img_paths,
        stage1_model_path=Path(args.stage1_model),
        stage1_mlmodel_path=Path(args.stage1_coreml),
        stage2_model_path=Path(args.stage2_model),
        stage2_mlmodel_path=Path(args.stage2_coreml),
        label_encoder_json_path=Path(args.label_encoder_json),
        image_size=args.image_size,
        stage1_model_name=args.stage1_model_name,
    )

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(json.dumps(result.to_dict(), indent=2))
    print(f"Saved parity report to {output}")


if __name__ == "__main__":
    main()

