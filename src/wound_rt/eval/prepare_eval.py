from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stage1 eval manifest")
    parser.add_argument("--valid-manifest", default="artifacts/manifests/valid_manifest.csv")
    parser.add_argument("--non-wound-manifest", default="artifacts/negatives/non_wound_manifest.csv")
    parser.add_argument("--out-csv", default="artifacts/eval/stage1_eval_manifest.csv")
    parser.add_argument("--negative-sample-size", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid = pd.read_csv(args.valid_manifest)
    wound_eval = valid[valid["image_exists"] == True].copy()  # noqa: E712
    wound_eval = wound_eval[["image_path"]].copy()
    wound_eval["label"] = 1

    neg = pd.read_csv(args.non_wound_manifest)
    neg = neg[["image_path"]].copy()
    if len(neg) > args.negative_sample_size:
        neg = neg.sample(args.negative_sample_size, random_state=42)
    neg["label"] = 0

    out = pd.concat([wound_eval, neg], ignore_index=True)
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
