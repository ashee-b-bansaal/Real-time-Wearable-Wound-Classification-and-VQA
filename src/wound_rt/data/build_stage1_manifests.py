from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def collect_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        files.extend(folder.rglob(ext))
    return sorted(set(files))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage1 manifests from folder-based wound/no_wound dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset_stage1"))
    parser.add_argument("--wound-dir-name", default="wound")
    parser.add_argument("--non-wound-dir-name", default="no_wound")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stage1_manifests"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wound_dir = args.dataset_root / args.wound_dir_name
    non_wound_dir = args.dataset_root / args.non_wound_dir_name
    if not wound_dir.exists():
        raise FileNotFoundError(f"Wound folder not found: {wound_dir}")
    if not non_wound_dir.exists():
        raise FileNotFoundError(f"Non-wound folder not found: {non_wound_dir}")

    wound_files = collect_images(wound_dir)
    non_wound_files = collect_images(non_wound_dir)

    wound_df = pd.DataFrame(
        {
            "encounter_id": [f"stage1_wound_{i:06d}" for i in range(len(wound_files))],
            "image_id": [p.name for p in wound_files],
            "image_path": [str(p) for p in wound_files],
            "image_exists": [True] * len(wound_files),
        }
    )
    wound_df = wound_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_valid_w = max(1, int(len(wound_df) * args.valid_ratio))
    valid_wound = wound_df.iloc[:n_valid_w].copy()
    train_wound = wound_df.iloc[n_valid_w:].copy()

    non_wound_df = pd.DataFrame(
        {
            "image_id": [p.name for p in non_wound_files],
            "image_path": [str(p) for p in non_wound_files],
            "label": [0] * len(non_wound_files),
        }
    ).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_valid_nw = max(1, int(len(non_wound_df) * args.valid_ratio))
    non_wound_df["split"] = "train"
    non_wound_df.iloc[:n_valid_nw, non_wound_df.columns.get_loc("split")] = "valid"
    non_wound_df["source"] = str(non_wound_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_wound_path = args.out_dir / "train_wound_manifest.csv"
    valid_wound_path = args.out_dir / "valid_wound_manifest.csv"
    non_wound_path = args.out_dir / "non_wound_manifest.csv"
    train_wound.to_csv(train_wound_path, index=False)
    valid_wound.to_csv(valid_wound_path, index=False)
    non_wound_df.to_csv(non_wound_path, index=False)

    print(f"train_wound: {len(train_wound)} -> {train_wound_path}")
    print(f"valid_wound: {len(valid_wound)} -> {valid_wound_path}")
    print(
        "non_wound: "
        f"{len(non_wound_df)} (train={(non_wound_df['split']=='train').sum()}, valid={(non_wound_df['split']=='valid').sum()}) "
        f"-> {non_wound_path}"
    )


if __name__ == "__main__":
    main()
