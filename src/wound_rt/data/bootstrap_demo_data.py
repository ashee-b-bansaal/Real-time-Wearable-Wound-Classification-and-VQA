from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _seed_from_text(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def make_synthetic_wound_image(key: str, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_text(key))
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = (rng.integers(20, 70), rng.integers(40, 90), rng.integers(90, 140))
    center = (int(rng.integers(70, 185)), int(rng.integers(70, 185)))
    axes = (int(rng.integers(25, 65)), int(rng.integers(15, 55)))
    angle = int(rng.integers(0, 180))
    color = (int(rng.integers(30, 80)), int(rng.integers(30, 80)), int(rng.integers(160, 235)))
    cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
    for _ in range(5):
        p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        p2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.line(img, p1, p2, (int(rng.integers(0, 20)), int(rng.integers(0, 20)), int(rng.integers(0, 30))), 1)
    return img


def make_synthetic_non_wound_image(key: str, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_text(key))
    img = np.zeros((size, size, 3), dtype=np.uint8)
    base = int(rng.integers(120, 190))
    img[:, :] = (base - 20, base - 10, base)
    noise = rng.normal(0, 7, size=(size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def materialize_missing_wound_images(manifest_path: Path) -> int:
    df = pd.read_csv(manifest_path)
    created = 0
    for _, row in df.iterrows():
        p = Path(row["image_path"])
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            img = make_synthetic_wound_image(f"{row['encounter_id']}|{row['image_id']}")
            cv2.imwrite(str(p), img)
            created += 1
    return created


def make_non_wound_set(out_dir: Path, count: int) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(count):
        name = f"synthetic_non_wound_{i:05d}.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), make_synthetic_non_wound_image(name))
        rows.append({"image_id": name, "image_path": str(path), "label": 0})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic bootstrap data for smoke training")
    parser.add_argument("--train-manifest", default="artifacts/manifests/train_manifest.csv")
    parser.add_argument("--valid-manifest", default="artifacts/manifests/valid_manifest.csv")
    parser.add_argument("--test-manifest", default="artifacts/manifests/test_manifest.csv")
    parser.add_argument("--non-wound-count", type=int, default=800)
    parser.add_argument("--non-wound-out-dir", default="artifacts/negatives/captured")
    parser.add_argument("--non-wound-manifest-out", default="artifacts/negatives/non_wound_manifest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created = 0
    for m in (Path(args.train_manifest), Path(args.valid_manifest), Path(args.test_manifest)):
        if m.exists():
            created += materialize_missing_wound_images(m)

    neg_df = make_non_wound_set(Path(args.non_wound_out_dir), count=args.non_wound_count)
    out_path = Path(args.non_wound_manifest_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    neg_df.to_csv(out_path, index=False)
    print(f"Created {created} synthetic wound images and {len(neg_df)} non-wound images")


if __name__ == "__main__":
    main()
