from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd


@dataclass(frozen=True)
class CaptureConfig:
    output_dir: Path
    camera_index: int
    every_n_frames: int
    max_images: int


def capture_non_wound_images(cfg: CaptureConfig) -> pd.DataFrame:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {cfg.camera_index}")

    frame_idx = 0
    saved = 0
    rows: list[dict[str, str | int]] = []
    while saved < cfg.max_images:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        preview = frame.copy()
        cv2.putText(
            preview,
            "Press q to quit, s to force-save frame",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("non_wound_capture", preview)

        key = cv2.waitKey(1) & 0xFF
        should_save = frame_idx % cfg.every_n_frames == 0 or key == ord("s")
        if should_save:
            name = f"non_wound_{saved:05d}.jpg"
            out_path = cfg.output_dir / name
            cv2.imwrite(str(out_path), frame)
            rows.append({"image_id": name, "image_path": str(out_path), "label": 0})
            saved += 1
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return pd.DataFrame(rows)


def merge_public_negatives(public_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for p in public_dir.rglob(ext):
            rows.append({"image_id": p.name, "image_path": str(p), "label": 0})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create non-wound negatives")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/negatives/captured"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--every-n-frames", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=600)
    parser.add_argument("--public-negatives-dir", type=Path, default=None)
    parser.add_argument("--manifest-out", type=Path, default=Path("artifacts/negatives/non_wound_manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture_df = capture_non_wound_images(
        CaptureConfig(
            output_dir=args.output_dir,
            camera_index=args.camera_index,
            every_n_frames=args.every_n_frames,
            max_images=args.max_images,
        )
    )
    dfs = [capture_df]
    if args.public_negatives_dir:
        dfs.append(merge_public_negatives(args.public_negatives_dir))

    all_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["image_path"])
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.manifest_out, index=False)
    print(f"Saved {len(all_df)} non-wound samples -> {args.manifest_out}")


if __name__ == "__main__":
    main()
