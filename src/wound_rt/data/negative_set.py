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
            rows.append({"image_id": p.name, "image_path": str(p), "label": 0, "split": "train"})
    return pd.DataFrame(rows)


def load_dataset2_normals(dataset2_root: Path, normal_classes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    split_map = {"Train": "train", "Test": "valid"}
    for split_dir, split_name in split_map.items():
        for normal_cls in normal_classes:
            cls_dir = dataset2_root / split_dir / normal_cls
            if not cls_dir.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                for p in cls_dir.rglob(ext):
                    rows.append(
                        {
                            "image_id": p.name,
                            "image_path": str(p),
                            "label": 0,
                            "split": split_name,
                            "source": f"dataset2/{split_dir}/{normal_cls}",
                        }
                    )
    return pd.DataFrame(rows)


def load_negative_dirs(extra_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for base_dir in extra_dirs:
        if not base_dir.exists():
            continue
        split_name = "valid" if "test" in str(base_dir).lower() else "train"
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for p in base_dir.rglob(ext):
                rows.append(
                    {
                        "image_id": p.name,
                        "image_path": str(p),
                        "label": 0,
                        "split": split_name,
                        "source": str(base_dir),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create non-wound negatives")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/negatives/captured"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--every-n-frames", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=600)
    parser.add_argument("--skip-camera-capture", action="store_true")
    parser.add_argument("--public-negatives-dir", type=Path, default=None)
    parser.add_argument("--dataset2-root", type=Path, default=None)
    parser.add_argument(
        "--dataset2-normal-classes",
        nargs="+",
        default=["N"],
        help="Class folders in dataset2 treated as non-wound (normal)",
    )
    parser.add_argument(
        "--extra-negative-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Extra folders to include as non-wound images (e.g., dataset3/Nomal dataset2/Train/BG)",
    )
    parser.add_argument("--manifest-out", type=Path, default=Path("artifacts/negatives/non_wound_manifest.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dfs: list[pd.DataFrame] = []
    if not args.skip_camera_capture:
        capture_df = capture_non_wound_images(
            CaptureConfig(
                output_dir=args.output_dir,
                camera_index=args.camera_index,
                every_n_frames=args.every_n_frames,
                max_images=args.max_images,
            )
        )
        if not capture_df.empty:
            capture_df = capture_df.assign(split="train", source="camera_capture")
            dfs.append(capture_df)

    if args.public_negatives_dir:
        dfs.append(merge_public_negatives(args.public_negatives_dir))
    if args.dataset2_root:
        dfs.append(load_dataset2_normals(args.dataset2_root, args.dataset2_normal_classes))
    if args.extra_negative_dirs:
        dfs.append(load_negative_dirs(args.extra_negative_dirs))

    if not dfs:
        raise ValueError("No negatives collected. Use camera capture and/or provide --dataset2-root/public-negatives-dir")

    all_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["image_path"])
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.manifest_out, index=False)
    print(f"Saved {len(all_df)} non-wound samples -> {args.manifest_out}")
    if "split" in all_df.columns:
        print(all_df["split"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
