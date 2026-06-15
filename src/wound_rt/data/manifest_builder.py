from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_COLUMNS = [
    "anatomic_locations",
    "wound_type",
    "wound_thickness",
    "tissue_color",
    "drainage_amount",
    "drainage_type",
    "infection",
]


@dataclass(frozen=True)
class ManifestConfig:
    challenge_json_dir: Path
    full_json_path: Path
    image_root: Path
    challenge_images_dir: Path
    output_dir: Path


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _norm_multi_label(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(str(x).strip() for x in value if str(x).strip())
    if value is None:
        return ""
    return str(value).strip()


def records_from_entries(
    entries: list[dict[str, Any]],
    split_name: str,
    image_root: Path,
    fallback_roots: list[Path] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [image_root] + (fallback_roots or [])
    for entry in entries:
        base_row = {
            "encounter_id": entry.get("encounter_id", ""),
            "split": split_name,
        }
        for label in LABEL_COLUMNS:
            base_row[label] = _norm_multi_label(entry.get(label))
        image_ids = entry.get("image_ids", [])
        for image_id in image_ids:
            p = image_root / image_id
            if not p.exists():
                for r in roots[1:]:
                    cand = r / image_id
                    if cand.exists():
                        p = cand
                        break
            row = dict(base_row)
            row["image_id"] = image_id
            row["image_path"] = str(p)
            row["image_exists"] = p.exists()
            rows.append(row)
    return rows


def build_manifests(cfg: ManifestConfig) -> dict[str, pd.DataFrame]:
    train_entries = _load_json(cfg.challenge_json_dir / "train.json")
    valid_entries = _load_json(cfg.challenge_json_dir / "valid.json")
    test_entries = _load_json(cfg.challenge_json_dir / "test.json")
    full_entries = _load_json(cfg.full_json_path)

    challenge_roots_by_split = {
        "train": cfg.challenge_images_dir / "images_train",
        "valid": cfg.challenge_images_dir / "images_valid",
        "test": cfg.challenge_images_dir / "images_test",
    }
    challenge_fallbacks = [cfg.image_root, cfg.challenge_images_dir / "images_train"]

    manifests = {
        "train": pd.DataFrame(
            records_from_entries(
                train_entries,
                "train",
                challenge_roots_by_split["train"],
                fallback_roots=challenge_fallbacks,
            )
        ),
        "valid": pd.DataFrame(
            records_from_entries(
                valid_entries,
                "valid",
                challenge_roots_by_split["valid"],
                fallback_roots=challenge_fallbacks,
            )
        ),
        "test": pd.DataFrame(
            records_from_entries(
                test_entries,
                "test",
                challenge_roots_by_split["test"],
                fallback_roots=challenge_fallbacks,
            )
        ),
        "full": pd.DataFrame(records_from_entries(full_entries, "full", cfg.image_root)),
    }
    return manifests


def save_manifests(manifests: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for split_name, df in manifests.items():
        csv_path = out_dir / f"{split_name}_manifest.csv"
        df.to_csv(csv_path, index=False)
        summary_rows.append(
            {
                "split": split_name,
                "rows": len(df),
                "unique_encounters": df["encounter_id"].nunique() if not df.empty else 0,
                "image_exists_rate": float(df["image_exists"].mean()) if "image_exists" in df.columns and not df.empty else 0.0,
                "manifest_path": str(csv_path),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "manifest_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wound dataset manifests")
    parser.add_argument(
        "--challenge-json-dir",
        type=Path,
        default=Path("osfstorage-archive (1)/dataset-challenge-mediqa-2025-wv"),
    )
    parser.add_argument(
        "--full-json-path",
        type=Path,
        default=Path("osfstorage-archive (1)/dataset-full-original/woundcarevqa.json"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("osfstorage-archive (1)/dataset-full-original/images_final"),
    )
    parser.add_argument(
        "--challenge-images-dir",
        type=Path,
        default=Path("osfstorage-archive (1)/dataset-challenge-mediqa-2025-wv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/manifests"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ManifestConfig(
        challenge_json_dir=args.challenge_json_dir,
        full_json_path=args.full_json_path,
        image_root=args.image_root,
        challenge_images_dir=args.challenge_images_dir,
        output_dir=args.output_dir,
    )
    manifests = build_manifests(cfg)
    save_manifests(manifests, cfg.output_dir)
    summary = pd.read_csv(cfg.output_dir / "manifest_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
