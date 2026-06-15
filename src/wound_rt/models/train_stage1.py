from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from wound_rt.models.datasets import Stage1BinaryDataset
from wound_rt.models.networks import Stage1BinaryClassifier


def _load_wounds(train_wound_manifest: Path, valid_wound_manifest: Path | None) -> pd.DataFrame:
    train_wound = pd.read_csv(train_wound_manifest)
    train_wound = train_wound[train_wound["image_exists"] == True].copy()  # noqa: E712
    train_wound["label"] = 1
    train_wound["split"] = "train"
    dfs = [train_wound]
    if valid_wound_manifest and valid_wound_manifest.exists():
        valid_wound = pd.read_csv(valid_wound_manifest)
        valid_wound = valid_wound[valid_wound["image_exists"] == True].copy()  # noqa: E712
        valid_wound["label"] = 1
        valid_wound["split"] = "valid"
        dfs.append(valid_wound)
    return pd.concat(dfs, ignore_index=True)


def read_data(
    train_wound_manifest: Path, valid_wound_manifest: Path | None, non_wound_manifest: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wound = _load_wounds(train_wound_manifest, valid_wound_manifest)
    neg = pd.read_csv(non_wound_manifest)
    if "label" not in neg.columns:
        neg["label"] = 0
    if "split" not in neg.columns:
        neg["split"] = "train"
    all_df = pd.concat(
        [
            wound[["image_path", "label", "split"]],
            neg[["image_path", "label", "split"]],
        ],
        ignore_index=True,
    )
    val = all_df[all_df["split"] == "valid"].copy()
    train = all_df[all_df["split"] != "valid"].copy()
    return train, val


def eval_loop(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            y_true.extend(y.long().numpy().tolist())
            y_pred.extend(pred.tolist())
    if len(y_true) == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_wound = Path(args.valid_wound_manifest) if args.valid_wound_manifest else None
    train_df, val_df = read_data(Path(args.wound_manifest), valid_wound, Path(args.non_wound_manifest))

    train_ds = Stage1BinaryDataset(train_df, train=True, size=args.image_size)
    val_ds = Stage1BinaryDataset(val_df, train=False, size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = Stage1BinaryClassifier(model_name=args.model_name).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"stage1 epoch {epoch + 1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        metrics = eval_loop(model, val_loader, device=device)
        print(
            f"epoch={epoch + 1} train_loss={np.mean(losses):.4f} "
            f"recall={metrics['recall']:.4f} precision={metrics['precision']:.4f} f1={metrics['f1']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), out_dir / "stage1_best.pt")
            with (out_dir / "stage1_model_meta.json").open("w", encoding="utf-8") as f:
                json.dump({"model_name": args.model_name, "image_size": args.image_size}, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage1 wound/non-wound model")
    parser.add_argument("--wound-manifest", default="artifacts/manifests/train_manifest.csv")
    parser.add_argument("--valid-wound-manifest", default="artifacts/manifests/valid_manifest.csv")
    parser.add_argument("--non-wound-manifest", default="artifacts/negatives/non_wound_manifest.csv")
    parser.add_argument("--output-dir", default="artifacts/models/stage1")
    parser.add_argument(
        "--model-name",
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "resnet50", "efficientnet_b0", "efficientnet_b1", "efficientnet_b5"],
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
