from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from wound_rt.models.datasets import LabelEncoders, META_COLUMNS, Stage2MultiHeadDataset
from wound_rt.models.networks import Stage2MultiHeadClassifier


def macro_f1_by_head(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    ys: dict[str, list[int]] = {h: [] for h in META_COLUMNS}
    ps: dict[str, list[int]] = {h: [] for h in META_COLUMNS}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            for head in META_COLUMNS:
                pred = torch.argmax(out[head], dim=1).cpu().numpy().tolist()
                ps[head].extend(pred)
                ys[head].extend(y[head].cpu().numpy().tolist())
    scores: dict[str, float] = {}
    for head in META_COLUMNS:
        if not ys[head]:
            scores[head] = 0.0
        else:
            scores[head] = f1_score(ys[head], ps[head], average="macro", zero_division=0)
    return scores


def macro_f1_by_head_on_train(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    # Same computation as validation F1, but on training loader for epoch diagnostics.
    return macro_f1_by_head(model=model, loader=loader, device=device)


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(args.train_manifest)
    valid_df = pd.read_csv(args.valid_manifest)
    train_df = train_df[train_df["image_exists"] == True].copy()  # noqa: E712
    valid_df = valid_df[valid_df["image_exists"] == True].copy()  # noqa: E712

    enc = LabelEncoders.from_dataframe(train_df[META_COLUMNS])
    with (out_dir / "label_encoders.json").open("w", encoding="utf-8") as f:
        json.dump(enc.mapping, f, indent=2)

    train_ds = Stage2MultiHeadDataset(train_df, encoders=enc, train=True, size=args.image_size)
    valid_ds = Stage2MultiHeadDataset(valid_df, encoders=enc, train=False, size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = {col: enc.num_classes(col) for col in META_COLUMNS}
    model = Stage2MultiHeadClassifier(num_classes=num_classes).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = {head: torch.nn.CrossEntropyLoss() for head in META_COLUMNS}

    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"stage2 epoch {epoch + 1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            out = model(x)
            loss = 0.0
            for head in META_COLUMNS:
                target = y[head].to(device)
                loss = loss + criterion[head](out[head], target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=float(np.mean(losses)))

        train_scores = macro_f1_by_head_on_train(model, train_loader, device=device)
        train_macro = float(np.mean(list(train_scores.values()))) if train_scores else 0.0
        scores = macro_f1_by_head(model, valid_loader, device=device)
        macro = float(np.mean(list(scores.values()))) if scores else 0.0
        print(
            f"epoch={epoch + 1} train_loss={np.mean(losses):.4f} "
            f"train_macro_f1={train_macro:.4f} val_macro_f1={macro:.4f}"
        )
        print("train_head_f1:", train_scores)
        print("val_head_f1:", scores)
        if macro > best_score:
            best_score = macro
            torch.save(model.state_dict(), out_dir / "stage2_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage2 metadata model")
    parser.add_argument("--train-manifest", default="artifacts/manifests/train_manifest.csv")
    parser.add_argument("--valid-manifest", default="artifacts/manifests/valid_manifest.csv")
    parser.add_argument("--output-dir", default="artifacts/models/stage2")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
