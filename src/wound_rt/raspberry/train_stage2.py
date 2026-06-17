from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from wound_rt.models.datasets import LabelEncoders, META_COLUMNS, Stage2MultiHeadDataset
from wound_rt.raspberry.networks import (
    RPI_STAGE2_BACKBONES,
    RaspberryStage2MultiHeadClassifier,
    RaspberryStage2TupleWrapper,
)


def _set_backbone_trainable(model: RaspberryStage2MultiHeadClassifier, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def _macro_f1_by_head(
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
    return {
        head: (
            f1_score(ys[head], ps[head], average="macro", zero_division=0) if ys[head] else 0.0
        )
        for head in META_COLUMNS
    }


def _export_torchscript_bundle(
    model: RaspberryStage2MultiHeadClassifier,
    output_dir: Path,
    image_size: int,
) -> None:
    model_cpu = model.to(torch.device("cpu")).eval()
    wrapped = RaspberryStage2TupleWrapper(model_cpu).eval()
    example = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    scripted = torch.jit.trace(wrapped, example)
    torch.jit.save(scripted, output_dir / "stage2_rpi_scripted.pt")


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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = {col: enc.num_classes(col) for col in META_COLUMNS}
    model = RaspberryStage2MultiHeadClassifier(
        num_classes=num_classes,
        backbone_name=args.backbone,
        head_dropout=args.head_dropout,
        pretrained_backbone=not args.no_pretrained_backbone,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, min_lr=1e-6)

    criterion: dict[str, torch.nn.Module] = {}
    for head in META_COLUMNS:
        labels = [enc.encode_row(row)[head] for _, row in train_df.iterrows()]
        counts = np.bincount(labels, minlength=num_classes[head]).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        raw_w = len(labels) / (num_classes[head] * counts)
        weights = np.power(raw_w, args.class_weight_power)
        weights = weights / np.mean(weights)
        w_t = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion[head] = torch.nn.CrossEntropyLoss(weight=w_t, label_smoothing=args.label_smoothing)

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve_epochs = 0
    for epoch in range(args.epochs):
        backbone_trainable = epoch >= args.freeze_backbone_epochs
        _set_backbone_trainable(model, trainable=backbone_trainable)
        model.train()
        losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"rpi-stage2 epoch {epoch + 1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            out = model(x)
            loss = 0.0
            for head in META_COLUMNS:
                target = y[head].to(device, non_blocking=True)
                loss = loss + criterion[head](out[head], target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=float(np.mean(losses)))

        train_scores = _macro_f1_by_head(model, train_loader, device=device)
        train_macro = float(np.mean(list(train_scores.values()))) if train_scores else 0.0
        scores = _macro_f1_by_head(model, valid_loader, device=device)
        macro = float(np.mean(list(scores.values()))) if scores else 0.0
        print(
            f"epoch={epoch + 1} backbone={args.backbone} train_loss={np.mean(losses):.4f} "
            f"train_macro_f1={train_macro:.4f} val_macro_f1={macro:.4f} "
            f"backbone_trainable={backbone_trainable}"
        )
        print("train_head_f1:", train_scores)
        print("val_head_f1:", scores)
        scheduler.step(macro)
        if macro > best_score:
            best_score = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        elif backbone_trainable:
            no_improve_epochs += 1
            if no_improve_epochs >= args.early_stopping_patience:
                print(
                    f"Early stopping: no validation macro-F1 improvement for {no_improve_epochs} epochs. "
                    f"Best val_macro_f1={best_score:.4f}"
                )
                break

    if best_state is None:
        raise RuntimeError("No model state captured during training.")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "stage2_rpi_best.pt")
    with (out_dir / "stage2_rpi_model_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone": args.backbone,
                "image_size": args.image_size,
                "heads": META_COLUMNS,
                "num_classes": num_classes,
            },
            f,
            indent=2,
        )
    if args.export_torchscript:
        _export_torchscript_bundle(model, output_dir=out_dir, image_size=args.image_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Raspberry/Hailo-ready Stage2 metadata model (separate from laptop pipeline)"
    )
    parser.add_argument("--train-manifest", default="artifacts/manifests/train_manifest.csv")
    parser.add_argument("--valid-manifest", default="artifacts/manifests/valid_manifest.csv")
    parser.add_argument("--output-dir", default="artifacts/models/raspberry/stage2")
    parser.add_argument("--backbone", default="efficientformer_l1", choices=RPI_STAGE2_BACKBONES)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--head-dropout", type=float, default=0.25)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Disable pretrained ImageNet initialization for backbone.",
    )
    parser.add_argument(
        "--export-torchscript",
        action="store_true",
        help="Also export traced TorchScript .pt bundle for external Hailo conversion flows.",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

