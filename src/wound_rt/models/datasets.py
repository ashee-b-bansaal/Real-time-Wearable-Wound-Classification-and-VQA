from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


META_COLUMNS = [
    "anatomic_locations",
    "wound_type",
    "wound_thickness",
    "tissue_color",
    "drainage_amount",
    "drainage_type",
    "infection",
]


@dataclass(frozen=True)
class LabelEncoders:
    mapping: dict[str, dict[str, int]]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "LabelEncoders":
        mapping: dict[str, dict[str, int]] = {}
        for c in META_COLUMNS:
            vals = sorted({str(v) for v in df[c].fillna("")})
            mapping[c] = {v: i for i, v in enumerate(vals)}
        return cls(mapping=mapping)

    def encode_row(self, row: pd.Series) -> dict[str, int]:
        out: dict[str, int] = {}
        for col in META_COLUMNS:
            key = str(row[col])
            out[col] = self.mapping[col].get(key, 0)
        return out

    def num_classes(self, col: str) -> int:
        return len(self.mapping[col])


def image_transform(train: bool, size: int) -> A.Compose:
    if train:
        return A.Compose(
            [
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.35),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.3),
                A.MotionBlur(blur_limit=(3, 7), p=0.35),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    return A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()])


class Stage1BinaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, size: int = 224) -> None:
        self.df = df.reset_index(drop=True)
        self.tf = image_transform(train=train, size=size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["image_path"]))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tf(image=img)["image"]
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y


class Stage2MultiHeadDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        encoders: LabelEncoders,
        train: bool,
        size: int = 224,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.encoders = encoders
        self.tf = image_transform(train=train, size=size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["image_path"]))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tf(image=img)["image"]
        enc = self.encoders.encode_row(row)
        y = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        return x, y


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
