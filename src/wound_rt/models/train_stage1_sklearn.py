from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from wound_rt.models.train_stage1 import read_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage1 sklearn classifier on EfficientNet-B0 features")
    parser.add_argument("--wound-manifest", default="artifacts/manifests/train_manifest.csv")
    parser.add_argument("--valid-wound-manifest", default="artifacts/manifests/valid_manifest.csv")
    parser.add_argument("--non-wound-manifest", default="artifacts/negatives/non_wound_manifest.csv")
    parser.add_argument("--classifier", choices=["logreg", "adaboost"], default="logreg")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="artifacts/models/stage1_sklearn")
    return parser.parse_args()


def build_feature_extractor(device: torch.device) -> torch.nn.Module:
    model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model.to(device)


def image_to_tensor(image_path: str, image_size: int) -> torch.Tensor:
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tf(img).unsqueeze(0)


def extract_features(df: pd.DataFrame, extractor: torch.nn.Module, device: torch.device, image_size: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="extract_features"):
            x = image_to_tensor(row["image_path"], image_size=image_size).to(device)
            f = extractor(x).cpu().numpy()[0]
            feats.append(f)
    return np.array(feats)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df = read_data(
        Path(args.wound_manifest),
        Path(args.valid_wound_manifest),
        Path(args.non_wound_manifest),
    )
    x_model = build_feature_extractor(device=device)
    x_train = extract_features(train_df, x_model, device=device, image_size=args.image_size)
    y_train = train_df["label"].astype(int).to_numpy()
    x_val = extract_features(val_df, x_model, device=device, image_size=args.image_size)
    y_val = val_df["label"].astype(int).to_numpy()

    if args.classifier == "logreg":
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    else:
        clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=200, random_state=42)

    clf.fit(x_train, y_train)
    pred = clf.predict(x_val)
    metrics = {
        "recall": float(recall_score(y_val, pred, zero_division=0)),
        "precision": float(precision_score(y_val, pred, zero_division=0)),
        "f1": float(f1_score(y_val, pred, zero_division=0)),
    }
    joblib.dump(clf, out_dir / f"stage1_{args.classifier}.joblib")
    with (out_dir / f"stage1_{args.classifier}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "stage1_sklearn_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"feature_backbone": "efficientnet_b0", "classifier": args.classifier}, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
