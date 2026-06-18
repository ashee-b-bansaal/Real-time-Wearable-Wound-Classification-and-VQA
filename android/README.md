# Android Pipeline (Separate)

This folder is an Android starter surface for running the same two-stage pipeline on Android.

- Keep laptop pipeline unchanged.
- Keep Raspberry/Hailo pipeline unchanged.
- Export ONNX models for Android runtime from existing checkpoints.

## 1) Train models

Stage 1:

```bash
.venv/bin/python scripts/train_stage1.py \
  --model-name resnext50_32x4d \
  --wound-manifest "artifacts/stage1_manifests/train_wound_manifest.csv" \
  --valid-wound-manifest "artifacts/stage1_manifests/valid_wound_manifest.csv" \
  --non-wound-manifest "artifacts/stage1_manifests/non_wound_manifest.csv" \
  --output-dir "artifacts/models/stage1_resnext50_32x4d"
```

Stage 2 (Raspberry model family used for Android export):

```bash
.venv/bin/python raspberry/scripts/train_stage2.py \
  --backbone efficientformer_l1 \
  --output-dir artifacts/models/raspberry/stage2_efficientformer_l1 \
  --export-torchscript
```

## 2) Export ONNX for Android

```bash
.venv/bin/pip install onnx
```

```bash
.venv/bin/python scripts/export_stage1_onnx.py \
  --stage1-model "artifacts/models/stage1_resnext50_32x4d/stage1_best.pt" \
  --stage1-model-name "resnext50_32x4d" \
  --output-path "artifacts/android/Stage1.onnx"
```

```bash
.venv/bin/python scripts/export_stage2_android_onnx.py \
  --stage2-model "artifacts/models/raspberry/stage2_efficientformer_l1/stage2_rpi_best.pt" \
  --stage2-meta "artifacts/models/raspberry/stage2_efficientformer_l1/stage2_rpi_model_meta.json" \
  --label-encoder-json "artifacts/models/raspberry/stage2_efficientformer_l1/label_encoders.json" \
  --output-path "artifacts/android/Stage2.onnx" \
  --runtime-config-out "artifacts/android/android_runtime_config.json"
```

## 3) Android app runtime stack

- Camera: CameraX
- Inference: ONNX Runtime Mobile with NNAPI delegate
- Pipeline:
  - stage1 -> threshold (`0.45`)
  - stable-frame gate
  - stage2 every N frames (default 3)
  - overlay metadata
  - optional Nebulon call with 2s delay

## 4) Suggested Snapdragon 888 preset

- Camera preview: `1280x720@30`
- Input tensor: `224x224`
- Stage1 each frame
- Stage2 every 3 frames (only when Stage1 positive)
- Clear metadata after 5 non-wound frames

