# Raspberry + Hailo Implementation (Separate)

This folder is intentionally separate from the laptop implementation.

- Laptop path remains unchanged (`scripts/`, `src/wound_rt/realtime/`, `laptop/`).
- Raspberry/Hailo Stage 2 training lives in `raspberry/scripts/`.

## Supported Raspberry Stage2 backbones

- `resnext50_32x4d`
- `efficientformer_l1`
- `efficientnet_l`
- `efficientnet_lite0`
- `efficientnet_lite1`
- `efficientnet_lite2`
- `efficientnet_lite3`
- `efficientnet_lite4`
- `efficientnet_m`
- `efficientnet_s`

## Install note

Raspberry Stage2 backbones (except `resnext50_32x4d`) require `timm`:

```bash
pip install timm
```

## Train Raspberry Stage2 (example)

```bash
.venv/bin/python raspberry/scripts/train_stage2.py \
  --backbone resnext50_32x4d \
  --output-dir artifacts/models/raspberry/stage2_resnext \
  --export-torchscript
```

```bash
.venv/bin/python raspberry/scripts/train_stage2.py \
  --backbone efficientformer_l1 \
  --output-dir artifacts/models/raspberry/stage2_efficientformer_l1 \
  --export-torchscript
```

Artifacts for your external Hailo conversion repo:

- `stage2_rpi_best.pt` (state_dict)
- `stage2_rpi_scripted.pt` (TorchScript, when `--export-torchscript` is enabled)
- `stage2_rpi_model_meta.json`
- `label_encoders.json`

