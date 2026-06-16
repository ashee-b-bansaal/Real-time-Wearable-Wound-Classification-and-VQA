# Realtime Wound Classification + Metadata

This repository implements a two-stage real-time pipeline for a USB-C wearable glasses camera:

1. **Stage 1**: wound vs non-wound
2. **Stage 2**: non-VLM wound metadata prediction
3. **Dimensions**: marker-based wound size estimation in mm/mm^2

## Project structure

- `src/wound_rt/data/manifest_builder.py`: build train/valid/test manifests from OSF JSON labels
- `src/wound_rt/data/negative_set.py`: capture non-wound negatives from live camera and merge optional public negatives
- `src/wound_rt/models/train_stage1.py`: train binary wound/non-wound classifier
- `src/wound_rt/models/train_stage2.py`: train multi-head metadata classifier
- `src/wound_rt/realtime/pipeline.py`: asynchronous quality-gated real-time inference loop
- `src/wound_rt/realtime/dimensions.py`: reference marker scale + wound dimension estimation
- `src/wound_rt/eval/prepare_eval.py`: build stage1 eval manifest
- `src/wound_rt/eval/benchmark.py`: benchmark latency and task metrics

## Metadata heads (stage2)

- `anatomic_locations`
- `wound_type`
- `wound_thickness`
- `tissue_color`
- `drainage_amount`
- `drainage_type`
- `infection`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## End-to-end commands

1) Build manifests from dataset JSON:

```bash
python scripts/build_manifests.py
```

2) Capture non-wound negatives from glasses camera:

```bash
python scripts/create_negative_set.py --camera-index 0 --max-images 600
```

Optional public negatives:

```bash
python scripts/create_negative_set.py --public-negatives-dir "/path/to/non_wound_images"
```

3) Train stage1:

```bash
python scripts/train_stage1.py
```

4) Train stage2:

```bash
python scripts/train_stage2.py
```

5) Prepare eval and benchmark:

```bash
python scripts/prepare_eval.py
python scripts/benchmark_models.py
```

6) Run realtime:

```bash
python scripts/run_realtime.py --camera-index 0
```

Realtime mode options:

```bash
# Stage 1 only (wound vs non-wound)
python scripts/run_realtime.py --run-mode stage1 --camera-index 0 --stage1-model "artifacts/models/stage1_resnet50/stage1_best.pt" --stage1-model-name resnet50

# Stage 2 only (metadata only)
python scripts/run_realtime.py --run-mode stage2 --camera-index 0 --stage2-model "artifacts/models/stage2_ds2/stage2_best.pt" --label-encoder-json "artifacts/models/stage2_ds2/label_encoders.json"

# Both stages together
python scripts/run_realtime.py --run-mode both --camera-index 0 --stage1-model "artifacts/models/stage1_resnet50/stage1_best.pt" --stage1-model-name resnet50 --stage2-model "artifacts/models/stage2_ds2/stage2_best.pt" --label-encoder-json "artifacts/models/stage2_ds2/label_encoders.json"
```

Press `q` to stop preview windows in capture/realtime modes.

## Current benchmark snapshot

Latest run summary: Stage 1 F1 (non-wound) = `0.9938`, Stage 1 F1 (wound) = `0.9963`, and Stage 2 average macro-F1 = `0.2765`.

Overfitting note: one likely reason is that pre-existing dataset annotations were created for a broader multimodal purpose (text plus image context), so some labels are not perfectly aligned to this strictly image-only real-time wound metadata use case.

## Notes

- If image archives are not extracted yet, manifest generation still runs and records `image_exists=false`.
- Stage1 prioritizes high recall by default; tune `--stage1-threshold` for your false-negative tolerance.
- Dimensions are returned only when the reference marker is confidently detected.
