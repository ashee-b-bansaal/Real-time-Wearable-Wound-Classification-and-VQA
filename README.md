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

Press `q` to stop preview windows in capture/realtime modes.

## Notes

- If image archives are not extracted yet, manifest generation still runs and records `image_exists=false`.
- Stage1 prioritizes high recall by default; tune `--stage1-threshold` for your false-negative tolerance.
- Dimensions are returned only when the reference marker is confidently detected.
