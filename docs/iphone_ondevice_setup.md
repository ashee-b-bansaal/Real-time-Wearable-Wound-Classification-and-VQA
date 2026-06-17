# iPhone On-Device Setup

This runbook keeps laptop and phone workflows separate:

- Laptop workflow: `laptop/`
- iPhone workflow: `iphone/WoundRealtimeApp/`

## 1) Prerequisites

- macOS + latest Xcode
- iPhone connected by cable
- iPhone Developer Mode enabled
- Apple signing team selected in Xcode target settings

## 2) Export CoreML models from trained PyTorch checkpoints

From repo root:

```bash
source .venv/bin/activate
pip install coremltools
python scripts/export_stage1_coreml.py \
  --stage1-model artifacts/models/stage1/stage1_best.pt \
  --output-path artifacts/mobile/Stage1.mlpackage

python scripts/export_stage2_coreml.py \
  --stage2-model artifacts/models/stage2_ds2/stage2_best.pt \
  --label-encoder-json artifacts/models/stage2_ds2/label_encoders.json \
  --output-path artifacts/mobile/Stage2.mlpackage \
  --mobile-config-path artifacts/mobile/mobile_config.json \
  --stage1-threshold 0.45 \
  --nebulon-delay-sec 2.0
```

## 3) Validate parity (desktop PyTorch vs CoreML)

```bash
python scripts/validate_coreml_parity.py \
  --manifest artifacts/manifests/valid_manifest.csv \
  --stage1-model artifacts/models/stage1/stage1_best.pt \
  --stage1-coreml artifacts/mobile/Stage1.mlpackage \
  --stage2-model artifacts/models/stage2_ds2/stage2_best.pt \
  --stage2-coreml artifacts/mobile/Stage2.mlpackage \
  --label-encoder-json artifacts/models/stage2_ds2/label_encoders.json \
  --output-path artifacts/mobile/parity_report.json
```

## 4) Wire models into iOS app target

1. Create/open an iOS SwiftUI app target in Xcode.
2. Add these source folders to the target:
   - `iphone/WoundRealtimeApp/Source/App/`
   - `iphone/WoundRealtimeApp/Source/Camera/`
   - `iphone/WoundRealtimeApp/Source/Inference/`
   - `iphone/WoundRealtimeApp/Source/Dimensions/`
   - `iphone/WoundRealtimeApp/Source/Networking/`
   - `iphone/WoundRealtimeApp/Source/Overlay/`
3. Add model/config artifacts to app bundle resources:
   - `artifacts/mobile/Stage1.mlmodelc` (compile from `.mlpackage` in Xcode)
   - `artifacts/mobile/Stage2.mlmodelc`
   - `artifacts/mobile/mobile_config.json`

## 5) Optional Nebulon remote path

- Nebulon is optional and delayed by 2 seconds before dispatch.
- Default endpoint in source is `http://192.168.1.248:3000`.
- Ensure iPhone and server are on reachable network.
- Keep `NSLocalNetworkUsageDescription` in `Info.plist` for local-network calls.

## 6) Acceptance checks

- Live status shows `WOUND` / `NOT WOUND` and probability.
- Stage2 metadata appears after stability gate.
- Metadata panel clears after consecutive non-wound streak.
- Nebulon text appears without Unicode `???` artifacts.
- `artifacts/mobile/parity_report.json` indicates strong parity before device release.

