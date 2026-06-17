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

## 4) Open provided Xcode project and configure signing

1. Open this project directly in Xcode:
   - `iphone/WoundRealtimeApp/WoundRealtimeApp.xcodeproj`
2. Select target `WoundRealtimeApp` -> **Signing & Capabilities**.
3. Set your Apple Team and keep **Automatically manage signing** enabled.
4. Keep bundle id unique if needed (for example append your initials).

## 5) Add CoreML models into the app bundle

1. In Finder, open `artifacts/mobile/`.
2. Drag `Stage1.mlpackage` and `Stage2.mlpackage` into the Xcode project navigator.
3. Ensure both are added to target `WoundRealtimeApp`.
4. Keep `iphone/WoundRealtimeApp/Resources/mobile_config.json` as the config source in the bundle.
5. Build once so Xcode compiles `.mlpackage` into `.mlmodelc` automatically.

## 6) Optional Nebulon remote path

- Nebulon is optional and delayed by 2 seconds before dispatch.
- Default endpoint in source is `http://192.168.1.248:3000`.
- Ensure iPhone and server are on reachable network.
- Keep `NSLocalNetworkUsageDescription` in `Info.plist` for local-network calls.

## 7) Run on iPhone 17 Pro Max

1. Connect iPhone by cable.
2. Select iPhone device target in Xcode toolbar.
3. Press **Run**.
4. On first run, allow camera and local-network permissions.

## 8) Acceptance checks

- Live status shows `WOUND` / `NOT WOUND` and probability.
- Stage2 metadata appears after stability gate.
- Metadata panel clears after consecutive non-wound streak.
- Nebulon text appears without Unicode `???` artifacts.
- `artifacts/mobile/parity_report.json` indicates strong parity before device release.

