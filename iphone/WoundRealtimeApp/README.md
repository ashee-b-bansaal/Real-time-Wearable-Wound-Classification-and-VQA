# iPhone On-Device Implementation

This folder contains the iPhone-specific runtime implementation for:

- Stage 1 wound vs non-wound (CoreML)
- Stage 2 multi-head metadata (CoreML)
- Marker-based wound dimensions on device
- Optional Nebulon remote recommendations with 2s delayed dispatch

## Folder layout

- `Source/App/`: app entry and root UI
- `Source/Camera/`: camera capture pipeline
- `Source/Inference/`: CoreML inference + temporal logic
- `Source/Dimensions/`: marker detection and dimension estimation
- `Source/Networking/`: optional Nebulon client
- `Source/Overlay/`: text sanitization and overlay formatting

## Open in Xcode

- Project path: `iphone/WoundRealtimeApp/WoundRealtimeApp.xcodeproj`
- Target: `WoundRealtimeApp`
- Deployment target: iOS 17.0

## Before first run

1. Export CoreML models into `artifacts/mobile/` (see `docs/iphone_ondevice_setup.md`).
2. Drag `Stage1.mlpackage` and `Stage2.mlpackage` into the Xcode project.
3. In target signing settings, select your Apple team.
4. Build and run on your device.

