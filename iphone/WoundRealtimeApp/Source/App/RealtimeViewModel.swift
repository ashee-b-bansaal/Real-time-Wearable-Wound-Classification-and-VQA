import AVFoundation
import Foundation
import SwiftUI

final class RealtimeViewModel: NSObject, ObservableObject {
    @Published var statusLine: String = "starting..."
    @Published var stage2Lines: [String] = []
    @Published var nebulonLines: [String] = []

    var cameraSession: AVCaptureSession { cameraProvider.session }

    private let config = MobileConfig.loadFromBundle()
    private let cameraProvider = CameraFrameProvider()
    private var modelRunner: ModelRunner?
    private var stabilityGate: TemporalStabilityGate!
    private let dimensionEstimator = DimensionEstimator()
    private let inferenceQueue = DispatchQueue(label: "wound.ios.inference", qos: .userInitiated)
    private var nebulonClient: NebulonClient?
    private var nebulonEnabled: Bool = true

    func start() {
        cameraProvider.delegate = self
        stabilityGate = TemporalStabilityGate(
            requiredStableFrames: config.temporal.stable_frames_required,
            clearAfterNonWoundFrames: config.temporal.clear_metadata_after_non_wound_frames
        )
        do {
            modelRunner = try ModelRunner(config: config)
            statusLine = "models loaded"
        } catch {
            statusLine = "model load failed: \(error.localizedDescription)"
        }
        if config.nebulon.optional_remote {
            nebulonClient = NebulonClient(baseURL: "http://192.168.1.248:3000", model: "llama3:latest", delaySec: config.nebulon.delay_sec)
        }
        cameraProvider.start()
    }

    func stop() {
        cameraProvider.stop()
    }
}

extension RealtimeViewModel: CameraFrameProviderDelegate {
    func cameraProvider(_ provider: CameraFrameProvider, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            guard let runner = self.modelRunner else { return }
            do {
                // Keep quality gate simple on iOS v1; always true.
                let qualityPass = true
                let pre = try runner.run(pixelBuffer: pixelBuffer, runStage2: false)
                let gate = self.stabilityGate.update(isWound: pre.isWound, qualityPass: qualityPass)
                var metadataLines: [String] = self.stage2Lines
                if gate.shouldClearMetadata {
                    metadataLines = []
                }
                if gate.shouldRunStage2 && pre.isWound {
                    let full = try runner.run(pixelBuffer: pixelBuffer, runStage2: true)
                    if let md = full.metadata {
                        let dims = self.dimensionEstimator.estimate(from: pixelBuffer)
                        metadataLines = self.buildMetadataLines(metadata: md, dims: dims)
                        if self.nebulonEnabled, let client = self.nebulonClient {
                            let prompt = self.buildNebulonPrompt(metadata: md, dims: dims)
                            client.schedulePrompt(prompt) { lines in
                                DispatchQueue.main.async {
                                    self.nebulonLines = Array(lines.prefix(5))
                                }
                            }
                        }
                    }
                }
                DispatchQueue.main.async {
                    let label = pre.isWound ? "WOUND" : "NOT WOUND"
                    self.statusLine = "\(label) p=\(String(format: "%.2f", pre.woundProbability)) thr=\(String(format: "%.2f", self.config.stage1.threshold)) stable=\(self.stabilityGate.stableCount)"
                    self.stage2Lines = metadataLines
                }
            } catch {
                DispatchQueue.main.async {
                    self.statusLine = "inference failed: \(error.localizedDescription)"
                }
            }
        }
    }
}

private extension RealtimeViewModel {
    func buildMetadataLines(metadata: [String: HeadPrediction], dims: DimensionResult) -> [String] {
        var lines: [String] = []
        for head in config.stage2.heads_order {
            guard let pred = metadata[head] else { continue }
            lines.append("\(head)=\(pred.label) (\(String(format: "%.2f", pred.confidence)))")
        }
        if dims.status == "ok",
           let area = dims.areaMm2, let major = dims.majorAxisMm, let minor = dims.minorAxisMm {
            lines.append("dim=ok area=\(String(format: "%.1f", area))mm2 major=\(String(format: "%.1f", major))mm minor=\(String(format: "%.1f", minor))mm")
        } else {
            lines.append("dim=\(dims.status)")
        }
        return lines.map { OverlayFormatter.sanitizeForOverlay($0) }
    }

    func buildNebulonPrompt(metadata: [String: HeadPrediction], dims: DimensionResult) -> String {
        var lines = [
            "You are a wound-care triage assistant.",
            "Given structured wound metadata, provide concise next medical steps.",
            "Output 3-5 bullet points, action-oriented, no disclaimer text.",
            "",
            "Metadata:"
        ]
        for head in config.stage2.heads_order {
            if let pred = metadata[head] {
                lines.append("- \(head): \(pred.label)")
            }
        }
        lines.append("Dimensions:")
        lines.append("- status: \(dims.status)")
        lines.append("- area_mm2: \(dims.areaMm2.map { String($0) } ?? "null")")
        lines.append("- major_axis_mm: \(dims.majorAxisMm.map { String($0) } ?? "null")")
        lines.append("- minor_axis_mm: \(dims.minorAxisMm.map { String($0) } ?? "null")")
        return lines.joined(separator: "\n")
    }
}

