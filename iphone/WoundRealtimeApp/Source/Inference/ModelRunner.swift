import CoreML
import CoreVideo
import Foundation

struct HeadPrediction {
    let label: String
    let confidence: Float
}

struct InferenceOutput {
    let woundProbability: Float
    let isWound: Bool
    let metadata: [String: HeadPrediction]?
}

final class ModelRunner {
    private let config: MobileConfig
    private let stage1Model: MLModel
    private let stage2Model: MLModel

    init(config: MobileConfig) throws {
        self.config = config
        let mlCfg = MLModelConfiguration()
        mlCfg.computeUnits = .all
        guard let stage1URL = Bundle.main.url(forResource: "Stage1", withExtension: "mlmodelc"),
              let stage2URL = Bundle.main.url(forResource: "Stage2", withExtension: "mlmodelc") else {
            throw NSError(domain: "ModelRunner", code: -1, userInfo: [NSLocalizedDescriptionKey: "CoreML models not found in app bundle"])
        }
        self.stage1Model = try MLModel(contentsOf: stage1URL, configuration: mlCfg)
        self.stage2Model = try MLModel(contentsOf: stage2URL, configuration: mlCfg)
    }

    func run(pixelBuffer: CVPixelBuffer, runStage2: Bool) throws -> InferenceOutput {
        let inputArray = try makeInputArray(from: pixelBuffer, size: config.image_size, mean: config.preprocess.normalize_mean, std: config.preprocess.normalize_std)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_image": inputArray])

        let stage1Out = try stage1Model.prediction(from: provider)
        let stage1Logit = (stage1Out.featureValue(for: "stage1_logit")?.multiArrayValue?[0].floatValue) ?? 0
        let prob = 1.0 / (1.0 + exp(-stage1Logit))
        let isWound = prob >= config.stage1.threshold
        guard runStage2 else {
            return InferenceOutput(woundProbability: prob, isWound: isWound, metadata: nil)
        }

        let stage2Out = try stage2Model.prediction(from: provider)
        var metadata: [String: HeadPrediction] = [:]
        for head in config.stage2.heads_order {
            let key = "\(head)_logits"
            guard let logits = stage2Out.featureValue(for: key)?.multiArrayValue else { continue }
            let probs = softmax(logits)
            let bestIdx = probs.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            let label = config.stage2.id_to_label[head]?[String(bestIdx)] ?? "unknown"
            metadata[head] = HeadPrediction(label: label, confidence: probs[bestIdx])
        }
        return InferenceOutput(woundProbability: prob, isWound: isWound, metadata: metadata)
    }
}

private func softmax(_ array: MLMultiArray) -> [Float] {
    let vals = (0..<array.count).map { array[$0].floatValue }
    let maxV = vals.max() ?? 0
    let exps = vals.map { exp($0 - maxV) }
    let s = exps.reduce(0, +)
    if s <= 0 { return Array(repeating: 0, count: vals.count) }
    return exps.map { $0 / s }
}

private func makeInputArray(from pixelBuffer: CVPixelBuffer, size: Int, mean: [Float], std: [Float]) throws -> MLMultiArray {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
        throw NSError(domain: "ModelRunner", code: -2, userInfo: [NSLocalizedDescriptionKey: "No pixel buffer base address"])
    }
    let srcWidth = CVPixelBufferGetWidth(pixelBuffer)
    let srcHeight = CVPixelBufferGetHeight(pixelBuffer)
    let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

    // Nearest-neighbor resize into CHW normalized float tensor.
    let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
    let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))

    for y in 0..<size {
        for x in 0..<size {
            let sx = min(srcWidth - 1, Int(Float(x) * Float(srcWidth) / Float(size)))
            let sy = min(srcHeight - 1, Int(Float(y) * Float(srcHeight) / Float(size)))
            let pixel = base.advanced(by: sy * srcBytesPerRow + sx * 4).assumingMemoryBound(to: UInt8.self)
            // BGRA layout from camera
            let b = Float(pixel[0]) / 255.0
            let g = Float(pixel[1]) / 255.0
            let r = Float(pixel[2]) / 255.0
            let idxR = y * size + x
            let idxG = size * size + idxR
            let idxB = 2 * size * size + idxR
            ptr[idxR] = (r - mean[0]) / std[0]
            ptr[idxG] = (g - mean[1]) / std[1]
            ptr[idxB] = (b - mean[2]) / std[2]
        }
    }
    return arr
}

