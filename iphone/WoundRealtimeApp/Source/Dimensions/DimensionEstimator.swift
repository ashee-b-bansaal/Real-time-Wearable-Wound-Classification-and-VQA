import CoreVideo
import Foundation

struct DimensionResult {
    let status: String
    let areaMm2: Float?
    let majorAxisMm: Float?
    let minorAxisMm: Float?
    let pxPerMm: Float?
}

final class DimensionEstimator {
    // Matches Python defaults in src/wound_rt/realtime/dimensions.py
    private let markerWidthMm: Float = 20.0
    private let markerHeightMm: Float = 20.0
    private let markerMinAreaPx: Int = 400

    func estimate(from pixelBuffer: CVPixelBuffer) -> DimensionResult {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return DimensionResult(status: "unavailable_marker_missing", areaMm2: nil, majorAxisMm: nil, minorAxisMm: nil, pxPerMm: nil)
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)

        var markerMinX = width, markerMinY = height, markerMaxX = -1, markerMaxY = -1
        var markerCount = 0
        var woundMinX = width, woundMinY = height, woundMaxX = -1, woundMaxY = -1
        var woundCount = 0

        for y in 0..<height {
            for x in 0..<width {
                let px = base.advanced(by: y * bpr + x * 4).assumingMemoryBound(to: UInt8.self)
                let b = Float(px[0]) / 255.0
                let g = Float(px[1]) / 255.0
                let r = Float(px[2]) / 255.0
                let (h, s, v) = rgbToHsv(r: r, g: g, b: b)
                if h >= 95 && h <= 135 && s >= 80 && v >= 50 {
                    markerCount += 1
                    markerMinX = min(markerMinX, x)
                    markerMaxX = max(markerMaxX, x)
                    markerMinY = min(markerMinY, y)
                    markerMaxY = max(markerMaxY, y)
                }
                let labA = approxLabA(r: r, g: g, b: b)
                if labA > 145 {
                    woundCount += 1
                    woundMinX = min(woundMinX, x)
                    woundMaxX = max(woundMaxX, x)
                    woundMinY = min(woundMinY, y)
                    woundMaxY = max(woundMaxY, y)
                }
            }
        }

        if markerCount < markerMinAreaPx || markerMaxX < markerMinX || markerMaxY < markerMinY {
            return DimensionResult(status: "unavailable_marker_missing", areaMm2: nil, majorAxisMm: nil, minorAxisMm: nil, pxPerMm: nil)
        }
        let markerW = Float(markerMaxX - markerMinX + 1)
        let markerH = Float(markerMaxY - markerMinY + 1)
        let markerPx = (markerW + markerH) / 2.0
        let markerMm = (markerWidthMm + markerHeightMm) / 2.0
        let pxPerMm = markerPx / markerMm

        if woundCount < 50 || woundMaxX < woundMinX || woundMaxY < woundMinY {
            return DimensionResult(status: "unavailable_small_contour", areaMm2: nil, majorAxisMm: nil, minorAxisMm: nil, pxPerMm: pxPerMm)
        }
        let woundW = Float(woundMaxX - woundMinX + 1)
        let woundH = Float(woundMaxY - woundMinY + 1)
        let areaPx = Float(woundCount)
        return DimensionResult(
            status: "ok",
            areaMm2: areaPx / (pxPerMm * pxPerMm),
            majorAxisMm: max(woundW, woundH) / pxPerMm,
            minorAxisMm: min(woundW, woundH) / pxPerMm,
            pxPerMm: pxPerMm
        )
    }
}

private func rgbToHsv(r: Float, g: Float, b: Float) -> (Float, Float, Float) {
    let maxV = max(r, max(g, b))
    let minV = min(r, min(g, b))
    let delta = maxV - minV
    var h: Float = 0
    if delta > 0 {
        if maxV == r {
            h = 60 * fmod(((g - b) / delta), 6)
        } else if maxV == g {
            h = 60 * (((b - r) / delta) + 2)
        } else {
            h = 60 * (((r - g) / delta) + 4)
        }
    }
    if h < 0 { h += 360 }
    let s = maxV == 0 ? 0 : (delta / maxV)
    // OpenCV ranges: H in [0,179], S and V in [0,255]
    return (h / 2.0, s * 255.0, maxV * 255.0)
}

private func approxLabA(r: Float, g: Float, b: Float) -> Float {
    // Fast approximation for a-channel style redness score (0..255).
    let redness = max(0, r - g) * 255.0
    return min(255.0, 128.0 + redness * 0.6)
}

