import AVFoundation
import Foundation

protocol CameraFrameProviderDelegate: AnyObject {
    func cameraProvider(_ provider: CameraFrameProvider, didOutput pixelBuffer: CVPixelBuffer, timestamp: CMTime)
}

final class CameraFrameProvider: NSObject {
    let session = AVCaptureSession()
    weak var delegate: CameraFrameProviderDelegate?

    private let outputQueue = DispatchQueue(label: "wound.camera.output", qos: .userInitiated)
    private let videoOutput = AVCaptureVideoDataOutput()

    func start() {
        configureSessionIfNeeded()
        if !session.isRunning {
            session.startRunning()
        }
    }

    func stop() {
        if session.isRunning {
            session.stopRunning()
        }
    }

    private func configureSessionIfNeeded() {
        guard session.inputs.isEmpty else { return }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        defer { session.commitConfiguration() }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }
        do {
            try device.lockForConfiguration()
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            device.unlockForConfiguration()
        } catch {
            return
        }

        guard let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            return
        }
        session.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: outputQueue)
        guard session.canAddOutput(videoOutput) else { return }
        session.addOutput(videoOutput)
    }
}

extension CameraFrameProvider: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        delegate?.cameraProvider(self, didOutput: pb, timestamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
    }
}

