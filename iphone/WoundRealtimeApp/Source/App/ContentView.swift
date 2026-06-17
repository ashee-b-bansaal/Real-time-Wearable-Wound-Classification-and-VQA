import AVFoundation
import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: RealtimeViewModel

    var body: some View {
        ZStack(alignment: .topLeading) {
            CameraPreviewView(session: viewModel.cameraSession)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 8) {
                Text(viewModel.statusLine)
                    .font(.system(size: 15, weight: .semibold, design: .monospaced))
                    .padding(10)
                    .background(Color.black.opacity(0.7))
                    .foregroundColor(.green)
                    .cornerRadius(8)

                if !viewModel.stage2Lines.isEmpty {
                    OverlayPanel(title: "Stage2 metadata", lines: viewModel.stage2Lines, color: .yellow)
                }

                Spacer()

                if !viewModel.nebulonLines.isEmpty {
                    OverlayPanel(title: "Next medical steps (Nebulon)", lines: viewModel.nebulonLines, color: .white)
                }
            }
            .padding(12)
        }
    }
}

struct OverlayPanel: View {
    let title: String
    let lines: [String]
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).bold()
            ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                Text(line).lineLimit(2)
            }
        }
        .font(.system(size: 13, weight: .regular, design: .monospaced))
        .foregroundColor(color)
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.78))
        .cornerRadius(8)
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        uiView.videoPreviewLayer.session = session
    }
}

final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
}

