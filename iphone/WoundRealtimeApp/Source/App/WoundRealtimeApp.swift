import SwiftUI

@main
struct WoundRealtimeApp: App {
    @StateObject private var viewModel = RealtimeViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .onAppear { viewModel.start() }
                .onDisappear { viewModel.stop() }
        }
    }
}

