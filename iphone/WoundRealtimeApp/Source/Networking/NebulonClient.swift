import Foundation

final class NebulonClient {
    private let session: URLSession
    private let baseURL: URL
    private let model: String
    private let delaySec: Double

    private var pendingWorkItem: DispatchWorkItem?
    private(set) var lastResponseLines: [String] = []

    init(baseURL: String, model: String, delaySec: Double = 2.0) {
        self.baseURL = URL(string: baseURL)!
        self.model = model
        self.delaySec = delaySec
        self.session = URLSession(configuration: .default)
    }

    func schedulePrompt(_ prompt: String, completion: @escaping ([String]) -> Void) {
        pendingWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in
            self?.sendPrompt(prompt, completion: completion)
        }
        pendingWorkItem = item
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delaySec, execute: item)
    }

    private func sendPrompt(_ prompt: String, completion: @escaping ([String]) -> Void) {
        var req = URLRequest(url: baseURL.appendingPathComponent("/api/ollama/generate"))
        req.httpMethod = "POST"
        req.timeoutInterval = 12
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "model": model,
            "prompt": prompt,
            "stream": false
        ]
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        session.dataTask(with: req) { [weak self] data, _, error in
            var lines: [String] = []
            defer {
                self?.lastResponseLines = lines
                completion(lines)
            }
            if let error = error {
                lines = ["Nebulon request failed: \(error.localizedDescription)"]
                return
            }
            guard let data = data,
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                lines = ["Nebulon request failed: invalid JSON response"]
                return
            }
            let text = (obj["response"] as? String) ?? (((obj["message"] as? [String: Any])?["content"]) as? String) ?? ""
            lines = text
                .split(separator: "\n")
                .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .map { OverlayFormatter.sanitizeForOverlay($0) }
                .map { $0.replacingOccurrences(of: "^-\\s*", with: "", options: .regularExpression) }
        }.resume()
    }
}

