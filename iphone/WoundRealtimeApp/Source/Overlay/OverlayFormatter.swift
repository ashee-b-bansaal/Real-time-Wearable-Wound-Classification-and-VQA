import Foundation

enum OverlayFormatter {
    static func sanitizeForOverlay(_ text: String) -> String {
        let replacements: [String: String] = [
            "•": "- ",
            "●": "- ",
            "·": "- ",
            "–": "-",
            "—": "-"
        ]
        var out = text
        for (k, v) in replacements {
            out = out.replacingOccurrences(of: k, with: v)
        }
        out = out.applyingTransform(.toLatin, reverse: false) ?? out
        out = out.applyingTransform(.stripCombiningMarks, reverse: false) ?? out
        out = out.unicodeScalars.filter { $0.isASCII }.map(String.init).joined()
        return out.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

