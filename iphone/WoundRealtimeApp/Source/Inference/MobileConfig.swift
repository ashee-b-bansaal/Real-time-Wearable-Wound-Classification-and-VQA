import Foundation

struct MobileConfig: Decodable {
    struct Preprocess: Decodable {
        let input_format: String
        let normalize_mean: [Float]
        let normalize_std: [Float]
    }
    struct Stage1: Decodable {
        let classes: [String]
        let threshold: Float
    }
    struct Stage2: Decodable {
        let heads_order: [String]
        let id_to_label: [String: [String: String]]
    }
    struct Temporal: Decodable {
        let stable_frames_required: Int
        let clear_metadata_after_non_wound_frames: Int
    }
    struct Nebulon: Decodable {
        let optional_remote: Bool
        let delay_sec: Double
    }

    let version: Int
    let image_size: Int
    let preprocess: Preprocess
    let stage1: Stage1
    let stage2: Stage2
    let temporal: Temporal
    let nebulon: Nebulon

    static func loadFromBundle() -> MobileConfig {
        guard let url = Bundle.main.url(forResource: "mobile_config", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let cfg = try? JSONDecoder().decode(MobileConfig.self, from: data) else {
            return MobileConfig(
                version: 1,
                image_size: 224,
                preprocess: .init(
                    input_format: "CHW_RGB_float32",
                    normalize_mean: [0.485, 0.456, 0.406],
                    normalize_std: [0.229, 0.224, 0.225]
                ),
                stage1: .init(classes: ["not_wound", "wound"], threshold: 0.45),
                stage2: .init(heads_order: [
                    "anatomic_locations",
                    "wound_type",
                    "wound_thickness",
                    "tissue_color",
                    "drainage_amount",
                    "drainage_type",
                    "infection"
                ], id_to_label: [:]),
                temporal: .init(stable_frames_required: 6, clear_metadata_after_non_wound_frames: 5),
                nebulon: .init(optional_remote: true, delay_sec: 2.0)
            )
        }
        return cfg
    }
}

