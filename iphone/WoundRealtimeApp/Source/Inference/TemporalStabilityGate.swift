import Foundation

final class TemporalStabilityGate {
    private let requiredStableFrames: Int
    private let clearAfterNonWoundFrames: Int
    private(set) var stableCount: Int = 0
    private(set) var nonWoundStreak: Int = 0

    init(requiredStableFrames: Int, clearAfterNonWoundFrames: Int) {
        self.requiredStableFrames = requiredStableFrames
        self.clearAfterNonWoundFrames = clearAfterNonWoundFrames
    }

    func update(isWound: Bool, qualityPass: Bool) -> (shouldRunStage2: Bool, shouldClearMetadata: Bool) {
        if isWound && qualityPass {
            stableCount += 1
            nonWoundStreak = 0
        } else {
            stableCount = 0
            if !isWound {
                nonWoundStreak += 1
            }
        }
        let shouldRun = stableCount >= requiredStableFrames
        if shouldRun { stableCount = 0 }
        return (shouldRun, nonWoundStreak > clearAfterNonWoundFrames)
    }
}

