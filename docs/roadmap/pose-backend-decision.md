---
id: pose-backend-decision
status: shipped
depends_on: []
blocks: [multi-person-tracking]
---

## Goal

Choose the single-person pose model that production uses, and record the decision and the evidence behind it in the repo.

## Decision

**Stay on arm A, legacy `mediapipe.solutions.pose`.** This is what `ShotPipeline` already runs (`engine/tracking/pose/mediapipe_estimator.py`, `mediapipe==0.10.9`). No change to the frozen `engine/tracking/` package is needed. Recorded 2026-10-02; the comparison itself was run earlier, and its outputs are cached in `assets/outputs/validation/pose_ab/` (gitignored and local-only).

## Arms tested

All arms were compared on raw per-frame detector output, with no visibility filter, persistence or smoothing, using `tools/pose_ab.py` and the same `engine.preprocessing` frame ingestion. Two clips were used: backhand (`sample_backhand2.mp4`, 293 frames; problem limb is the right arm) and forehand (the "Serious Squash" clip, 941 frames; problem limb is `left_ankle`).

| Arm | Model / mode | Backhand: right_elbow / right_wrist, % frames with visibility ≥ 0.5 | Forehand: left_ankle, % frames with visibility ≥ 0.5 |
|---|---|---|---|
| A | legacy `solutions.pose` (production) | 53.24 / 49.83 | 85.55 |
| B | Tasks `pose_landmarker_heavy`, VIDEO mode | 16.38 / 51.19 | 89.69 |
| C | heavy model, IMAGE mode (no temporal tracking) | 35.49 / 55.97 | 84.17 |
| D | heavy model, VIDEO mode, person-ROI crop upscaled to 256×256 | 22.18 / 50.85 | 82.78 |
| E | heavy model, VIDEO mode, forced tracker reset after >5 consecutive low-confidence frames | 42.66 / 70.99 | 92.24 |

A, B, C and D are from `report.json`. E is computed from `backhand_E.json` and `forehand_E.json` in the same cache, with the same ≥ 0.5 rule. `right_shoulder` is 100% for every arm.

## Why not B, C or D

None of them beats A on the backhand's racket arm, which is the limb the swing and phase detection depends on. B and D lose badly on `right_elbow` (16% and 22% against A's 53%), and C trades elbow coverage for slightly more wrist coverage. B's forehand `left_ankle` gain (+4 points) is not worth losing two-thirds of the backhand's elbow coverage. Switching would also mean changing the frozen tracking package for no net gain.

## Why E was disqualified

E is the only arm that recovers coverage on the hard limbs: `right_wrist` goes from 50% to 71% on the backhand, and `left_ankle` from 86% to 92% on the forehand. It was disqualified because of how that coverage is produced. Each forced reset makes the tracker re-acquire the limb from scratch. Those reappearances arrive **at high reported confidence** but sit at **positions with no continuity guarantee**:

- The production chain cannot see them as discontinuities. Running E's output through `ThresholdVisibilityFilter` → `MissedFramePersistence` → `MovingAverageSmoother` → `find_landmark_discontinuities`, 8 of 26 `right_elbow` reset events and 25 of 27 `right_wrist` reset events reappeared at visibility 0.51–0.98, **all unflagged** (`docs/bugs/discontinuity-detector-gap.md`). Recomputed on 2026-10-02 from the cache: the first post-reset `right_wrist` frame reports visibility 0.84–0.98.
- The original investigation measured position errors of 16–40px on those confident reappearances. **Not re-verified on 2026-10-02:** the reference method used for that figure isn't recorded in the repo. A re-check against arm A's position at the same frame (arm A isn't ground truth either) gave 1–20px for most first post-reset frames, plus two outliers of 71px and 112px, both at frame 232. The conclusion doesn't depend on the exact range. Some confident reappearances are measurably wrong, and nothing in the existing checks flags them.

A wrong position reported confidently is a worse failure than an honest gap. A gap is flagged invalid, and every downstream calculator withholds rather than guesses. A confident wrong position flows into velocities, angles and phase detection as trusted data. E's coverage gain therefore isn't real coverage.

Investigating E did surface a real gap in the production arm (A) too: the same gap-then-reappear pattern happens with no forced resets. That was fixed in `MissedFramePersistence` under the freeze process (`docs/bugs/discontinuity-detector-gap.md`).

## Known costs of staying on A

- `mediapipe==0.10.9` is pinned, because later Windows wheels (0.10.35+ and all of 1.0.0) dropped `mp.solutions.pose`. Moving off this pin means revisiting this decision.
- The racket-arm coverage is still about 50% on the backhand clip, so the gap remains. It is flagged honestly, not solved.
- The decision is single-person only. Multi-person scenes need their own A/B process (`multi-person-tracking`).

## Evidence

- `tools/pose_ab.py` (arms A–D in `ARMS`, arm E in `run_arm_e`) and `assets/outputs/validation/pose_ab/` (`report.json`, per-arm JSON, contact sheets, `arm_e_vs_a_b_backhand.png`). The cache is local-only and regenerable with `python tools/pose_ab.py`.
- `docs/bugs/discontinuity-detector-gap.md`.
