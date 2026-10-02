# Bug: landmark reappearance after a max-hold gap is invisible to the discontinuity detector

Status: **fixed**, in `MissedFramePersistence` (`engine/tracking/pose/persistence.py`, frozen), with the freeze process's evidence requirements satisfied below: full test suite before/after, complete real-clip pipeline diff (every changed value, not a sample), and re-verification of the exact 7 transitions this report originally found. Only fix location 1 from the original proposal was implemented — `find_landmark_discontinuities` (`validation/diagnostics.py`) was deliberately left untouched, one fix / one mechanism.

## What the gap is

`MissedFramePersistence.apply()` (`engine/tracking/pose/persistence.py`, frozen) holds a landmark's last known position for up to `max_missed_frames` (production default: 5) consecutive frames after it drops below the visibility/presence threshold, decaying its confidence each held frame. Beyond that budget, the landmark is marked absent — `MissedFramePersistence`'s own docstring: *"beyond that budget ... it stays absent."*

`find_landmark_discontinuities()` (`validation/diagnostics.py`) — the discontinuity/jump detector used throughout this project's validation tooling — computes a frame-to-frame position delta only between two *consecutive, both-present* frames; it explicitly skips any pair where either side is absent (resets `prev_position`/`prev_index` to `None` and continues).

The two behaviors compose into a gap: **any landmark that goes absent for longer than `max_missed_frames` and then reappears is structurally invisible to the discontinuity detector**, regardless of how large the position change was, and regardless of the model's own reported visibility on the reappearance frame — because there is never a valid "both sides present, consecutive" pair spanning the gap for the detector to evaluate. The reappearance frame is evaluated in isolation, reads as an ordinary confident detection, and nothing downstream is told it just reappeared from nothing.

This is not a smoothing effect and not a magnitude/threshold-tuning question — the detector's precondition (two adjacent present frames) simply doesn't hold across a >5-frame gap, by construction, independent of how the gap arose.

## How it was found

Surfaced while diagnostic-testing an experimental pose-estimation arm (`tools/pose_ab.py`'s arm E — heavy-model VIDEO mode with forced tracker resets after a stall). Arm E's resets are triggered by more than 5 consecutive low-confidence frames on a watched landmark — which is exactly `MissedFramePersistence`'s own hold budget — so every reset produces this exact gap-then-reappear pattern by construction, making it easy to spot: running arm E's output through the actual production chain (`ThresholdVisibilityFilter` → `MissedFramePersistence` → `MovingAverageSmoother` → `find_landmark_discontinuities`), 8 of 26 `right_elbow` reset events and 25 of 27 `right_wrist` reset events reappeared with visibility 0.51-0.98, all unflagged.

Arm E is throwaway diagnostic tooling, not part of the product — so the important question was whether this is an artifact specific to that tooling or a real gap in the system already in production use.

## Reproduction in the current frozen pipeline (arm A, real video)

Confirmed it is **not** an arm E artifact. Ran the same check against **arm A** — `mediapipe.solutions.pose`, the exact detector `ShotPipeline` uses today, no experimental reset logic — on the same real backhand fixture clip (`assets/sample_videos/backhand/sample_backhand2.mp4`), through the same full production chain (`ThresholdVisibilityFilter(min_visibility=0.5, min_presence=0.5)` → `MissedFramePersistence()` with `max_missed_frames=5` → `MovingAverageSmoother(window_size=5)`, all production defaults) and `find_landmark_discontinuities()`.

**7 gap-then-reappear events found on `right_elbow` alone, in 293 frames of ordinary real footage, with no reset logic involved at all:**

| Gap (frames, inclusive) | Gap length | Reappears at | Post-reappearance visibility | Flagged? |
|---|---|---|---|---|
| 15-23 | 9 | 24 | 0.514 | No |
| 66-73 | 8 | 74 | 0.523 | No |
| 118-126 | 9 | 127 | 0.533 | No |
| 204-212 | 9 | 213 | 0.536 | No |
| 223-237 | 15 | 238 | 0.520 | No |
| 255-262 | 8 | 263 | 0.503 | No |
| 276-290 | 15 | 291 | 0.520 | No |

All 7 gaps run 8-15 frames — well past the 5-frame hold budget. All 7 reappear at visibility just above the 0.5 gate (0.503-0.536) — confident enough to pass `ThresholdVisibilityFilter` outright and be treated as an ordinary fresh detection. **None are flagged.** This is the current, frozen, already-shipped pipeline's own behavior on real video, not a hypothetical.

## Severity

**Silent bad data past a confidence gate** — the specific failure mode the visibility-threshold/persistence/discontinuity-detector system exists to prevent. A position that just reappeared after 8-15 frames of no detection at all carries no continuity guarantee whatsoever (the player's elbow could be anywhere), yet:

- It passes `ThresholdVisibilityFilter` (visibility ≥ 0.5).
- `MissedFramePersistence` has no opinion on it — it only decays *held* frames; a fresh reappearance is never touched.
- `find_landmark_discontinuities` cannot see it, per the mechanism above.
- Every downstream consumer — `VelocityCalculator`/`AccelerationCalculator` (confirmed in the prior arm E investigation: real, non-`invalid` velocity/acceleration values get computed straight through these reappearance frames), joint-angle calculators, `weight_transfer`, any future scoring/feedback layer — receives it as an ordinary, trustworthy, confidently-tracked frame.

7 occurrences in one 293-frame clip on one landmark is not rare.

## Fix implemented

**Location: `MissedFramePersistence.apply()`, `engine/tracking/pose/persistence.py` (frozen). Fix location 2 (`find_landmark_discontinuities`) was explicitly not touched — one fix, one mechanism.**

`MissedFramePersistence` already tracked `missed_counts` per landmark internally. The fix adds one more piece of state: an `exhausted` set, marking landmarks that went genuinely absent (hold budget fully spent, at least one frame with no output at all — not merely a run that stayed within budget the whole time). When a landmark in that set reappears with a fresh raw detection, its visibility/presence for that one frame are overridden by construction — run through the same `ConfidenceDecayModel` already used for held frames, continued one step past the last held frame (`max_missed_frames + 1`) — regardless of what the raw detector itself reported. `last_known` is still updated from the *pristine* fresh reading (not the de-rated output), so the existing "decay never compounds against an already-decayed value" invariant (`ConfidenceDecayModel`'s own docstring contract) holds for this case too; only the one reappearance frame's output is de-rated, tracking trusts the next frame normally.

A landmark reappearing from a gap that *stayed within* budget (never genuinely absent, every intervening frame successfully held) is deliberately unaffected — `find_landmark_discontinuities` can and does evaluate that transition normally, since both sides are present there; that case was never the blind spot.

### 1. Full test suite, before and after

Isolated the fix with `git stash` (only `persistence.py` + its test file) so this reflects the fix's effect alone, nothing else:

- **Before** (fix stashed out): `286 tests, 0 failures, 0 errors, 0 skipped`.
- **After** (fix applied): `291 tests, 0 failures, 0 errors, 0 skipped`.

The 5 new tests (`tests/tracking/pose/test_persistence.py`, `MissedFramePersistenceReappearanceAfterGapTests`) cover: the exact decay formula applied on reappearance; a high-raw-confidence (0.99) reappearance still landing below the 0.5 threshold at production's `max_missed_frames=5`; that a within-budget reappearance is untouched (regression-locks the pre-existing `test_confidence_resets_to_full_when_a_fresh_detection_reappears` behavior); that position is the fresh reading, not the stale one; and that a hold immediately after a de-rated reappearance decays from the reappearance's *raw* value, not its de-rated output (the non-compounding invariant). All 15 pre-existing tests in the file pass unchanged — none of their assertions touch the gap-exceeding-budget case, confirmed by inspection before writing the fix.

### 2. Real-clip pipeline diff, before vs. fixed — every changed value

Ran both real fixture clips (`backhand` = `sample_backhand2.mp4`, 293 frames; `forehand` = the "Serious Squash..." clip, 941 frames) through the exact chain `ShotPipeline` uses (`ThresholdVisibilityFilter` → `MissedFramePersistence` → `MovingAverageSmoother`, then `_compute_joint_angles`/`_compute_kinematics`/`_compute_posture`) on the same cached raw arm-A detections (raw detection happens before persistence, so it's identical in both runs — confirmed no other differences leak in). Diffed every leaf value in the output, before vs. after.

**Total changed values: 1671** (backhand: 72 angle_measurements, 100 kinematics, 44 landmark_frames, 112 posture; forehand: 327 / 343 / 182 / 491 respectively). That number is large but fully accounted for — it's not 1671 independent changes, it's 113 root causes plus their deterministic, frozen-and-unmodified downstream consequences. All 113 root changes, complete, no sampling:

| Clip | Frame | Landmark | Visibility before | Visibility after |
|---|---|---|---|---|
| backhand | 23 | right_wrist | 0.5081 | 0.0904 |
| backhand | 24 | right_elbow | 0.5136 | 0.0914 |
| backhand | 49 | right_elbow | 0.5011 | 0.0892 |
| backhand | 50 | right_wrist | 0.5271 | 0.0938 |
| backhand | 70 | right_wrist | 0.5031 | 0.0895 |
| backhand | 74 | right_elbow | 0.5232 | 0.0931 |
| backhand | 101 | right_wrist | 0.5395 | 0.0960 |
| backhand | 125 | right_wrist | 0.5167 | 0.0920 |
| backhand | 127 | right_elbow | 0.5331 | 0.0949 |
| backhand | 155 | right_elbow | 0.5061 | 0.0901 |
| backhand | 156 | right_wrist | 0.5031 | 0.0895 |
| backhand | 179 | right_elbow | 0.5011 | 0.0892 |
| backhand | 179 | right_wrist | 0.5004 | 0.0891 |
| backhand | 213 | right_elbow | 0.5358 | 0.0954 |
| backhand | 214 | right_wrist | 0.5139 | 0.0915 |
| backhand | 237 | right_wrist | 0.5226 | 0.0930 |
| backhand | 238 | right_elbow | 0.5199 | 0.0925 |
| backhand | 263 | right_elbow | 0.5029 | 0.0895 |
| backhand | 265 | right_wrist | 0.5208 | 0.0927 |
| backhand | 287 | right_knee | 0.5187 | 0.0923 |
| backhand | 289 | right_wrist | 0.5229 | 0.0931 |
| backhand | 291 | right_elbow | 0.5197 | 0.0925 |
| forehand | 23 | left_elbow | 0.5012 | 0.0892 |
| forehand | 46 | left_ankle | 0.5174 | 0.0921 |
| forehand | 49 | left_elbow | 0.5172 | 0.0920 |
| forehand | 52 | left_knee | 0.5118 | 0.0911 |
| forehand | 52 | left_wrist | 0.5167 | 0.0920 |
| forehand | 63 | right_wrist | 0.5311 | 0.0945 |
| forehand | 74 | left_knee | 0.5151 | 0.0917 |
| forehand | 75 | left_wrist | 0.5068 | 0.0902 |
| forehand | 79 | left_elbow | 0.5033 | 0.0896 |
| forehand | 97 | left_elbow | 0.5243 | 0.0933 |
| forehand | 114 | right_wrist | 0.5196 | 0.0925 |
| forehand | 128 | left_wrist | 0.5131 | 0.0913 |
| forehand | 130 | left_elbow | 0.5207 | 0.0927 |
| forehand | 147 | left_elbow | 0.5009 | 0.0891 |
| forehand | 170 | right_wrist | 0.5303 | 0.0944 |
| forehand | 204 | left_elbow | 0.5035 | 0.0896 |
| forehand | 221 | right_wrist | 0.5241 | 0.0933 |
| forehand | 242 | left_elbow | 0.5119 | 0.0911 |
| forehand | 247 | right_wrist | 0.5328 | 0.0948 |
| forehand | 254 | left_foot_index | 0.5062 | 0.0901 |
| forehand | 260 | left_elbow | 0.5017 | 0.0893 |
| forehand | 262 | left_foot_index | 0.5107 | 0.0909 |
| forehand | 263 | left_ankle | 0.5195 | 0.0925 |
| forehand | 265 | left_knee | 0.5229 | 0.0931 |
| forehand | 265 | left_wrist | 0.5167 | 0.0920 |
| forehand | 285 | right_wrist | 0.5008 | 0.0891 |
| forehand | 299 | left_knee | 0.5110 | 0.0909 |
| forehand | 313 | left_elbow | 0.5181 | 0.0922 |
| forehand | 318 | left_knee | 0.5041 | 0.0897 |
| forehand | 333 | right_wrist | 0.5028 | 0.0895 |
| forehand | 349 | left_elbow | 0.5288 | 0.0941 |
| forehand | 366 | left_knee | 0.5065 | 0.0901 |
| forehand | 367 | left_elbow | 0.5282 | 0.0940 |
| forehand | 382 | right_wrist | 0.5275 | 0.0939 |
| forehand | 424 | left_elbow | 0.5139 | 0.0915 |
| forehand | 430 | left_wrist | 0.5109 | 0.0909 |
| forehand | 439 | right_wrist | 0.5025 | 0.0894 |
| forehand | 454 | left_wrist | 0.5410 | 0.0963 |
| forehand | 455 | left_elbow | 0.5297 | 0.0943 |
| forehand | 477 | left_elbow | 0.5142 | 0.0915 |
| forehand | 496 | right_wrist | 0.5304 | 0.0944 |
| forehand | 510 | left_wrist | 0.5333 | 0.0949 |
| forehand | 512 | left_elbow | 0.5069 | 0.0902 |
| forehand | 532 | left_ankle | 0.5050 | 0.0899 |
| forehand | 533 | left_elbow | 0.5153 | 0.0917 |
| forehand | 535 | left_knee | 0.5134 | 0.0914 |
| forehand | 551 | right_wrist | 0.5024 | 0.0894 |
| forehand | 561 | left_elbow | 0.5057 | 0.0900 |
| forehand | 562 | left_wrist | 0.5359 | 0.0954 |
| forehand | 588 | left_elbow | 0.5481 | 0.0976 |
| forehand | 589 | left_foot_index | 0.5208 | 0.0927 |
| forehand | 590 | left_ankle | 0.5209 | 0.0927 |
| forehand | 591 | left_wrist | 0.5115 | 0.0910 |
| forehand | 596 | left_knee | 0.5374 | 0.0956 |
| forehand | 609 | right_wrist | 0.5349 | 0.0952 |
| forehand | 613 | left_wrist | 0.5249 | 0.0934 |
| forehand | 614 | left_elbow | 0.5187 | 0.0923 |
| forehand | 625 | left_knee | 0.5060 | 0.0901 |
| forehand | 640 | left_elbow | 0.5085 | 0.0905 |
| forehand | 646 | left_ankle | 0.5057 | 0.0900 |
| forehand | 647 | left_foot_index | 0.5026 | 0.0894 |
| forehand | 648 | left_knee | 0.5047 | 0.0898 |
| forehand | 649 | left_wrist | 0.5211 | 0.0927 |
| forehand | 658 | right_wrist | 0.5147 | 0.0916 |
| forehand | 673 | left_knee | 0.5009 | 0.0891 |
| forehand | 674 | left_wrist | 0.5276 | 0.0939 |
| forehand | 675 | left_elbow | 0.5069 | 0.0902 |
| forehand | 689 | left_knee | 0.5038 | 0.0897 |
| forehand | 695 | left_elbow | 0.5098 | 0.0907 |
| forehand | 714 | right_wrist | 0.5379 | 0.0957 |
| forehand | 731 | left_wrist | 0.5301 | 0.0943 |
| forehand | 732 | left_elbow | 0.5300 | 0.0943 |
| forehand | 749 | left_elbow | 0.5246 | 0.0934 |
| forehand | 754 | left_knee | 0.5093 | 0.0906 |
| forehand | 772 | right_wrist | 0.5427 | 0.0966 |
| forehand | 786 | left_elbow | 0.5102 | 0.0908 |
| forehand | 787 | left_wrist | 0.5366 | 0.0955 |
| forehand | 822 | right_wrist | 0.5314 | 0.0946 |
| forehand | 829 | left_foot_index | 0.5124 | 0.0912 |
| forehand | 832 | left_knee | 0.5137 | 0.0914 |
| forehand | 842 | left_ankle | 0.5225 | 0.0930 |
| forehand | 844 | left_wrist | 0.5161 | 0.0919 |
| forehand | 846 | left_elbow | 0.5107 | 0.0909 |
| forehand | 864 | left_elbow | 0.5196 | 0.0925 |
| forehand | 866 | left_wrist | 0.5121 | 0.0912 |
| forehand | 876 | right_wrist | 0.5133 | 0.0914 |
| forehand | 899 | left_wrist | 0.5209 | 0.0927 |
| forehand | 917 | left_elbow | 0.5402 | 0.0961 |
| forehand | 919 | left_wrist | 0.5153 | 0.0917 |
| forehand | 920 | left_knee | 0.5030 | 0.0895 |
| forehand | 933 | right_wrist | 0.5352 | 0.0953 |

Every one of these 113: visibility and presence drop from just above the 0.5 gate (0.500-0.548 — confident enough to have passed `ThresholdVisibilityFilter` outright) to ~0.089-0.098 (`raw_visibility * 0.75**6`). **Position is unchanged in all 113** — confirmed explicitly, this fix never touches position, exactly as designed.

Everything else in the 1671 is a deterministic consequence of those 113, through frozen code this fix did not modify:

- **`angle_measurements` (399 changes across both clips):** at each root frame, the relevant joint angle's `is_valid` flips `True → False` (e.g. backhand frame 24, `elbow_right`: `angle_degrees=144.18°, confidence=0.5136, is_valid=True` → `angle_degrees=None, confidence=0.0, is_valid=False`) — `ThreePointAngleCalculator` already enforces `MIN_LANDMARK_VISIBILITY=0.5` on its inputs; it was never given the chance to reject this frame before, because the input never dropped below its threshold. Nothing about the angle math changed — the calculator's own pre-existing check now actually fires.
- **`kinematics` (443 changes):** at each root frame, angular velocity/acceleration likewise flip to `is_valid=False` there. At neighboring frames, the *numeric value* shifts slightly (e.g. backhand frame 25 angular velocity: -108.93°/s → -98.43°/s) — this is `differentiate_trajectory` (unmodified, frozen) correctly recomputing its central-difference stencil against the nearest *valid* neighbor now that one sample dropped out of the trajectory, not a new bug.
- **`posture` (603 changes):** mostly `center_of_mass`/`weight_transfer` **confidence increasing** at root frames, not decreasing (e.g. backhand frame 24 `weight_transfer`: `confidence=0.5136 → 0.7122`). `CenterOfMassCalculator` already drops any body segment whose landmarks aren't individually usable (`visibility ≥ 0.5`) and renormalizes over what's left — that logic existed and was already tested, but was inert for these frames because the bad elbow/wrist reading was passing the gate. Now that it's correctly excluded, the segment-dropping path finally engages: the weak link no longer drags the confidence `min()` down, and the position estimate itself shifts slightly (visible in `right_foot_ratio`) because the average no longer includes an untrustworthy arm segment. This is the fix *improving* a downstream number's accuracy, not just invalidating one.

### 3. Original 7 transitions, re-verified against the fix

| Frame | Visibility before | Visibility after | Below 0.5? | `find_landmark_discontinuities` flagged? |
|---|---|---|---|---|
| 24 | 0.514 | 0.0914 | Yes | No |
| 74 | 0.523 | 0.0931 | Yes | No |
| 127 | 0.533 | 0.0949 | Yes | No |
| 213 | 0.536 | 0.0954 | Yes | No |
| 238 | 0.520 | 0.0925 | Yes | No |
| 263 | 0.503 | 0.0895 | Yes | No |
| 291 | 0.520 | 0.0925 | Yes | No |

**Handled, not flagged** — exactly as expected given the fix's mechanism. `find_landmark_discontinuities` still doesn't fire (it was deliberately left untouched — it only ever looks at position, never visibility, so a confidence-only fix was never going to change its output). But every one of these frames is now correctly below the 0.5 threshold every downstream calculator already enforces, so `ThreePointAngleCalculator`, `CenterOfMassCalculator`, `WeightTransferCalculator`, and everything else that gates on landmark confidence now rejects them instead of silently trusting them. That's the actual failure mode from the "Severity" section above — closed.

### What was not touched

Only `engine/tracking/pose/persistence.py` (frozen, this fix) and its test file changed. `validation/diagnostics.py`/`find_landmark_discontinuities` — untouched, per "one fix, one mechanism." No other frozen file was modified.
