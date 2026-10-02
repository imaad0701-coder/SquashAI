---
id: contact-phase-detection
status: in-progress
depends_on: [tracking-biomechanics]
blocks: [findings-engine, shot-type-classifier]
---

## Goal

Detect each swing in a clip, its contact frame, and its phase boundaries (prep, backswing, forward swing, contact, follow-through, recovery) from wrist-speed and rotation kinematics.

## What's needed

- A different detection approach for the `forward_swing` boundary. No correction layered on the current signals predicts its error.
- A combined duration-and-prominence window filter to replace duration-only `SWING_MIN_WINDOW_MS`.
- A fix for the neighbour-widening side effect that costs `follow_through` accuracy.
- A larger labelled corpus. There are 6 clips in `labels/` today.

## Validation bar

`tools/eval_phases.py` against labelled ground truth, plus human phase review (`tools/review_phases.py`, `reviews/phase_review.json`). It isn't validated for production until every boundary that `AnalysisResult` reports as reliable meets an agreed error tolerance across a corpus covering several players and camera angles.

## Open questions

- Tight rally pacing under-segments windows, and no single threshold on depth or duration separates real rests from mid-swing dips (`docs/STATUS.md`).
- Swings whose footage starts mid-motion have no usable prep signal. This comes from the source footage and can't be tuned away.

## Evidence so far

- `engine/phases/contact_detection.py` and `phase_detector.py` (`KinematicPhaseDetector`) are implemented. Contact match rate is about 94% with sub-5-frame error on the 6-clip eval (from the `AnalysisResult` docstring; numbers change as `labels/` grows).
- Four known limitations are quantified in `docs/STATUS.md` (2026-08-27 to 2026-08-29).
