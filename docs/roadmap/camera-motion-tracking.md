---
id: camera-motion-tracking
status: research
depends_on: [court-calibration]
blocks: []
---

## Goal

Keep a court calibration valid while the camera moves (handheld, panning or zooming), so that court coordinates work on the footage people actually record, without requiring a tripod.

## What was tried (time-boxed experiment, 2026-10-02)

Calibrate once, track floor features from frame to frame, and compose per-step homographies to carry the calibration forward. The method, numbers and plot are in `docs/evidence/camera_motion/README.md`, produced by `tools/camera_motion_experiment.py`.

## Findings

- **sample_forehand1:** error at the calibration points, measured against ground truth placed independently, is 1.4–3.8 px at 1–3 s, 7–9 px at 7 s and 12–20 px at 13 s. That's steady accumulation of roughly 1–1.5 px per second, 4–8× better than not tracking (14 px at 1 s and 50–76 px from 5 s on).
- **No tracking gaps** were observed on either tested clip (forehand1, 13.3 s; backhand3, 3.6 s forward), because the wood-grain floors give plenty of texture. What a gap costs is **unmeasured**: the gap-handling path never ran on real data.
- Errors are reported against real elapsed time, not frame count, so the curve can be compared across clips at different frame rates.

## Assessment

Promising, but it needs more R&D before anything depends on it.
- It works within the calibration's own error for about 3–5 s.
- Longer clips need drift control. Options are re-anchoring to the court lines periodically, matching each frame directly to the calibration frame (or to keyframes) instead of chaining, or bundle-adjusting the chain.
- The evidence is one clip with ground truth, so it isn't a general result.

## What's needed before this leaves research

1. Ground truth on at least two more moving-camera clips (backhand3, forehand3) to see whether ~1–1.5 px/s holds.
2. A drift-control method, measured on the same ground truth.
3. Footage that actually produces tracking gaps (a plain floor, heavy occlusion, fast pans), to measure what a gap costs and how recovery works.
4. Tracking both forwards **and backwards** from the calibration frame. Only forwards was tested.

## Validation bar

On held-out moving-camera clips, error at the calibration points stays within the calibration's own leave-one-out error for the full clip length. Any span where tracking was lost is flagged in the output, never bridged silently.

## Open questions

- Is a tripod requirement (enforced by the pre-flight camera-motion check) simpler and good enough for the first release?
- Does zoom (the scale change seen on forehand3, 4.6%) behave differently from panning for drift?
