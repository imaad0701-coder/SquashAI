---
id: camera-motion-tracking
status: in-progress
depends_on: [court-calibration]
blocks: [continuous-camera-tracking]
---

## Goal

Court positions for **individual swings** on moving-camera footage. The calibration is carried a short way, forwards or backwards, from its anchor frame to a swing's contact frame by tracking the floor. This short-horizon, per-swing case is the **v1 scope**.

Continuous tracking of the whole clip, with drift correction, is **not** in this node. It's a separate, not-started item: `continuous-camera-tracking`.

## What's built (2026-10-02)

- **`engine/calibration/floor_tracking.py`:**
  - Tracks floor-only features, with the floor polygon warped along with the camera and the player masked out.
  - Fits a RANSAC homography per step between frames, in either direction from the anchor. Backward decoding works in chunks.
  - **Stops at the first gap.** Nothing past a gap is returned.
- **`engine/calibration/court_capabilities.py`,** the practical split:
  - **Per swing** (`contact_court_positions`): available within **5 s** of the anchor in either direction on a moving camera. Each result carries the direction, the real elapsed time and the implied tracking error in pixels and metres at the feet. Beyond 5 s, or past a tracking gap, the result is unavailable with the reason. A static camera maps directly at any distance.
  - **Whole clip** (`movement_trail`): available **only on a static camera**, as classified by the camera-motion check. On a moving or unmeasurable camera it is unavailable, with a message saying why.
- **`engine/calibration/camera_motion.py`:** the camera-motion measurement and its classification (static, moving or unknown). It moved here from the demo so the engine can gate on it; the demo pre-flight check re-exports it.

## Evidence

`docs/evidence/camera_motion/README.md` has the full tables. The ground truth was placed by eye, independently of the tracker, at 30 points.

- **Forward** (sample_forehand1, from frame 0): 1.4–3.8 px at 1–3 s, 7–9 px at 7 s, 12–20 px at 13 s.
- **Backward** (sample_backhand3, from frame 286): 2–8 px at 1.7–3.2 s, up to 11 px at 4.9 s, 10–17 px at 6.8–8.5 s.
- **Normalized by frame diagonal, both directions behave the same,** growing at roughly 0.05–0.1% of the diagonal per second. Backward doesn't degrade differently.
- **The 5 s horizon and the error envelope are provisional,** taken from these two clips. Every reading within 5 s fell inside 0.3% + 0.05% per second of the frame diagonal.
- **No tracking gaps occurred** on either clip, which both have wood-grain floors. Backward reached frame 0 with at least 68 inliers. What a gap costs on real footage is still unmeasured.

On the real clips, `tools/court_capability_report.py` gives positions for 3 of 4 detected swings on backhand3 and 3 of 6 on forehand1. The rest are unavailable as beyond the horizon, and the movement trail is unavailable on both.

## Caveats carried by every per-swing position

- The implied error is **tracking error only**. It doesn't include the calibration's own error.
- On this rear-wall footage every foot position is also **extrapolated**, because the players stand nearer the camera than any clickable court point. The real uncertainty is therefore larger than the ± figure (see `court-calibration`).
- The evidence is two clips. The horizon and the envelope need re-checking on more footage.

## Validation bar for v1

- Ground truth on at least two more moving-camera clips confirms that the 5 s envelope holds, or tightens it.
- At least one clip with real tracking gaps shows the gap path behaving as designed.
- Per-swing positions near the anchor are spot-checked against foot positions read by eye.

## Open questions

- Should the horizon depend on measured drift speed (a clip that moves faster drifts faster)?
- With several calibration anchors per clip, which needs several human clicks per clip, more swings could fall within 5 s of one. Is that an acceptable workflow?
