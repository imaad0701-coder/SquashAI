---
id: continuous-camera-tracking
status: not-started
depends_on: [camera-motion-tracking]
blocks: []
---

## Goal

Court coordinates for **every frame** of a moving-camera clip, so that whole-clip outputs (movement trails, heatmaps) work without a tripod.

**Explicitly out of scope for now.** Until this exists, whole-clip outputs require a static camera, and on a moving camera they report themselves as unavailable (`engine/calibration/court_capabilities.movement_trail`).

## Why it's separate from camera-motion-tracking

Chained frame-to-frame floor tracking accumulates error at roughly 0.05–0.1% of the frame diagonal per second, in both directions (`docs/evidence/camera_motion/`). It stays inside the calibration's own error for only a few seconds. A full clip needs **drift correction**, which is a different problem from short-horizon tracking.

## What's needed

- A drift-correction method, with candidates including:
  - re-anchoring to the court lines periodically (automatic line matching)
  - matching each frame directly to one or more keyframes instead of chaining
  - optimizing the whole chain at once (bundle adjustment)
- Ground truth across whole moving-camera clips, measured the same way as before: independent readings by eye, with error against real elapsed time.
- Gap handling that recovers after occlusion or fast pans, not just stops.

## Validation bar

On held-out moving-camera clips, error at known court points stays within the calibration's own leave-one-out error for the **entire** clip. Any unmeasured span is flagged, never bridged.

## Evidence so far

None for this node beyond the drift measurements that motivate it.
