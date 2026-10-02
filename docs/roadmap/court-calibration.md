---
id: court-calibration
status: in-progress
depends_on: [tracking-biomechanics]
blocks: [single-player-heatmap, shot-outcome, multiplayer-heatmap, camera-motion-tracking]
---

## Goal

Map a player's foot position from pixel space to real court coordinates (metres), through a floor homography computed from manually placed court landmarks. Automatic court-line detection waits until manual calibration is proven.

## What's built (2026-10-02)

- **`engine/calibration/court_geometry.py`:** singles court floor dimensions, 14 named clickable floor points (line intersections and line/wall junctions) and the floor-line set used for overlays. The origin is the front-left floor corner; x runs across the court and y runs back from the front wall.
- **`engine/calibration/court_homography.py`:**
  - The fit is a normalized DLT in pure numpy (the engine stays free of cv2), giving pixel↔court conversion.
  - It refuses degenerate point sets, including 3-on-one-line-plus-1, which looks fine in pixels because of click noise. Without that check the fit returns garbage.
  - It includes leave-one-out, an extrapolation flag (outside the clicked points' hull), and horizon/behind-camera handling.
  - Foot positions use each foot's toe landmark, gated on visibility.
- **`tools/calibrate_court.py`:** an interactive click tool in the style of `label.py`, with a live line overlay. It also takes `--set` for non-interactive entry, and each point has an optional note (the `review_phases.py` convention). It writes `calibrations/<clip>.json`, committed like `labels/`.
- **`tools/validate_court_calibration.py`:** runs leave-one-out, start/middle/end line overlays and foot sanity checks, and writes `docs/evidence/court_calibration/`.

## Validation result: not yet trustworthy

The full numbers and images are in `docs/evidence/court_calibration/README.md`.

- **The camera moves in both calibrated clips** (sample_backhand3 pans, and sample_forehand1 drifts or zooms). A single homography is valid only near its calibration frame. On those frames the projected lines sit on the real lines by eye.
- **Leave-one-out error depends strongly on position:**
  - Service-box points: 0.03–0.15 m.
  - Front-wall corners predicted from 5+ m away: 0.5 m (forehand1) and 1.75 m (backhand3; that corner was inferred, not directly visible).
  - Five points can't separate lens distortion from extrapolation leverage or click error, so **whether distortion correction is needed is undetermined, not ruled out.**
- **Every foot position is extrapolated:** players stand nearer the camera than any clickable point on this rear-view footage.

## What's needed next

1. **Handle camera motion.** The pre-flight check now flags moving cameras (provisional 3% drift warning; it flags both calibrated clips). Per-frame re-estimation is being researched in `camera-motion-tracking`: on forehand1 it holds within about 4 px for about 3 s and drifts by about 1–1.5 px/s after that. The options remain: re-estimate the homography per frame (register each frame to the calibration frame, or track the clicked lines), or require tripod footage for this feature and check for motion at upload.
2. **More points spread across the floor**, plus at least one known-position checkpoint left out of the fit, on footage that shows them. That is what decides the lens-distortion question.
3. **A human re-click** of the existing calibrations with `tools/calibrate_court.py`. The current points were placed by Claude from zoomed crops.

## Validation bar

- Held-out error, per point and per court region, below an agreed tolerance on footage with a verified static camera (or with per-frame tracking).
- The overlay is checked by eye on the start, middle and end frames.
- Positions in the region where the player actually stands are not extrapolated.

## Open questions

- Is rear-wall footage enough for this at all, given that the back third of the court is never in front of the camera?
- Toe landmarks lift with the foot. Is a "planted foot" filter needed before positions feed a heatmap?
