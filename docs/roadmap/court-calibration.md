---
id: court-calibration
status: not-started
depends_on: [tracking-biomechanics]
blocks: [single-player-heatmap, shot-outcome, multiplayer-heatmap]
---

## Goal

Map a player's foot and body position from pixel space to real court coordinates through a homography computed from manually clicked court corners and line intersections. Automatic court-line detection waits until manual calibration is proven.

## What's needed

- A click-to-calibrate tool that collects at least 4 known court points (floor corners, short line, half-court line, service boxes) per fixed camera.
- A homography fit, and a `CalibrationProvider` implementation (`pixel_to_world()` already exists as a Protocol in `engine/calibration/interfaces.py`).
- A floor-contact point per frame (the ankles or feet from the existing landmarks), with validity propagated from the landmark confidence.

## Validation bar

Reprojection error measured on held-out court points that weren't used in the fit, reported in centimetres. A known-position test, with a player standing on the T and in each corner, lands within an agreed tolerance.

## Open questions

- Is a single floor-plane homography enough, given the camera angles in the corpus? Does lens distortion need `CameraIntrinsics`?
- Does the camera move within a clip? If so, calibration has to be per-segment.

## Evidence so far

- Only the interface stub exists (`engine/calibration/interfaces.py`: `CalibrationProvider.load()`, `pixel_to_world()`, `CameraIntrinsics`, `CalibrationConfig`).
