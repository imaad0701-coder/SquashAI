"""Camera-motion measurement: how far the camera drifts over a clip,
from background features. Lives in the engine (not the demo) because
court-coordinate capabilities depend on it: whole-clip court positions are
only offered when the camera is static (engine.calibration.court_capabilities).
The demo's pre-flight check reports it to the user.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Final

# Camera motion: sampled in real time (not a frame count) so the check means
# the same thing at 30 and 60 fps.
MOTION_SAMPLE_INTERVAL_MS: Final[float] = 100.0
MOTION_ANALYSIS_SHORT_SIDE_PX: Final[int] = 360
MOTION_MAX_FEATURES: Final[int] = 300
MOTION_MIN_INLIERS: Final[int] = 12  # below this, a frame-to-frame transform isn't trusted; the span is a gap
MOTION_RANSAC_PX: Final[float] = 2.0
# PROVISIONAL, from 7 clips (2026-10-02): visually confirmed static cameras
# measured 0.19-1.14% (the 1.14% is false drift on a close-up where the
# player fills the frame -- the check's noise floor), visually confirmed
# moving cameras 6.76-9.87%. 3% sits in that gap; it is not an established
# threshold. For calibration purposes even ~1% matters (1% of a 720x1280
# diagonal is ~15 px, above the click error of a calibration point), so a
# PASS here means "no large motion detected", not "safe to calibrate".
CAMERA_MOTION_WARN_PCT: Final[float] = 3.0


class CameraMotionStatus(Enum):
    STATIC = "static"  # drift below CAMERA_MOTION_WARN_PCT and no tracking gaps (drift < ~1% is undetectable either way)
    MOVING = "moving"
    UNKNOWN = "unknown"  # unmeasurable, or tracking gaps left part of the clip unmeasured


@dataclass(frozen=True)
class CameraMotion:
    """Camera drift relative to the first frame, from background features.

    max_drift_pct: largest displacement of any frame corner from its
    first-frame position, as a percentage of the frame diagonal (resolution
    independent). Measured only over spans where tracking held; time where it
    didn't is reported in gap_ms, never interpolated across."""

    max_drift_pct: float
    max_drift_at_ms: float
    max_rotation_deg: float
    max_scale_change_pct: float
    measured_ms: float  # real time covered by trusted frame-to-frame transforms
    gap_ms: float  # real time where too few background features tracked to measure
    samples: int
    median_inlier_ratio: float


def measure_camera_motion(path: str) -> CameraMotion | None:
    """Tracks corner features (Shi-Tomasi) between frames sampled every
    MOTION_SAMPLE_INTERVAL_MS with pyramidal Lucas-Kanade optical flow, fits
    a RANSAC similarity transform per step (so features on the moving player
    are rejected as outliers as long as the background dominates), and
    composes the steps into a camera pose relative to frame 0. When a step
    has too few inliers, that span is counted as a gap and the chain
    restarts from the current frame's pose being unknown: drift after a gap
    is measured relative to the frame where tracking resumed, which can only
    under-report total drift, never invent it."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(MOTION_SAMPLE_INTERVAL_MS * fps / 1000.0))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    h0, w0 = frame.shape[:2]
    scale = MOTION_ANALYSIS_SHORT_SIDE_PX / min(h0, w0)
    size = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
    corners = np.float32([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]).reshape(-1, 1, 2)
    diag = float(np.hypot(*size))

    def gray(img):
        return cv2.cvtColor(cv2.resize(img, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    prev = gray(frame)
    pose = np.eye(3)  # maps current-segment reference pixels -> current frame pixels
    max_drift = max_rot = max_scale = 0.0
    max_at = measured = gap = 0.0
    ratios = []
    samples = 0
    index = 0
    dt = step * 1000.0 / fps
    while True:
        for _ in range(step - 1):
            if not cap.grab():
                break
        ok, frame = cap.read()
        if not ok:
            break
        index += step
        samples += 1
        cur = gray(frame)
        pts = cv2.goodFeaturesToTrack(prev, MOTION_MAX_FEATURES, 0.01, 8)
        affine = None
        if pts is not None and len(pts) >= MOTION_MIN_INLIERS:
            nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
            good = status.reshape(-1) == 1
            if good.sum() >= MOTION_MIN_INLIERS:
                affine, inliers = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                              ransacReprojThreshold=MOTION_RANSAC_PX)
                n_in = 0 if inliers is None else int(inliers.sum())
                if affine is None or n_in < MOTION_MIN_INLIERS:
                    affine = None
                else:
                    ratios.append(n_in / len(pts))
        if affine is None:
            gap += dt
            pose = np.eye(3)  # pose unknown across the gap: restart the reference here
        else:
            measured += dt
            pose = np.vstack([affine, [0, 0, 1]]) @ pose
            moved = cv2.perspectiveTransform(corners, pose)
            drift = float(np.max(np.linalg.norm((moved - corners).reshape(-1, 2), axis=1))) / diag * 100
            a, b = pose[0, 0], pose[1, 0]
            rot = abs(float(np.degrees(np.arctan2(b, a))))
            scl = abs(float(np.hypot(a, b)) - 1.0) * 100
            if drift > max_drift:
                max_drift, max_at = drift, index * 1000.0 / fps
            max_rot, max_scale = max(max_rot, rot), max(max_scale, scl)
        prev = cur
    cap.release()
    return CameraMotion(
        max_drift_pct=max_drift, max_drift_at_ms=max_at, max_rotation_deg=max_rot, max_scale_change_pct=max_scale,
        measured_ms=measured, gap_ms=gap, samples=samples,
        median_inlier_ratio=statistics.median(ratios) if ratios else 0.0,
    )


def classify_camera_motion(motion: "CameraMotion | None") -> CameraMotionStatus:
    if motion is None or motion.measured_ms == 0:
        return CameraMotionStatus.UNKNOWN
    if motion.max_drift_pct >= CAMERA_MOTION_WARN_PCT:
        return CameraMotionStatus.MOVING
    return CameraMotionStatus.UNKNOWN if motion.gap_ms > 0 else CameraMotionStatus.STATIC
