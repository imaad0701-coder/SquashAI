"""Carries a single-frame court calibration a short way through a
moving-camera clip by tracking the floor, forwards or backwards from the
calibration frame (the "anchor").

Method: Shi-Tomasi corners detected inside the floor region (the
calibration's floor polygon, warped along with the camera) with the player
masked out, pyramidal Lucas-Kanade flow between consecutive frames, a RANSAC
homography per step. Floor points lie on one plane, so a homography is the
exact frame-to-frame map for the floor even when the camera translates.
Composed steps give anchor-pixels -> frame-pixels.

Error accumulates with every composed step. Measured against independently
placed ground truth (docs/evidence/camera_motion/), it stayed within the
calibration's own point error for only the first few seconds -- this module
is for short horizons around the anchor (see court_capabilities), not for
continuous whole-clip tracking, which needs drift correction that doesn't
exist yet.

A step with fewer than MIN_INLIERS floor inliers is a GAP: the camera's
motion during it is unknown. Tracking stops there -- nothing past a gap is
returned as if it were measured.

This is the one engine module that uses OpenCV (already installed as a
mediapipe dependency); it is imported lazily so the rest of the engine
doesn't pull it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from engine.calibration.court_geometry import COURT_LENGTH_M, COURT_WIDTH_M
from engine.calibration.court_homography import CourtCalibration

MIN_INLIERS: Final[int] = 15
RANSAC_PX: Final[float] = 2.0
MAX_FEATURES: Final[int] = 400
_CHUNK_FRAMES: Final[int] = 30  # backward decoding: frames held in memory at once


@dataclass(frozen=True)
class TrackStep:
    frame_index: int
    elapsed_ms: float  # |timestamp - anchor timestamp|, real time
    anchor_to_frame: np.ndarray  # 3x3: anchor-frame pixels -> this frame's pixels
    inliers: int


@dataclass(frozen=True)
class FloorTrack:
    anchor_frame: int
    direction: str  # "forward" | "backward"
    steps: tuple[TrackStep, ...]  # contiguous from the anchor; ends early at a gap
    stopped_at_gap: int | None  # first frame whose step had too few floor inliers, if any
    min_inliers: int | None

    def step_for(self, frame_index: int) -> TrackStep | None:
        for s in self.steps:
            if s.frame_index == frame_index:
                return s
        return None


def _floor_mask(cal: CourtCalibration, shape) -> np.ndarray:
    import cv2

    corners = [(0, 0), (COURT_WIDTH_M, 0), (COURT_WIDTH_M, COURT_LENGTH_M), (0, COURT_LENGTH_M)]
    px = [cal.homography.court_to_pixel(*c) for c in corners]
    mask = np.zeros(shape[:2], np.uint8)
    if all(p is not None for p in px):
        cv2.fillPoly(mask, [np.int32(np.round(px))], 255)
    return mask


def _frames(cap, start: int, stop: int, direction: str):
    """Yields (index, gray) from start towards stop (inclusive), one frame at
    a time; backward runs decode forward in chunks and replay them reversed."""
    import cv2

    if direction == "forward":
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for index in range(start, stop + 1):
            ok, frame = cap.read()
            if not ok:
                return
            yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return
    hi = start
    while hi >= stop:
        lo = max(stop, hi - _CHUNK_FRAMES + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        chunk = []
        for _ in range(lo, hi + 1):
            ok, frame = cap.read()
            if not ok:
                break
            chunk.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for offset in range(len(chunk) - 1, -1, -1):
            yield lo + offset, chunk[offset]
        hi = lo - 1


def track_floor(video_path: str, calibration: CourtCalibration, anchor_frame: int,
                timestamps_ms: Mapping[int, float], direction: str, stop_frame: int,
                player_boxes: Mapping[int, tuple[int, int, int, int]] | None = None) -> FloorTrack:
    """Tracks from anchor_frame to stop_frame (inclusive) in `direction`.
    timestamps_ms: real per-frame timestamps (e.g. from LandmarkFrame.timing),
    so elapsed time is real time, not frame count x nominal fps.
    player_boxes: per-frame (x0, y0, x1, y1) regions to exclude from feature
    detection, e.g. the pose landmarks' bounding box plus a margin."""
    import cv2

    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")
    cap = cv2.VideoCapture(video_path)
    frames = _frames(cap, anchor_frame, stop_frame, direction)
    try:
        index, prev = next(frames)
    except StopIteration:
        cap.release()
        return FloorTrack(anchor_frame, direction, (), None, None)
    floor0 = _floor_mask(calibration, prev.shape)
    t0 = timestamps_ms[anchor_frame]
    transform = np.eye(3)
    steps = [TrackStep(anchor_frame, 0.0, transform.copy(), 0)]
    stopped = None
    min_inl = None
    for index, cur in frames:
        prev_index = steps[-1].frame_index
        mask = cv2.warpPerspective(floor0, transform, (prev.shape[1], prev.shape[0]))
        if player_boxes and prev_index in player_boxes:
            x0, y0, x1, y1 = player_boxes[prev_index]
            mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
        pts = cv2.goodFeaturesToTrack(prev, MAX_FEATURES, 0.01, 7, mask=mask)
        n_in, H = 0, None
        if pts is not None and len(pts) >= MIN_INLIERS:
            nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
            good = status.reshape(-1) == 1
            if good.sum() >= MIN_INLIERS:
                H, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, RANSAC_PX)
                n_in = 0 if inl is None else int(inl.sum())
        if H is None or n_in < MIN_INLIERS:
            stopped = index
            break
        transform = H @ transform
        min_inl = n_in if min_inl is None else min(min_inl, n_in)
        steps.append(TrackStep(index, abs(timestamps_ms[index] - t0), transform.copy(), n_in))
        prev = cur
    cap.release()
    return FloorTrack(anchor_frame, direction, tuple(steps), stopped, min_inl)


def player_boxes_from_frames(frames, margin_px: int = 40) -> dict[int, tuple[int, int, int, int]]:
    """Pose-landmark bounding box per frame, widened by margin (more above the
    head, where the racket is). Used to keep player features out of the
    floor fit."""
    boxes = {}
    for fr in frames:
        xs = [lm.position.x for lm in fr.pose_landmarks.values()]
        ys = [lm.position.y for lm in fr.pose_landmarks.values()]
        if xs:
            boxes[fr.timing.frame_index] = (int(min(xs)) - margin_px, int(min(ys)) - 3 * margin_px,
                                            int(max(xs)) + margin_px, int(max(ys)) + margin_px)
    return boxes


def apply(transform: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = transform @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])
