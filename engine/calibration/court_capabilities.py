"""What court-coordinate output a clip can honestly support, given its
calibration and camera motion. Two tiers:

1. Per-swing positions (foot position at a swing's contact frame). Needs only
   a short stretch of the clip, so on a moving camera the calibration is
   carried from its anchor frame to the contact by floor tracking
   (engine.calibration.floor_tracking), forwards or backwards, but only
   within SHORT_HORIZON_MS of the anchor. Every result carries the real
   elapsed time and the implied tracking error -- never presented as exact.
   On a static camera the calibration applies directly.

2. Whole-clip output (movement trail; later heatmaps). Needs continuous court
   coordinates for every frame, which tracking can't deliver yet: its error
   keeps growing (~0.05-0.1% of the frame diagonal per second) and drift
   correction doesn't exist. So this tier requires the camera-motion check to
   classify the camera as STATIC. On a moving or unmeasurable camera it is
   UNAVAILABLE, with the reason -- not degraded silently.

Evidence for the numbers below: docs/evidence/camera_motion/ (ground truth
placed independently at 30 points over two clips, both directions).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Final, Sequence

import numpy as np

from engine.biomechanics.posture.angle_calculator import MIN_LANDMARK_PRESENCE, MIN_LANDMARK_VISIBILITY
from engine.calibration.camera_motion import CameraMotionStatus
from engine.calibration.court_homography import CourtCalibration, FootCourtPosition, foot_court_positions
from engine.calibration.floor_tracking import FloorTrack, apply, track_floor
from engine.types.landmarks import LandmarkFrame, PoseLandmarkName

# PROVISIONAL (2 clips, 2026-10-02): tracked error stayed within
# TRACKING_ERROR_BASE_PCT + TRACKING_ERROR_PER_S_PCT * seconds (of the frame
# diagonal) at every ground-truth point up to 5 s from the anchor, forwards
# (sample_forehand1) and backwards (sample_backhand3) alike; past ~5 s it
# exceeds the calibration's own point error and keeps growing.
SHORT_HORIZON_MS: Final[float] = 5000.0
TRACKING_ERROR_BASE_PCT: Final[float] = 0.3
TRACKING_ERROR_PER_S_PCT: Final[float] = 0.05


class Availability(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class PositionMethod(Enum):
    STATIC_CAMERA = "static_camera"  # calibration applied directly
    TRACKED = "tracked"  # calibration carried from the anchor by floor tracking


@dataclass(frozen=True)
class ContactCourtPosition:
    swing_index: int
    contact_frame: int | None
    availability: Availability
    reason: str | None  # set when UNAVAILABLE
    method: PositionMethod | None
    direction: str | None  # "forward" / "backward" from the anchor (TRACKED only)
    elapsed_ms: float | None  # real time between the anchor and the contact frame
    left_foot_m: tuple[float, float] | None
    right_foot_m: tuple[float, float] | None
    extrapolated: bool  # outside the calibration's clicked region
    implied_tracking_error_px: float | None  # TRACKED only; provisional envelope, see module docstring
    implied_tracking_error_m: float | None  # the pixel bound converted at the feet's location


@dataclass(frozen=True)
class MovementTrail:
    availability: Availability
    reason: str | None
    positions: tuple[FootCourtPosition, ...]


def implied_tracking_error_px(elapsed_ms: float, frame_diagonal_px: float) -> float:
    return frame_diagonal_px * (TRACKING_ERROR_BASE_PCT + TRACKING_ERROR_PER_S_PCT * elapsed_ms / 1000.0) / 100.0


def _feet_px(frame: LandmarkFrame) -> list[tuple[float, float] | None]:
    out = []
    for name in (PoseLandmarkName.LEFT_FOOT_INDEX, PoseLandmarkName.RIGHT_FOOT_INDEX):
        lm = frame.pose_landmarks.get(name)
        ok = lm is not None and lm.visibility >= MIN_LANDMARK_VISIBILITY and lm.presence >= MIN_LANDMARK_PRESENCE
        out.append((lm.position.x, lm.position.y) if ok else None)
    return out


def _metres_for_pixels(cal: CourtCalibration, px: tuple[float, float], radius_px: float) -> float | None:
    """Largest court-space displacement caused by moving `px` by radius_px
    in any of 8 directions -- the pixel error bound expressed in metres at
    that spot (perspective makes it larger further from the camera)."""
    centre = cal.homography.pixel_to_court(*px)
    if centre is None:
        return None
    worst = 0.0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707)):
        moved = cal.homography.pixel_to_court(px[0] + dx * radius_px, px[1] + dy * radius_px)
        if moved is None:
            return None
        worst = max(worst, ((moved[0] - centre[0]) ** 2 + (moved[1] - centre[1]) ** 2) ** 0.5)
    return worst


def _unavailable(i: int, contact: int | None, reason: str, elapsed: float | None = None) -> ContactCourtPosition:
    return ContactCourtPosition(i, contact, Availability.UNAVAILABLE, reason, None, None, elapsed, None, None, False, None, None)


def contact_court_positions(
    video_path: str,
    frames: Sequence[LandmarkFrame],
    contact_frames: Sequence[int | None],
    calibration: CourtCalibration,
    anchor_frame: int,
    camera: CameraMotionStatus,
    tracker: Callable[..., FloorTrack] = track_floor,
    player_boxes: dict | None = None,
) -> list[ContactCourtPosition]:
    """Foot positions (court metres) at each swing's contact frame.
    contact_frames: one entry per swing, e.g. [s.contact_frame for s in AnalysisResult.swings]."""
    by_index = {f.timing.frame_index: f for f in frames}
    ts = {f.timing.frame_index: f.timing.timestamp_ms for f in frames}
    w, h = calibration.resolution
    diag = (w * w + h * h) ** 0.5
    t0 = ts.get(anchor_frame)
    if t0 is None:
        return [_unavailable(i, c, f"calibration anchor frame {anchor_frame} is not in this clip's frames")
                for i, c in enumerate(contact_frames)]

    # Track once per direction, only as far as the furthest contact inside the horizon.
    tracks: dict[str, FloorTrack] = {}
    if camera is not CameraMotionStatus.STATIC:
        for direction in ("forward", "backward"):
            targets = [c for c in contact_frames if c is not None and c in ts
                       and (c > anchor_frame if direction == "forward" else c < anchor_frame)
                       and abs(ts[c] - t0) <= SHORT_HORIZON_MS]
            if targets:
                stop = max(targets) if direction == "forward" else min(targets)
                tracks[direction] = tracker(video_path, calibration, anchor_frame, ts, direction, stop, player_boxes)

    out = []
    for i, contact in enumerate(contact_frames):
        if contact is None or contact not in by_index:
            out.append(_unavailable(i, contact, "no detected contact frame for this swing"))
            continue
        elapsed = abs(ts[contact] - t0)
        feet = _feet_px(by_index[contact])
        if camera is CameraMotionStatus.STATIC:
            mapped = [None if p is None else calibration.homography.pixel_to_court(*p) for p in feet]
            method, direction, err_px, to_anchor = PositionMethod.STATIC_CAMERA, None, None, None
        else:
            if elapsed > SHORT_HORIZON_MS:
                out.append(_unavailable(
                    i, contact,
                    f"contact is {elapsed / 1000:.1f} s from the calibration frame; on a moving camera court positions are only "
                    f"carried {SHORT_HORIZON_MS / 1000:.0f} s by tracking (error beyond that is unmeasured and growing)",
                    elapsed))
                continue
            direction = "forward" if contact > anchor_frame else "backward"
            if contact == anchor_frame:
                frame_to_anchor = None
            else:
                track = tracks.get(direction)
                step = None if track is None else track.step_for(contact)
                if step is None:
                    gap = None if track is None else track.stopped_at_gap
                    out.append(_unavailable(i, contact, f"floor tracking lost lock at frame {gap} before reaching this contact"
                                            if gap is not None else "floor tracking did not reach this contact", elapsed))
                    continue
                frame_to_anchor = np.linalg.inv(step.anchor_to_frame)
            to_anchor = (lambda p: p) if frame_to_anchor is None else (lambda p, m=frame_to_anchor: apply(m, *p))
            mapped = [None if p is None else calibration.homography.pixel_to_court(*to_anchor(p)) for p in feet]
            method, err_px = PositionMethod.TRACKED, implied_tracking_error_px(elapsed, diag)
        err_m = None
        if err_px is not None:
            ms = [_metres_for_pixels(calibration, to_anchor(p), err_px) for p in feet if p is not None]
            ms = [m for m in ms if m is not None]
            err_m = max(ms) if ms else None
        left, right = mapped
        extrapolated = any(p is not None and calibration.homography.is_extrapolated(p) for p in mapped)
        out.append(ContactCourtPosition(i, contact, Availability.AVAILABLE, None, method, direction, elapsed,
                                        left, right, extrapolated, err_px, err_m))
    return out


def movement_trail(frames: Sequence[LandmarkFrame], calibration: CourtCalibration,
                   camera: CameraMotionStatus, camera_drift_pct: float | None = None) -> MovementTrail:
    """Foot positions for every frame. Only on a STATIC camera: continuous
    tracking with drift correction is future work (docs/roadmap/camera-motion-tracking.md)."""
    if camera is not CameraMotionStatus.STATIC:
        measured = "" if camera_drift_pct is None else f" (measured drift {camera_drift_pct:.1f}% of the frame diagonal)"
        why = ("the camera moves during this clip" + measured if camera is CameraMotionStatus.MOVING
               else "the camera's motion could not be measured for the whole clip")
        return MovementTrail(
            Availability.UNAVAILABLE,
            f"A movement trail needs court positions for every frame, which needs a static camera; {why}. "
            "Continuous tracking through camera motion is not built yet. Per-swing positions near the calibration "
            "frame may still be available.",
            (),
        )
    return MovementTrail(Availability.AVAILABLE, None, tuple(foot_court_positions(frames, calibration)))
