"""Ball-contact frame detection from racket-side wrist kinematics.

Two independent candidate rules over the same resultant-speed signal:

  - peak_speed_contact: the frame of maximum wrist speed.
  - deceleration_onset_contact: the frame of steepest deceleration (most
    negative d(speed)/dt) anywhere in the trajectory.

These usually land near each other but are computed independently -- one is
an extremum of speed itself, the other an extremum of its derivative, over
the whole trajectory rather than one anchored to the other's answer -- so
they can and do disagree when the post-peak deceleration is gradual rather
than sharp. See tools/eval_phases.py for how often they agree in practice
and which tracks labelled ground truth more closely; no threshold here has
been tuned against that data.

Multi-swing clips: segment_swing_windows() splits a clip's wrist-speed
trace into candidate swing windows first (sustained above-threshold
stretches separated by rest gaps); contact_candidates_in_windows() then
applies one of the two rules above independently inside each window, by
slicing the speed series rather than changing the rules themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Final

from engine.biomechanics.kinematics.derivatives import DerivativeConfig, DerivativeMethod
from engine.biomechanics.kinematics.velocity import VelocityCalculator
from engine.types.landmarks import LandmarkFrame, PoseLandmarkName
from engine.utils.geometry import magnitude

_DERIVATIVE_CONFIG = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)

# SWING_SPEED_THRESHOLD_FRACTION is still an untuned starting guess -- not fit
# against labels/. The other two were tuned against the 5 labelled clips in
# labels/ (2026-08-27); see tools/eval_phases.py for where they stand.
SWING_SPEED_THRESHOLD_FRACTION: Final[float] = 0.25  # of the clip's own peak speed, counts as "swinging"

# Kept at 15, not lowered: sample_backhand3's 3 merged windows need <=5 to
# split (measured: shortest below-threshold run between two real contacts is
# 5 frames, between labelled contacts 191 and 236). But sample_forehand1 has
# a genuine single swing (contact frame 600, window 558-609) with a 12-frame
# internal below-threshold dip -- going below 13 splits that real swing in
# two. Checked whether a separate, lower "how still" depth threshold could
# decouple the two clips instead of "how long": it can't -- backhand3's real
# rests measure 5.1-7.0% of clip peak speed at their deepest point, and
# forehand1 has genuine internal dips at 6.4% and 6.9% (one clip's real dip
# is literally deeper, 3.4%, than all four of the other clip's real rests).
# No single threshold, on duration or depth, separates them. Left at the
# value that's safe for forehand1 (and the other 4 clips); backhand3 stays
# under-segmented as a known, currently-unfixed limitation.
SWING_MIN_REST_GAP_FRAMES: Final[int] = 15  # consecutive below-threshold frames required to split two swings

# Raised from 5: every incidental (non-real) window across the 5 labelled
# clips is <=12 frames (worst: sample_forehand2's window at frames 191-202,
# 12 frames) except two windows (sample_backhand3's trailing 25-frame window,
# sample_forehand2's other 22-frame window) that are longer than some real
# swings elsewhere and can't be filtered by length without losing those --
# left as unfixed by this constant. Every real (labelled) window across all 5
# clips is >=19 frames (sample_forehand1's shortest). 13 sits in the clean gap
# between the two (>12, <19), with margin on both sides.
SWING_MIN_WINDOW_FRAMES: Final[int] = 13  # minimum active-window length to count as a real swing, not noise


@dataclass(frozen=True)
class WristSpeedSample:
    frame_index: int
    timestamp_ms: float
    speed: float | None  # resultant |velocity|; None when velocity couldn't be computed this frame


@dataclass(frozen=True)
class ContactCandidate:
    frame_index: int
    rule_name: str


@dataclass(frozen=True)
class SwingWindow:
    start_frame: int
    end_frame: int


def infer_racket_side(frames: tuple[LandmarkFrame, ...]) -> PoseLandmarkName:
    """PhaseDetector.detect() (the existing ABC contract) takes only
    LandmarkFrame tuples -- no handedness reaches it. Infer which wrist is
    the racket wrist from the data itself instead: the racket arm should
    reach a much higher peak speed than the off arm during a swing. Ties or
    all-invalid data default to the right wrist (an arbitrary but explicit
    choice, not a silent one)."""
    left_peak = _peak_speed(frames, PoseLandmarkName.LEFT_WRIST)
    right_peak = _peak_speed(frames, PoseLandmarkName.RIGHT_WRIST)
    return PoseLandmarkName.LEFT_WRIST if left_peak > right_peak else PoseLandmarkName.RIGHT_WRIST


def _peak_speed(frames: tuple[LandmarkFrame, ...], wrist: PoseLandmarkName) -> float:
    speeds = [s.speed for s in wrist_speed_series(frames, wrist) if s.speed is not None]
    return max(speeds, default=0.0)


def wrist_speed_series(frames: tuple[LandmarkFrame, ...], wrist: PoseLandmarkName) -> tuple[WristSpeedSample, ...]:
    position_samples = [
        (f.timing, f.pose_landmarks[wrist].position if wrist in f.pose_landmarks else None) for f in frames
    ]
    velocities = VelocityCalculator().compute(position_samples, _DERIVATIVE_CONFIG)
    return tuple(
        WristSpeedSample(
            frame_index=v.frame_index,
            timestamp_ms=v.timestamp_ms,
            speed=magnitude(v.velocity) if (v.is_valid and v.velocity is not None) else None,
        )
        for v in velocities
    )


def peak_speed_contact(speeds: tuple[WristSpeedSample, ...]) -> ContactCandidate | None:
    valid = [s for s in speeds if s.speed is not None]
    if not valid:
        return None
    best = max(valid, key=lambda s: s.speed)
    return ContactCandidate(frame_index=best.frame_index, rule_name="peak_speed")


def deceleration_onset_contact(speeds: tuple[WristSpeedSample, ...]) -> ContactCandidate | None:
    valid = [s for s in speeds if s.speed is not None]
    if len(valid) < 2:
        return None
    steepest_drop = 0.0
    steepest_frame: int | None = None
    for prev, cur in zip(valid, valid[1:]):
        dt_seconds = (cur.timestamp_ms - prev.timestamp_ms) / 1000.0
        if dt_seconds <= 0:
            continue
        d_speed_dt = (cur.speed - prev.speed) / dt_seconds  # negative == decelerating
        if d_speed_dt < steepest_drop:
            steepest_drop = d_speed_dt
            steepest_frame = cur.frame_index
    if steepest_frame is None:
        return None
    return ContactCandidate(frame_index=steepest_frame, rule_name="deceleration_onset")


CONTACT_RULES = {
    "peak_speed": peak_speed_contact,
    "deceleration_onset": deceleration_onset_contact,
}


def segment_swing_windows(
    speeds: tuple[WristSpeedSample, ...],
    speed_threshold_fraction: float = SWING_SPEED_THRESHOLD_FRACTION,
    min_rest_gap_frames: int = SWING_MIN_REST_GAP_FRAMES,
    min_window_frames: int = SWING_MIN_WINDOW_FRAMES,
) -> tuple[SwingWindow, ...]:
    """Splits a clip into candidate swing windows: stretches of frame
    indices where wrist speed sustains at or above `speed_threshold_fraction`
    of the clip's own peak speed, separated by rest gaps of at least
    `min_rest_gap_frames` consecutive below-threshold frames (a shorter gap
    is treated as jitter within one swing, not a real separation). Windows
    shorter than `min_window_frames` are dropped as noise. Operates on frame
    index, not list position, so a missing/invalid sample can't silently
    stitch two separate active stretches together."""
    valid = [s for s in speeds if s.speed is not None]
    if not valid:
        return ()
    peak = max(s.speed for s in valid)
    if peak <= 0:
        return ()
    threshold = speed_threshold_fraction * peak

    active_frames = sorted(s.frame_index for s in valid if s.speed >= threshold)
    if not active_frames:
        return ()

    runs: list[list[int]] = []
    for frame_index in active_frames:
        if runs and frame_index - runs[-1][-1] - 1 < min_rest_gap_frames:
            runs[-1].append(frame_index)
        else:
            runs.append([frame_index])

    return tuple(
        SwingWindow(start_frame=run[0], end_frame=run[-1])
        for run in runs
        if (run[-1] - run[0] + 1) >= min_window_frames
    )


def contact_candidates_in_windows(
    speeds: tuple[WristSpeedSample, ...],
    windows: tuple[SwingWindow, ...],
    rule_fn: Callable[[tuple[WristSpeedSample, ...]], ContactCandidate | None],
) -> tuple[ContactCandidate | None, ...]:
    """Applies an existing contact rule independently inside each window, by
    restricting the speed series to that window's frame range before calling
    it -- the rule functions themselves are unmodified. One result per
    window, positionally aligned with `windows`; None where the rule found
    nothing usable inside that particular window."""
    results = []
    for window in windows:
        windowed_speeds = tuple(s for s in speeds if window.start_frame <= s.frame_index <= window.end_frame)
        results.append(rule_fn(windowed_speeds))
    return tuple(results)
