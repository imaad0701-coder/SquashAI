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
from engine.phases.frame_timing import ms_per_frame, ms_to_frames
from engine.types.landmarks import LandmarkFrame, PoseLandmarkName
from engine.utils.geometry import magnitude

_DERIVATIVE_CONFIG = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)

# SWING_SPEED_THRESHOLD_FRACTION is still an untuned starting guess -- not fit
# against labels/. The other two are expressed in real time (milliseconds),
# not frame counts (2026-08-29): the corpus is mixed frame rate --
# sample_forehand1.mp4 measures 59.895fps, the other 5 clips 30.000fps -- and
# a frame count means a different real-time tolerance depending on which one
# a clip happens to be. segment_swing_windows() converts these to a
# frame-equivalent per clip via engine.phases.frame_timing, using that
# clip's own measured rate, not an assumed 30fps. See
# docs/STATUS.md's engine/phases known limitations for why duration alone
# (in *either* unit) still can't cleanly separate real swings from
# incidental motion -- these are the values chosen despite that, not a claim
# the underlying problem is solved.
SWING_SPEED_THRESHOLD_FRACTION: Final[float] = 0.25  # of the clip's own peak speed, counts as "swinging"

# 500ms = 15 frames at 30fps -- the exact value tuned 2026-08-27. Kept, not
# lowered: sample_backhand3's 3 merged windows need <=~167ms to split
# (measured: shortest below-threshold run between two real contacts is 5
# frames = 167ms at 30fps). But sample_forehand1 has a genuine single swing
# with an internal below-threshold dip measuring 12 frames = 401ms at its
# own 59.895fps rate -- going below that splits that real swing in two.
# Checked whether a separate, lower "how still" depth threshold could
# decouple the two clips instead of "how long": it can't -- backhand3's real
# rests measure 5.1-7.0% of clip peak speed at their deepest point, and
# forehand1 has genuine internal dips at 6.4% and 6.9% (one clip's real dip
# is literally deeper, 3.4%, than all four of the other clip's real rests).
# No single threshold, on duration or depth, separates them, in frames or in
# ms. Left at the value that's safe for forehand1 (and the other 4 clips);
# backhand3 stays under-segmented as a known, currently-unfixed limitation.
SWING_MIN_REST_GAP_MS: Final[float] = 500.0  # real time required to split two swings

# Re-tuned 2026-08-29 from 433ms (the exact 30fps-equivalent of the prior
# 13-frame value) down to 250ms, after the frame-rate-independence fix
# exposed that 433ms would have dropped 5 of sample_forehand1's 7 real
# swings (19f/317ms-24f/401ms at its own 59.895fps rate) -- the frame-based
# value only ever worked because it was tuned on a fixed-rate corpus where
# forehand1's swings happened to be long enough in frames, not because
# duration cleanly separates real swings from incidental motion. It doesn't,
# in either unit -- see docs/STATUS.md's engine/phases known limitations.
# 250ms preserves every real window in the labelled corpus but re-admits a
# small number of incidental windows that 433ms/13-frames used to filter
# (quantified in the Part 0 regression report, 2026-08-29) -- a real,
# accepted cost of frame-rate independence, not a free improvement.
SWING_MIN_WINDOW_MS: Final[float] = 250.0  # minimum active-window duration to count as a real swing, not noise


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
    min_rest_gap_frames: int | None = None,
    min_window_frames: int | None = None,
) -> tuple[SwingWindow, ...]:
    """Splits a clip into candidate swing windows: stretches of frame
    indices where wrist speed sustains at or above `speed_threshold_fraction`
    of the clip's own peak speed, separated by rest gaps of at least
    `min_rest_gap_frames` consecutive below-threshold frames (a shorter gap
    is treated as jitter within one swing, not a real separation). Windows
    shorter than `min_window_frames` are dropped as noise. Operates on frame
    index, not list position, so a missing/invalid sample can't silently
    stitch two separate active stretches together.

    `min_rest_gap_frames`/`min_window_frames` left as None (the normal case)
    derive from SWING_MIN_REST_GAP_MS/SWING_MIN_WINDOW_MS and this clip's own
    measured frame rate (engine.phases.frame_timing), not an assumed 30fps --
    pass an explicit int to override that (existing tests that probe specific
    frame-count edge cases do, and keep working unchanged)."""
    valid = [s for s in speeds if s.speed is not None]
    if not valid:
        return ()
    peak = max(s.speed for s in valid)
    if peak <= 0:
        return ()
    threshold = speed_threshold_fraction * peak

    if min_rest_gap_frames is None or min_window_frames is None:
        rate = ms_per_frame(speeds)
        if min_rest_gap_frames is None:
            min_rest_gap_frames = ms_to_frames(SWING_MIN_REST_GAP_MS, rate)
        if min_window_frames is None:
            min_window_frames = ms_to_frames(SWING_MIN_WINDOW_MS, rate)

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
