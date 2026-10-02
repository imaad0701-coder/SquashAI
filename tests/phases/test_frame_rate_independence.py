"""Part 0 proof: the SAME real-time swing definition, sampled independently
at 30fps and 60fps, must produce real-time-equivalent detection under the
ms-derived constants (engine.phases.frame_timing) -- and did NOT under the
old raw-frame-count constants, which is the bug this fixes.

Motion is generated as piecewise-constant velocity in time (rest speed
outside the swing window, swing speed inside it), evaluated in closed form
at each sample's own timestamp and independently sampled at each fps --
not frame duplication (zero displacement, zero speed, tests nothing) and
not linear interpolation of an already-curved real trajectory (tried first;
found to systematically bias speed low at points of curvature via the
straight-chord-vs-true-arc effect, fabricating dips that don't exist in the
real signal -- see docs/STATUS.md/PR notes, 2026-08-29). Straight-line 1D
motion in time has no curvature to cut a corner on, so resampling it at a
different rate is exact, isolating the property actually under test."""

from __future__ import annotations

import unittest

from engine.phases.contact_detection import (
    CONTACT_RULES, contact_candidates_in_windows, infer_racket_side, segment_swing_windows, wrist_speed_series,
)
from engine.phases.frame_timing import ms_per_frame
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import FrameTiming


def _landmark(x: float) -> Landmark:
    return Landmark(position=Point3D(x=x, y=0.0, z=0.0), visibility=0.9, presence=0.9)


def _frames_at_fps(
    fps: float, duration_ms: float, swing_windows_ms: tuple[tuple[float, float], ...],
    rest_speed_per_ms: float = 0.02, swing_speed_per_ms: float = 3.0,
) -> tuple[LandmarkFrame, ...]:
    """Racket-side (right) wrist position generated as a closed-form integral
    of a piecewise-constant velocity in time (rest_speed_per_ms outside every
    (start_ms, end_ms) in swing_windows_ms, swing_speed_per_ms inside one),
    then sampled at `fps`. Same swing definition at any fps -> same position
    at any shared timestamp, exactly (no discretization in the position
    function itself, only in which timestamps get sampled)."""
    dt_ms = 1000.0 / fps
    n_frames = int(duration_ms / dt_ms) + 1

    def position_at(t_ms: float) -> float:
        x = 0.0
        last_t = 0.0
        # integrate the piecewise-constant velocity from 0 to t_ms
        breakpoints = sorted({0.0, t_ms, *[b for w in swing_windows_ms for b in w if b <= t_ms]})
        for a, b in zip(breakpoints, breakpoints[1:]):
            in_swing = any(start <= (a + b) / 2 <= end for start, end in swing_windows_ms)
            speed = swing_speed_per_ms if in_swing else rest_speed_per_ms
            x += speed * (b - a)
        return x

    frames = []
    for i in range(n_frames):
        t_ms = i * dt_ms
        timing = FrameTiming(frame_index=i, timestamp_ms=t_ms, delta_time_ms=dt_ms)
        pose_landmarks = {
            PoseLandmarkName.LEFT_WRIST: _landmark(x=t_ms * 0.0005),  # off-arm: negligible drift
            PoseLandmarkName.RIGHT_WRIST: _landmark(x=position_at(t_ms)),
            PoseLandmarkName.LEFT_SHOULDER: _landmark(x=-20.0),
            PoseLandmarkName.RIGHT_SHOULDER: _landmark(x=20.0),
        }
        frames.append(LandmarkFrame(timing=timing, pose_landmarks=pose_landmarks))
    return tuple(frames)


def _frames_with_triangular_peak(
    fps: float, duration_ms: float, peak_center_ms: float, half_width_ms: float,
    rest_speed_per_ms: float = 0.02, peak_speed_per_ms: float = 3.0,
) -> tuple[LandmarkFrame, ...]:
    """Like _frames_at_fps, but the swing's velocity ramps linearly up to a
    single unambiguous peak at peak_center_ms and back down (a real swing's
    accelerate-to-contact-then-decelerate shape), not a flat plateau -- a
    rectangular pulse has no single well-defined maximum, which is fine for
    testing window *duration* but degenerates peak_speed_contact's frame
    selection into a tie-break, unrelated to the frame-rate property under
    test. A plain (linear-ramp) triangle has the opposite problem for
    deceleration_onset_contact: its derivative is constant across the whole
    down-ramp, so "steepest deceleration" is equally degenerate there
    instead. Uses a raised-cosine (Hann-shaped) profile: smooth, single
    unambiguous velocity maximum at peak_center_ms (derivative zero there,
    so peak_speed_contact has one answer), and analytically a single
    steepest-deceleration point at peak_center_ms + half_width_ms/2 on the
    descending side (so deceleration_onset_contact has one answer too).
    Integrated at a fixed fine internal resolution (0.5ms), decoupled from
    the sampling fps, so both rates observe the exact same continuous
    position function -- only which timestamps get sampled differs."""
    import math

    dt_ms = 1000.0 / fps
    n_frames = int(duration_ms / dt_ms) + 1
    integration_step_ms = 0.5

    def velocity_at(t_ms: float) -> float:
        dist = abs(t_ms - peak_center_ms)
        if dist >= half_width_ms:
            return rest_speed_per_ms
        return rest_speed_per_ms + (peak_speed_per_ms - rest_speed_per_ms) * 0.5 * (
            1.0 + math.cos(math.pi * dist / half_width_ms)
        )

    # precompute position at fine resolution once, then sample it
    n_steps = int(duration_ms / integration_step_ms) + 1
    fine_positions = [0.0] * n_steps
    x = 0.0
    for i in range(1, n_steps):
        t_mid = (i - 0.5) * integration_step_ms
        x += velocity_at(t_mid) * integration_step_ms
        fine_positions[i] = x

    def position_at(t_ms: float) -> float:
        idx = min(n_steps - 1, round(t_ms / integration_step_ms))
        return fine_positions[idx]

    frames = []
    for i in range(n_frames):
        t_ms = i * dt_ms
        timing = FrameTiming(frame_index=i, timestamp_ms=t_ms, delta_time_ms=dt_ms)
        pose_landmarks = {
            PoseLandmarkName.LEFT_WRIST: _landmark(x=t_ms * 0.0005),
            PoseLandmarkName.RIGHT_WRIST: _landmark(x=position_at(t_ms)),
            PoseLandmarkName.LEFT_SHOULDER: _landmark(x=-20.0),
            PoseLandmarkName.RIGHT_SHOULDER: _landmark(x=20.0),
        }
        frames.append(LandmarkFrame(timing=timing, pose_landmarks=pose_landmarks))
    return tuple(frames)


class FrameRateIndependenceTests(unittest.TestCase):
    """One real-time swing definition (a single 1200ms-long swing inside an
    8000ms clip, with real rest before/after), sampled at 30fps and 60fps."""

    SWING_MS = ((3000.0, 4200.0),)  # a single swing, 1200ms long, well inside SWING_MIN_WINDOW_MS's safe range
    DURATION_MS = 8000.0

    def _windows_ms(self, fps: float) -> list[tuple[float, float, float]]:
        frames = _frames_at_fps(fps, self.DURATION_MS, self.SWING_MS)
        wrist = infer_racket_side(frames)
        speeds = wrist_speed_series(frames, wrist)
        rate = ms_per_frame(speeds)
        windows = segment_swing_windows(speeds)  # ms-derived defaults -- the fix under test
        return [(w.start_frame * rate, w.end_frame * rate, (w.end_frame - w.start_frame + 1) * rate) for w in windows]

    def test_window_count_matches_across_frame_rates(self) -> None:
        windows_30 = self._windows_ms(30.0)
        windows_60 = self._windows_ms(60.0)
        self.assertEqual(len(windows_30), 1, windows_30)
        self.assertEqual(len(windows_60), 1, windows_60)

    def test_window_boundaries_agree_in_real_time_within_one_frame(self) -> None:
        (s30, e30, _), = self._windows_ms(30.0)
        (s60, e60, _), = self._windows_ms(60.0)
        # tolerance = one 30fps frame (the coarser of the two rates) -- the
        # real discretization floor, not an arbitrarily loose bound
        tolerance_ms = 1000.0 / 30.0
        self.assertLessEqual(abs(s30 - s60), tolerance_ms, f"start: {s30}ms vs {s60}ms")
        self.assertLessEqual(abs(e30 - e60), tolerance_ms, f"end: {e30}ms vs {e60}ms")

    def test_contact_frame_agrees_in_real_time_across_frame_rates(self) -> None:
        peak_center_ms, half_width_ms = 3600.0, 600.0  # matches SWING_MS=(3000,4200)'s midpoint/span
        for rule_name, rule_fn in CONTACT_RULES.items():
            with self.subTest(rule=rule_name):
                frames_30 = _frames_with_triangular_peak(30.0, self.DURATION_MS, peak_center_ms, half_width_ms)
                frames_60 = _frames_with_triangular_peak(60.0, self.DURATION_MS, peak_center_ms, half_width_ms)

                def contact_ms(frames):
                    wrist = infer_racket_side(frames)
                    speeds = wrist_speed_series(frames, wrist)
                    rate = ms_per_frame(speeds)
                    windows = segment_swing_windows(speeds)
                    contacts = contact_candidates_in_windows(speeds, windows, rule_fn)
                    self.assertEqual(len(contacts), 1)
                    self.assertIsNotNone(contacts[0])
                    return contacts[0].frame_index * rate

                tolerance_ms = 1000.0 / 30.0
                self.assertLessEqual(abs(contact_ms(frames_30) - contact_ms(frames_60)), tolerance_ms)

    def test_a_swing_shorter_than_min_window_ms_is_dropped_at_both_rates(self) -> None:
        # 150ms swing -- well under SWING_MIN_WINDOW_MS (250ms) -- must be
        # filtered as noise identically at both rates, not just at whichever
        # one happens to round it to a keepable frame count.
        short_swing = ((3000.0, 3150.0),)
        for fps in (30.0, 60.0):
            with self.subTest(fps=fps):
                frames = _frames_at_fps(fps, self.DURATION_MS, short_swing)
                wrist = infer_racket_side(frames)
                speeds = wrist_speed_series(frames, wrist)
                windows = segment_swing_windows(speeds)
                self.assertEqual(windows, (), f"a {150}ms swing should be dropped as noise at {fps}fps too")

    def test_old_raw_frame_count_threshold_was_NOT_rate_independent(self) -> None:
        # Documents the bug this whole fix addresses, so it can't silently
        # regress back: a fixed 13-frame min_window_frames means a real-time
        # tolerance that shrinks as fps rises. A 300ms swing is comfortably
        # over 250ms (kept at any rate under the fix) but under the OLD
        # 13-frames-at-30fps=433ms tolerance -- so the raw-13 threshold drops
        # it at 30fps, while at 60fps 13 frames is only 217ms, so it's kept.
        # Same swing, same real duration, opposite outcome -- exactly the
        # frame-rate dependence this module exists to remove.
        boundary_swing = ((3000.0, 3300.0),)  # 300ms
        frames_30 = _frames_at_fps(30.0, self.DURATION_MS, boundary_swing)
        frames_60 = _frames_at_fps(60.0, self.DURATION_MS, boundary_swing)
        speeds_30 = wrist_speed_series(frames_30, infer_racket_side(frames_30))
        speeds_60 = wrist_speed_series(frames_60, infer_racket_side(frames_60))

        windows_30_old = segment_swing_windows(speeds_30, min_rest_gap_frames=15, min_window_frames=13)
        windows_60_old = segment_swing_windows(speeds_60, min_rest_gap_frames=15, min_window_frames=13)
        self.assertEqual(windows_30_old, (), "old threshold should drop a 300ms swing at 30fps (13f=433ms)")
        self.assertEqual(len(windows_60_old), 1, "old threshold should KEEP the same 300ms swing at 60fps (13f=217ms)")

        # the fix: both rates now agree (both keep it, since 300ms > 250ms)
        windows_30_new = segment_swing_windows(speeds_30)
        windows_60_new = segment_swing_windows(speeds_60)
        self.assertEqual(len(windows_30_new), 1)
        self.assertEqual(len(windows_60_new), 1)


if __name__ == "__main__":
    unittest.main()
