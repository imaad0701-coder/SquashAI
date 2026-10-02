"""Tests for engine.phases.contact_detection."""

from __future__ import annotations

import unittest

from engine.phases.contact_detection import (
    ContactCandidate,
    SwingWindow,
    WristSpeedSample,
    contact_candidates_in_windows,
    deceleration_onset_contact,
    infer_racket_side,
    peak_speed_contact,
    segment_swing_windows,
    wrist_speed_series,
)
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import FrameTiming

_STILL = Landmark(position=Point3D(x=0.0, y=0.0, z=0.0), visibility=0.9, presence=0.9)


def _sample(frame_index: int, timestamp_ms: float, speed: float | None) -> WristSpeedSample:
    return WristSpeedSample(frame_index=frame_index, timestamp_ms=timestamp_ms, speed=speed)


def _frame_with_wrists(index: int, left_x: float | None, right_x: float | None) -> LandmarkFrame:
    timing = FrameTiming(frame_index=index, timestamp_ms=index * (1000.0 / 30.0), delta_time_ms=1000.0 / 30.0)
    pose_landmarks = {}
    if left_x is not None:
        pose_landmarks[PoseLandmarkName.LEFT_WRIST] = Landmark(
            position=Point3D(x=left_x, y=0.0, z=0.0), visibility=0.9, presence=0.9
        )
    if right_x is not None:
        pose_landmarks[PoseLandmarkName.RIGHT_WRIST] = Landmark(
            position=Point3D(x=right_x, y=0.0, z=0.0), visibility=0.9, presence=0.9
        )
    return LandmarkFrame(timing=timing, pose_landmarks=pose_landmarks)


class PeakSpeedContactTests(unittest.TestCase):
    def test_returns_the_frame_with_maximum_speed(self) -> None:
        speeds = (_sample(0, 0.0, 1.0), _sample(1, 33.3, 9.0), _sample(2, 66.7, 4.0))
        result = peak_speed_contact(speeds)
        self.assertEqual(result, ContactCandidate(frame_index=1, rule_name="peak_speed"))

    def test_ignores_invalid_none_speed_samples(self) -> None:
        speeds = (_sample(0, 0.0, None), _sample(1, 33.3, 5.0), _sample(2, 66.7, None))
        result = peak_speed_contact(speeds)
        self.assertEqual(result.frame_index, 1)

    def test_none_when_every_sample_is_invalid(self) -> None:
        speeds = (_sample(0, 0.0, None), _sample(1, 33.3, None))
        self.assertIsNone(peak_speed_contact(speeds))

    def test_none_for_empty_input(self) -> None:
        self.assertIsNone(peak_speed_contact(()))


class DecelerationOnsetContactTests(unittest.TestCase):
    def test_returns_the_frame_of_steepest_speed_drop(self) -> None:
        # speed rises gently, then crashes sharply between frame 2 and 3.
        speeds = (
            _sample(0, 0.0, 1.0),
            _sample(1, 33.3, 5.0),
            _sample(2, 66.7, 9.0),
            _sample(3, 100.0, 1.0),  # steep drop lands here
            _sample(4, 133.3, 0.5),  # gentler drop
        )
        result = deceleration_onset_contact(speeds)
        self.assertEqual(result, ContactCandidate(frame_index=3, rule_name="deceleration_onset"))

    def test_none_when_speed_never_decreases(self) -> None:
        speeds = (_sample(0, 0.0, 1.0), _sample(1, 33.3, 2.0), _sample(2, 66.7, 3.0))
        self.assertIsNone(deceleration_onset_contact(speeds))

    def test_none_with_fewer_than_two_valid_samples(self) -> None:
        self.assertIsNone(deceleration_onset_contact((_sample(0, 0.0, 1.0),)))
        self.assertIsNone(deceleration_onset_contact(()))

    def test_skips_non_positive_time_gaps_rather_than_raising(self) -> None:
        speeds = (_sample(0, 0.0, 5.0), _sample(1, 0.0, 1.0), _sample(2, 33.3, 0.5))
        result = deceleration_onset_contact(speeds)
        self.assertIsNotNone(result)


class InferRacketSideTests(unittest.TestCase):
    def test_picks_the_wrist_with_higher_peak_speed(self) -> None:
        # Right wrist swings from 0 to 500px over a few frames; left wrist barely moves.
        frames = tuple(
            _frame_with_wrists(i, left_x=float(i), right_x=float(i) * 100.0) for i in range(5)
        )
        self.assertEqual(infer_racket_side(frames), PoseLandmarkName.RIGHT_WRIST)

    def test_picks_left_wrist_when_it_moves_more(self) -> None:
        frames = tuple(
            _frame_with_wrists(i, left_x=float(i) * 100.0, right_x=float(i)) for i in range(5)
        )
        self.assertEqual(infer_racket_side(frames), PoseLandmarkName.LEFT_WRIST)

    def test_defaults_to_right_wrist_when_neither_moves(self) -> None:
        frames = tuple(_frame_with_wrists(i, left_x=0.0, right_x=0.0) for i in range(5))
        self.assertEqual(infer_racket_side(frames), PoseLandmarkName.RIGHT_WRIST)


class WristSpeedSeriesTests(unittest.TestCase):
    def test_speed_is_none_when_wrist_missing_from_every_frame(self) -> None:
        frames = tuple(_frame_with_wrists(i, left_x=None, right_x=float(i)) for i in range(3))
        series = wrist_speed_series(frames, PoseLandmarkName.LEFT_WRIST)
        self.assertTrue(all(s.speed is None for s in series))

    def test_moving_wrist_yields_a_higher_peak_than_a_still_one(self) -> None:
        moving = tuple(_frame_with_wrists(i, left_x=None, right_x=float(i) * 50.0) for i in range(6))
        still = tuple(_frame_with_wrists(i, left_x=None, right_x=1.0) for i in range(6))
        moving_speeds = [s.speed for s in wrist_speed_series(moving, PoseLandmarkName.RIGHT_WRIST) if s.speed]
        still_speeds = [s.speed for s in wrist_speed_series(still, PoseLandmarkName.RIGHT_WRIST) if s.speed]
        self.assertTrue(moving_speeds)
        self.assertGreater(max(moving_speeds), max(still_speeds, default=0.0))

    def test_frame_indices_and_timestamps_match_input_frames(self) -> None:
        frames = tuple(_frame_with_wrists(i, left_x=None, right_x=float(i)) for i in range(4))
        series = wrist_speed_series(frames, PoseLandmarkName.RIGHT_WRIST)
        self.assertEqual([s.frame_index for s in series], [0, 1, 2, 3])
        self.assertEqual([s.timestamp_ms for s in series], [f.timing.timestamp_ms for f in frames])


def _rect_speeds(n: int, active_ranges: tuple[tuple[int, int], ...], rest: float = 1.0, active: float = 100.0):
    return tuple(
        _sample(i, i * 33.333, active if any(lo <= i <= hi for lo, hi in active_ranges) else rest)
        for i in range(n)
    )


class SegmentSwingWindowsTests(unittest.TestCase):
    def test_single_sustained_active_stretch_is_one_window(self) -> None:
        speeds = _rect_speeds(40, ((10, 20),))
        windows = segment_swing_windows(speeds, speed_threshold_fraction=0.25, min_rest_gap_frames=5, min_window_frames=3)
        self.assertEqual(windows, (SwingWindow(start_frame=10, end_frame=20),))

    def test_two_stretches_separated_by_a_long_enough_gap_are_two_windows(self) -> None:
        speeds = _rect_speeds(60, ((5, 15), (40, 50)))
        windows = segment_swing_windows(speeds, speed_threshold_fraction=0.25, min_rest_gap_frames=5, min_window_frames=3)
        self.assertEqual(windows, (SwingWindow(5, 15), SwingWindow(40, 50)))

    def test_gap_shorter_than_min_rest_gap_merges_into_one_window(self) -> None:
        # Active at 5-15, rest 16-18 (3 frames -- shorter than min_rest_gap_frames=5), active again 19-25.
        speeds = _rect_speeds(30, ((5, 15), (19, 25)))
        windows = segment_swing_windows(speeds, speed_threshold_fraction=0.25, min_rest_gap_frames=5, min_window_frames=3)
        self.assertEqual(windows, (SwingWindow(start_frame=5, end_frame=25),))

    def test_window_shorter_than_min_window_frames_is_dropped_as_noise(self) -> None:
        speeds = _rect_speeds(30, ((10, 11),))  # 2 frames, below min_window_frames=3
        windows = segment_swing_windows(speeds, speed_threshold_fraction=0.25, min_rest_gap_frames=5, min_window_frames=3)
        self.assertEqual(windows, ())

    def test_zero_speed_throughout_returns_no_windows(self) -> None:
        speeds = tuple(_sample(i, i * 33.333, 0.0) for i in range(20))
        self.assertEqual(segment_swing_windows(speeds), ())

    def test_empty_input_returns_no_windows(self) -> None:
        self.assertEqual(segment_swing_windows(()), ())


class ContactCandidatesInWindowsTests(unittest.TestCase):
    def test_one_candidate_per_window_using_each_windows_own_extremum(self) -> None:
        # Window A peaks at frame 12, window B peaks at frame 42 -- a global
        # search would still find both extrema, but per-window search must
        # attribute each to the correct window, not both to the taller one.
        speeds = tuple(
            _sample(i, i * 33.333, 50.0 if i == 12 else (80.0 if i == 42 else 1.0)) for i in range(60)
        )
        windows = (SwingWindow(5, 20), SwingWindow(35, 50))
        results = contact_candidates_in_windows(speeds, windows, peak_speed_contact)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].frame_index, 12)
        self.assertEqual(results[1].frame_index, 42)

    def test_none_for_a_window_with_no_valid_speed_data(self) -> None:
        speeds = tuple(_sample(i, i * 33.333, None) for i in range(20))
        windows = (SwingWindow(5, 15),)
        results = contact_candidates_in_windows(speeds, windows, peak_speed_contact)
        self.assertEqual(results, (None,))

    def test_empty_windows_returns_empty_tuple(self) -> None:
        speeds = _rect_speeds(20, ((5, 10),))
        self.assertEqual(contact_candidates_in_windows(speeds, (), peak_speed_contact), ())


if __name__ == "__main__":
    unittest.main()
