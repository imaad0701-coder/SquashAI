"""Tests for MovingAverageSmoother."""

from __future__ import annotations

import unittest

from engine.preprocessing.smoothing import MovingAverageSmoother, SmoothingConfig, SmoothingMethod
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import FrameTiming


def _frame(
    index: int,
    x: float | None,
    delta_time_ms: float = 100.0,
    visibility: float = 0.9,
    presence: float = 0.8,
) -> LandmarkFrame:
    pose_landmarks = {}
    if x is not None:
        pose_landmarks[PoseLandmarkName.NOSE] = Landmark(
            position=Point3D(x=x, y=0.0, z=0.0), visibility=visibility, presence=presence
        )
    timing = FrameTiming(
        frame_index=index,
        timestamp_ms=index * 100.0,
        delta_time_ms=0.0 if index == 0 else delta_time_ms,
    )
    return LandmarkFrame(timing=timing, pose_landmarks=pose_landmarks)


class MovingAverageSmootherTests(unittest.TestCase):
    def test_averages_trailing_window_weighted_by_real_time_gaps(self) -> None:
        # Constant 100ms spacing, except frame0 has delta_time_ms=0.0 (it is
        # the first frame in the sequence, so it has no known prior gap) —
        # that near-zero weight is the correct, honest consequence of not
        # assuming constant spacing: we don't know what duration frame0's
        # reading represents, only what frame1/frame2's real gaps were.
        frames = (_frame(0, 0.0), _frame(1, 3.0), _frame(2, 6.0))
        config = SmoothingConfig(method=SmoothingMethod.MOVING_AVERAGE, window_size=2)

        result = MovingAverageSmoother().smooth(frames, config)

        self.assertAlmostEqual(result[0].pose_landmarks[PoseLandmarkName.NOSE].position.x, 0.0)
        self.assertAlmostEqual(result[1].pose_landmarks[PoseLandmarkName.NOSE].position.x, 3.0, places=3)
        self.assertAlmostEqual(result[2].pose_landmarks[PoseLandmarkName.NOSE].position.x, 4.5)

    def test_irregular_spacing_weights_the_longer_gap_more_heavily(self) -> None:
        # frame1 arrives only 10ms after frame0, but frame2 arrives 1000ms
        # after frame1. A naive (unweighted) average of [0, 100, 200] would
        # be 100.0; a correct time-weighted average leans hard toward 200,
        # since frame2's reading was in effect for a much longer real span.
        frames = (
            _frame(0, 0.0, delta_time_ms=0.0),
            _frame(1, 100.0, delta_time_ms=10.0),
            _frame(2, 200.0, delta_time_ms=1000.0),
        )
        config = SmoothingConfig(method=SmoothingMethod.MOVING_AVERAGE, window_size=3)

        result = MovingAverageSmoother().smooth(frames, config)

        smoothed_x = result[2].pose_landmarks[PoseLandmarkName.NOSE].position.x
        naive_average = (0.0 + 100.0 + 200.0) / 3
        self.assertAlmostEqual(smoothed_x, 201000 / 1010, delta=0.01)
        self.assertGreater(smoothed_x, naive_average + 50)

    def test_window_larger_than_sequence_averages_all_available(self) -> None:
        frames = (_frame(0, 0.0), _frame(1, 10.0))
        config = SmoothingConfig(method=SmoothingMethod.MOVING_AVERAGE, window_size=100)

        result = MovingAverageSmoother().smooth(frames, config)

        self.assertAlmostEqual(result[0].pose_landmarks[PoseLandmarkName.NOSE].position.x, 0.0)
        self.assertAlmostEqual(result[1].pose_landmarks[PoseLandmarkName.NOSE].position.x, 10.0, places=3)

    def test_skips_frames_missing_the_landmark_within_the_window(self) -> None:
        frames = (_frame(0, 0.0), _frame(1, None), _frame(2, 10.0))
        config = SmoothingConfig(method=SmoothingMethod.MOVING_AVERAGE, window_size=2)

        result = MovingAverageSmoother().smooth(frames, config)

        self.assertNotIn(PoseLandmarkName.NOSE, result[1].pose_landmarks)
        self.assertAlmostEqual(result[2].pose_landmarks[PoseLandmarkName.NOSE].position.x, 10.0)

    def test_visibility_and_presence_pass_through_unaveraged(self) -> None:
        frames = (_frame(0, 0.0, visibility=0.4, presence=0.3), _frame(1, 2.0, visibility=0.9, presence=0.7))
        config = SmoothingConfig(method=SmoothingMethod.MOVING_AVERAGE, window_size=2)

        result = MovingAverageSmoother().smooth(frames, config)

        self.assertEqual(result[1].pose_landmarks[PoseLandmarkName.NOSE].visibility, 0.9)
        self.assertEqual(result[1].pose_landmarks[PoseLandmarkName.NOSE].presence, 0.7)

    def test_preserves_timing_metadata(self) -> None:
        frames = (_frame(0, 0.0),)
        config = SmoothingConfig(method=SmoothingMethod.MOVING_AVERAGE, window_size=2)

        result = MovingAverageSmoother().smooth(frames, config)

        self.assertEqual(result[0].timing, frames[0].timing)

    def test_rejects_unsupported_method(self) -> None:
        config = SmoothingConfig(method=SmoothingMethod.SAVITZKY_GOLAY, window_size=2)

        with self.assertRaises(ValueError):
            MovingAverageSmoother().smooth((_frame(0, 0.0),), config)

    def test_rejects_non_positive_window_size(self) -> None:
        config = SmoothingConfig(method=SmoothingMethod.MOVING_AVERAGE, window_size=0)

        with self.assertRaises(ValueError):
            MovingAverageSmoother().smooth((_frame(0, 0.0),), config)


if __name__ == "__main__":
    unittest.main()
