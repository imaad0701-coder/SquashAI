"""Tests for HeadStabilityCalculator using a synthetic 5-frame window with a
hand-computed expected normalized displacement."""

from __future__ import annotations

import unittest

from engine.biomechanics.posture.head_stability import HeadStabilityCalculator
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.scoring import BenchmarkSpec, ScoreBand
from engine.types.video import FrameTiming


def _landmark(x: float, y: float, z: float = 0.0, visibility: float = 0.9, presence: float = 0.9) -> Landmark:
    return Landmark(position=Point3D(x=x, y=y, z=z), visibility=visibility, presence=presence)


def _frame(index: int, nose: Landmark | None, include_shoulders: bool = True) -> LandmarkFrame:
    landmarks: dict[PoseLandmarkName, Landmark] = {}
    if nose is not None:
        landmarks[PoseLandmarkName.NOSE] = nose
    if include_shoulders:
        landmarks[PoseLandmarkName.LEFT_SHOULDER] = _landmark(-1, 0)
        landmarks[PoseLandmarkName.RIGHT_SHOULDER] = _landmark(1, 0)
    timing = FrameTiming(frame_index=index, timestamp_ms=index * 33.33, delta_time_ms=33.33)
    return LandmarkFrame(timing=timing, pose_landmarks=landmarks)


# Nose still at every frame except the center, which is displaced to (3, 4, 0).
# mean = (0.6, 0.8, 0); displacement = |(3,4,0) - (0.6,0.8,0)| = sqrt(2.4^2+3.2^2) = 4.0.
# shoulder width at center = distance((-1,0,0), (1,0,0)) = 2.0 -> ratio = 4.0 / 2.0 = 2.0.
def _five_frame_window() -> list[LandmarkFrame]:
    still = _landmark(0, 0, 0)
    displaced = _landmark(3, 4, 0)
    return [_frame(0, still), _frame(1, still), _frame(2, displaced), _frame(3, still), _frame(4, still)]


class HeadStabilityCalculatorTests(unittest.TestCase):
    def test_matches_hand_computed_normalized_displacement(self) -> None:
        frames = _five_frame_window()
        calculator = HeadStabilityCalculator()

        measurement = calculator.calculate(frames, center_index=2, window_size=5)

        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.normalized_displacement, 2.0, places=6)
        self.assertEqual(measurement.frame_index, 2)

    def test_perfectly_still_head_is_zero(self) -> None:
        still = _landmark(0, 0, 0)
        frames = [_frame(i, still) for i in range(5)]

        measurement = HeadStabilityCalculator().calculate(frames, center_index=2, window_size=5)

        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.normalized_displacement, 0.0, places=6)

    def test_window_truncates_at_sequence_start(self) -> None:
        # center_index=0, window_size=5 -> half_window=2 -> window is
        # frames[0:3] only (no frames before index 0 exist).
        still = _landmark(0, 0, 0)
        displaced = _landmark(3, 4, 0)
        frames = [_frame(0, displaced), _frame(1, still), _frame(2, still)]

        measurement = HeadStabilityCalculator().calculate(frames, center_index=0, window_size=5)

        self.assertTrue(measurement.is_valid)
        # mean of (3,4,0),(0,0,0),(0,0,0) = (1,4/3,0); displacement from (3,4,0).
        expected_mean = Point3D(x=1.0, y=4.0 / 3.0, z=0.0)
        expected_displacement = ((3 - expected_mean.x) ** 2 + (4 - expected_mean.y) ** 2) ** 0.5
        self.assertAlmostEqual(measurement.normalized_displacement, expected_displacement / 2.0, places=6)

    def test_single_frame_window_is_invalid(self) -> None:
        frames = [_frame(0, _landmark(0, 0, 0))]

        measurement = HeadStabilityCalculator().calculate(frames, center_index=0, window_size=5)

        self.assertFalse(measurement.is_valid)
        self.assertIsNone(measurement.normalized_displacement)

    def test_missing_center_nose_is_invalid(self) -> None:
        frames = _five_frame_window()
        frames[2] = _frame(2, None)

        calculator = HeadStabilityCalculator()
        measurement = calculator.calculate(frames, center_index=2, window_size=5)

        self.assertFalse(measurement.is_valid)
        self.assertFalse(calculator.validate(frames, center_index=2, window_size=5))

    def test_missing_shoulders_at_center_is_invalid(self) -> None:
        frames = _five_frame_window()
        frames[2] = _frame(2, _landmark(3, 4, 0), include_shoulders=False)

        measurement = HeadStabilityCalculator().calculate(frames, center_index=2, window_size=5)

        self.assertFalse(measurement.is_valid)

    def test_low_visibility_nose_is_invalid(self) -> None:
        frames = _five_frame_window()
        frames[2] = _frame(2, _landmark(3, 4, 0, visibility=0.1))

        self.assertFalse(HeadStabilityCalculator().calculate(frames, center_index=2, window_size=5).is_valid)

    def test_confidence_is_weakest_link(self) -> None:
        frames = _five_frame_window()
        frames[2] = _frame(2, _landmark(3, 4, 0, presence=0.6))
        calculator = HeadStabilityCalculator()

        self.assertAlmostEqual(calculator.confidence(frames, center_index=2, window_size=5), 0.6)


class HeadStabilityBenchmarkTests(unittest.TestCase):
    def test_displacement_within_range_scores_100(self) -> None:
        frames = [_frame(i, _landmark(0, 0, 0)) for i in range(5)]
        calculator = HeadStabilityCalculator()
        measurement = calculator.calculate(frames, center_index=2, window_size=5)
        spec = BenchmarkSpec(metric_name="head_stability", band=ScoreBand.ADVANCED, min_value=0.0, max_value=0.1)

        result = calculator.benchmark(measurement, spec)

        self.assertEqual(result.score, 100.0)
        self.assertAlmostEqual(result.raw_value, 0.0)

    def test_displacement_outside_range_scores_less_than_100(self) -> None:
        frames = _five_frame_window()  # normalized displacement == 2.0
        calculator = HeadStabilityCalculator()
        measurement = calculator.calculate(frames, center_index=2, window_size=5)
        spec = BenchmarkSpec(metric_name="head_stability", band=ScoreBand.BEGINNER, min_value=0.0, max_value=0.1)

        result = calculator.benchmark(measurement, spec)

        self.assertLess(result.score, 100.0)
        self.assertGreaterEqual(result.score, 0.0)

    def test_invalid_measurement_scores_zero(self) -> None:
        calculator = HeadStabilityCalculator()
        measurement = calculator.calculate([_frame(0, None)], center_index=0, window_size=5)
        spec = BenchmarkSpec(metric_name="head_stability", band=ScoreBand.BEGINNER, min_value=0.0, max_value=1.0)

        result = calculator.benchmark(measurement, spec)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw_value, 0.0)


if __name__ == "__main__":
    unittest.main()
