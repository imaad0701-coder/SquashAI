"""Tests for CenterOfMassCalculator using a symmetric synthetic stick figure
with hand-computed expected center-of-mass position and height_ratio."""

from __future__ import annotations

import unittest

from engine.biomechanics.posture.center_of_mass import CenterOfMassCalculator
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.scoring import BenchmarkSpec, ScoreBand
from engine.types.video import FrameTiming

_TIMING = FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0)


def _landmark(x: float, y: float, z: float = 0.0, visibility: float = 0.9, presence: float = 0.9) -> Landmark:
    return Landmark(position=Point3D(x=x, y=y, z=z), visibility=visibility, presence=presence)


def _full_body_frame() -> LandmarkFrame:
    # Symmetric about x=0 (left at x=-1, right at x=1), so the weighted
    # centroid's x and z are always 0 regardless of which segments drop out
    # -- only y needs to be hand-computed.
    landmarks = {
        PoseLandmarkName.NOSE: _landmark(0, 0),
        PoseLandmarkName.LEFT_SHOULDER: _landmark(-1, 2),
        PoseLandmarkName.RIGHT_SHOULDER: _landmark(1, 2),
        PoseLandmarkName.LEFT_ELBOW: _landmark(-1, 4),
        PoseLandmarkName.RIGHT_ELBOW: _landmark(1, 4),
        PoseLandmarkName.LEFT_WRIST: _landmark(-1, 6),
        PoseLandmarkName.RIGHT_WRIST: _landmark(1, 6),
        PoseLandmarkName.LEFT_HIP: _landmark(-1, 8),
        PoseLandmarkName.RIGHT_HIP: _landmark(1, 8),
        PoseLandmarkName.LEFT_KNEE: _landmark(-1, 12),
        PoseLandmarkName.RIGHT_KNEE: _landmark(1, 12),
        PoseLandmarkName.LEFT_ANKLE: _landmark(-1, 16),
        PoseLandmarkName.RIGHT_ANKLE: _landmark(1, 16),
        PoseLandmarkName.LEFT_FOOT_INDEX: _landmark(-1, 17),
        PoseLandmarkName.RIGHT_FOOT_INDEX: _landmark(1, 17),
    }
    return LandmarkFrame(timing=_TIMING, pose_landmarks=landmarks)


# Hand-computed from Winter's mass fractions against the fixture above:
# head 0.081*0 + trunk 0.497*5 + arms 2*(0.028*3+0.022*5) + legs
# 2*(0.100*10+0.0465*14+0.0145*16.5) == 6.6535, total weight == 1.0.
_EXPECTED_COM_Y = 6.6535
_EXPECTED_HEIGHT_RATIO = (16.0 - _EXPECTED_COM_Y) / 16.0  # ankle_mid.y=16, nose.y=0


class CenterOfMassCalculatorTests(unittest.TestCase):
    def test_full_body_matches_hand_computed_weighted_centroid(self) -> None:
        calculator = CenterOfMassCalculator()

        measurement = calculator.calculate(_full_body_frame())

        self.assertTrue(measurement.is_valid)
        self.assertIsNotNone(measurement.position)
        self.assertAlmostEqual(measurement.position.x, 0.0, places=6)
        self.assertAlmostEqual(measurement.position.z, 0.0, places=6)
        self.assertAlmostEqual(measurement.position.y, _EXPECTED_COM_Y, places=3)

    def test_height_ratio_matches_hand_computed_value(self) -> None:
        measurement = CenterOfMassCalculator().calculate(_full_body_frame())

        self.assertIsNotNone(measurement.height_ratio)
        self.assertAlmostEqual(measurement.height_ratio, _EXPECTED_HEIGHT_RATIO, places=3)
        # Sanity: a relaxed standing posture's COM sits roughly half to
        # three-fifths of the way up from the feet.
        self.assertGreater(measurement.height_ratio, 0.5)
        self.assertLess(measurement.height_ratio, 0.65)

    def test_missing_trunk_landmark_is_invalid(self) -> None:
        frame = _full_body_frame()
        del frame.pose_landmarks[PoseLandmarkName.LEFT_HIP]
        calculator = CenterOfMassCalculator()

        measurement = calculator.calculate(frame)

        self.assertFalse(measurement.is_valid)
        self.assertIsNone(measurement.position)
        self.assertIsNone(measurement.height_ratio)
        self.assertFalse(calculator.validate(frame))
        self.assertEqual(calculator.confidence(frame), 0.0)

    def test_missing_limb_landmark_drops_that_segment_but_stays_valid(self) -> None:
        frame = _full_body_frame()
        del frame.pose_landmarks[PoseLandmarkName.LEFT_WRIST]

        measurement = CenterOfMassCalculator().calculate(frame)

        self.assertTrue(measurement.is_valid)
        # The dropped left forearm+hand segment shifts the renormalized
        # centroid away from the full-body value (and off the x=0 axis,
        # since that segment's counterpart on the right side is still in).
        self.assertNotAlmostEqual(measurement.position.y, _EXPECTED_COM_Y, places=3)

    def test_missing_ankle_leaves_position_valid_but_height_ratio_none(self) -> None:
        frame = _full_body_frame()
        del frame.pose_landmarks[PoseLandmarkName.LEFT_ANKLE]

        measurement = CenterOfMassCalculator().calculate(frame)

        self.assertTrue(measurement.is_valid)
        self.assertIsNone(measurement.height_ratio)

    def test_empty_frame_never_raises(self) -> None:
        measurement = CenterOfMassCalculator().calculate(LandmarkFrame(timing=_TIMING, pose_landmarks={}))
        self.assertFalse(measurement.is_valid)

    def test_confidence_is_weakest_link_across_used_landmarks(self) -> None:
        frame = _full_body_frame()
        frame.pose_landmarks[PoseLandmarkName.LEFT_HIP] = _landmark(-1, 8, visibility=0.55)
        calculator = CenterOfMassCalculator()

        measurement = calculator.calculate(frame)

        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.confidence, 0.55)
        self.assertAlmostEqual(calculator.confidence(frame), 0.55)


class CenterOfMassBenchmarkTests(unittest.TestCase):
    def test_height_ratio_within_range_scores_100(self) -> None:
        calculator = CenterOfMassCalculator()
        measurement = calculator.calculate(_full_body_frame())
        spec = BenchmarkSpec(
            metric_name="com_height_ratio", band=ScoreBand.INTERMEDIATE, min_value=0.5, max_value=0.65
        )

        result = calculator.benchmark(measurement, spec)

        self.assertEqual(result.score, 100.0)
        self.assertAlmostEqual(result.raw_value, _EXPECTED_HEIGHT_RATIO, places=3)

    def test_height_ratio_outside_range_scores_less_than_100(self) -> None:
        calculator = CenterOfMassCalculator()
        measurement = calculator.calculate(_full_body_frame())
        spec = BenchmarkSpec(metric_name="com_height_ratio", band=ScoreBand.ADVANCED, min_value=0.0, max_value=0.3)

        result = calculator.benchmark(measurement, spec)

        self.assertLess(result.score, 100.0)
        self.assertGreaterEqual(result.score, 0.0)

    def test_invalid_measurement_scores_zero(self) -> None:
        calculator = CenterOfMassCalculator()
        measurement = calculator.calculate(LandmarkFrame(timing=_TIMING, pose_landmarks={}))
        spec = BenchmarkSpec(metric_name="com_height_ratio", band=ScoreBand.BEGINNER, min_value=0.0, max_value=1.0)

        result = calculator.benchmark(measurement, spec)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw_value, 0.0)


if __name__ == "__main__":
    unittest.main()
