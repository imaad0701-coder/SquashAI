"""Comprehensive tests for the single-frame joint-angle calculators (each
living in its own module under engine.biomechanics.posture: knee_angle.py,
hip_angle.py, elbow_angle.py, shoulder_angle.py, ankle_angle.py,
trunk_inclination.py, pelvis_rotation.py, shoulder_rotation.py), using
synthetic landmark configurations with known, hand-computed expected angles.
Grouped by shared behavior (all ThreePointAngleCalculator subclasses share
identical edge-case handling from angle_calculator.py) rather than split
1:1 with source files, to avoid duplicating that edge-case coverage per
calculator.
"""

from __future__ import annotations

import math
import unittest

from engine.biomechanics.posture.ankle_angle import AnkleAngleCalculator
from engine.biomechanics.posture.elbow_angle import ElbowAngleCalculator
from engine.biomechanics.posture.hip_angle import HipAngleCalculator
from engine.biomechanics.posture.knee_angle import KneeAngleCalculator
from engine.biomechanics.posture.pelvis_rotation import PelvisRotationCalculator
from engine.biomechanics.posture.shoulder_angle import ShoulderAngleCalculator
from engine.biomechanics.posture.shoulder_rotation import ShoulderRotationCalculator
from engine.biomechanics.posture.trunk_inclination import TrunkInclinationCalculator
from engine.types.biomechanics import JointAngleType, Side
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.scoring import BenchmarkSpec, ScoreBand
from engine.types.video import FrameTiming

_TIMING = FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0)


def _landmark(x: float, y: float, z: float, visibility: float = 0.9, presence: float = 0.9) -> Landmark:
    return Landmark(position=Point3D(x=x, y=y, z=z), visibility=visibility, presence=presence)


def _frame(landmarks: dict[PoseLandmarkName, Landmark]) -> LandmarkFrame:
    return LandmarkFrame(timing=_TIMING, pose_landmarks=dict(landmarks))


# ---------------------------------------------------------------------------
# Three-point (vertex) calculators: knee, hip, elbow, shoulder, ankle.
# Pattern used throughout: vertex at origin, "near" landmark at (0,-1,0).
# LEFT cases put "far" at (1,0,0) -> a clean, known 90 degree angle.
# RIGHT cases put "far" at (0,1,0) -> directly opposite "near" -> 180 degrees.
# ---------------------------------------------------------------------------

_THREE_POINT_CASES = [
    (
        "knee",
        KneeAngleCalculator,
        JointAngleType.KNEE,
        (PoseLandmarkName.LEFT_HIP, PoseLandmarkName.LEFT_KNEE, PoseLandmarkName.LEFT_ANKLE),
        (PoseLandmarkName.RIGHT_HIP, PoseLandmarkName.RIGHT_KNEE, PoseLandmarkName.RIGHT_ANKLE),
    ),
    (
        "hip",
        HipAngleCalculator,
        JointAngleType.HIP,
        (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.LEFT_HIP, PoseLandmarkName.LEFT_KNEE),
        (PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_HIP, PoseLandmarkName.RIGHT_KNEE),
    ),
    (
        "elbow",
        ElbowAngleCalculator,
        JointAngleType.ELBOW,
        (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.LEFT_ELBOW, PoseLandmarkName.LEFT_WRIST),
        (PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_ELBOW, PoseLandmarkName.RIGHT_WRIST),
    ),
    (
        "shoulder",
        ShoulderAngleCalculator,
        JointAngleType.SHOULDER,
        (PoseLandmarkName.LEFT_HIP, PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.LEFT_ELBOW),
        (PoseLandmarkName.RIGHT_HIP, PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_ELBOW),
    ),
    (
        "ankle",
        AnkleAngleCalculator,
        JointAngleType.ANKLE,
        (PoseLandmarkName.LEFT_KNEE, PoseLandmarkName.LEFT_ANKLE, PoseLandmarkName.LEFT_FOOT_INDEX),
        (PoseLandmarkName.RIGHT_KNEE, PoseLandmarkName.RIGHT_ANKLE, PoseLandmarkName.RIGHT_FOOT_INDEX),
    ),
]


def _ninety_degree_frame(near: PoseLandmarkName, vertex: PoseLandmarkName, far: PoseLandmarkName) -> LandmarkFrame:
    return _frame({near: _landmark(0, -1, 0), vertex: _landmark(0, 0, 0), far: _landmark(1, 0, 0)})


def _straight_frame(near: PoseLandmarkName, vertex: PoseLandmarkName, far: PoseLandmarkName) -> LandmarkFrame:
    return _frame({near: _landmark(0, -1, 0), vertex: _landmark(0, 0, 0), far: _landmark(0, 1, 0)})


class ThreePointAngleCalculatorTests(unittest.TestCase):
    def test_known_90_degree_configuration_left_side(self) -> None:
        for name, cls, joint_angle_type, left_triplet, _right_triplet in _THREE_POINT_CASES:
            with self.subTest(joint=name):
                near, vertex, far = left_triplet
                frame = _ninety_degree_frame(near, vertex, far)
                measurement = cls().calculate(frame, side=Side.LEFT)

                self.assertTrue(measurement.is_valid)
                self.assertEqual(measurement.joint_angle_type, joint_angle_type)
                self.assertEqual(measurement.side, Side.LEFT)
                self.assertAlmostEqual(measurement.angle_degrees, 90.0)
                self.assertGreater(measurement.confidence, 0.0)

    def test_known_180_degree_configuration_right_side(self) -> None:
        for name, cls, joint_angle_type, _left_triplet, right_triplet in _THREE_POINT_CASES:
            with self.subTest(joint=name):
                near, vertex, far = right_triplet
                frame = _straight_frame(near, vertex, far)
                measurement = cls().calculate(frame, side=Side.RIGHT)

                self.assertTrue(measurement.is_valid)
                self.assertEqual(measurement.side, Side.RIGHT)
                self.assertAlmostEqual(measurement.angle_degrees, 180.0)

    def test_validate_true_for_complete_frame(self) -> None:
        for name, cls, _jat, left_triplet, _rt in _THREE_POINT_CASES:
            with self.subTest(joint=name):
                near, vertex, far = left_triplet
                frame = _ninety_degree_frame(near, vertex, far)
                self.assertTrue(cls().validate(frame, side=Side.LEFT))

    def test_missing_side_raises_nothing_and_is_invalid(self) -> None:
        for name, cls, _jat, left_triplet, _rt in _THREE_POINT_CASES:
            with self.subTest(joint=name):
                near, vertex, far = left_triplet
                frame = _ninety_degree_frame(near, vertex, far)
                calculator = cls()

                measurement = calculator.calculate(frame, side=None)

                self.assertFalse(measurement.is_valid)
                self.assertIsNone(measurement.angle_degrees)
                self.assertEqual(measurement.confidence, 0.0)
                self.assertFalse(calculator.validate(frame, side=None))
                self.assertEqual(calculator.confidence(frame, side=None), 0.0)


class KneeAngleCalculatorEdgeCaseTests(unittest.TestCase):
    """Detailed edge-case coverage using the simplest three-point calculator;
    the failure modes here are identical across all ThreePointAngleCalculator
    subclasses since they share the same base implementation."""

    def _valid_frame(self) -> LandmarkFrame:
        return _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, -1, 0),
                PoseLandmarkName.LEFT_KNEE: _landmark(0, 0, 0),
                PoseLandmarkName.LEFT_ANKLE: _landmark(1, 0, 0),
            }
        )

    def test_missing_landmark_is_invalid_not_an_exception(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, -1, 0),
                PoseLandmarkName.LEFT_KNEE: _landmark(0, 0, 0),
                # LEFT_ANKLE occluded/never detected
            }
        )
        calculator = KneeAngleCalculator()

        measurement = calculator.calculate(frame, side=Side.LEFT)

        self.assertFalse(measurement.is_valid)
        self.assertIsNone(measurement.angle_degrees)
        self.assertEqual(measurement.confidence, 0.0)
        self.assertFalse(calculator.validate(frame, side=Side.LEFT))
        self.assertEqual(calculator.confidence(frame, side=Side.LEFT), 0.0)

    def test_empty_frame_is_invalid_not_an_exception(self) -> None:
        calculator = KneeAngleCalculator()
        measurement = calculator.calculate(_frame({}), side=Side.LEFT)
        self.assertFalse(measurement.is_valid)

    def test_low_visibility_landmark_is_invalid(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, -1, 0),
                PoseLandmarkName.LEFT_KNEE: _landmark(0, 0, 0),
                PoseLandmarkName.LEFT_ANKLE: _landmark(1, 0, 0, visibility=0.1),
            }
        )
        calculator = KneeAngleCalculator()
        self.assertFalse(calculator.validate(frame, side=Side.LEFT))
        self.assertFalse(calculator.calculate(frame, side=Side.LEFT).is_valid)

    def test_low_presence_landmark_is_invalid(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, -1, 0),
                PoseLandmarkName.LEFT_KNEE: _landmark(0, 0, 0),
                PoseLandmarkName.LEFT_ANKLE: _landmark(1, 0, 0, presence=0.2),
            }
        )
        calculator = KneeAngleCalculator()
        self.assertFalse(calculator.validate(frame, side=Side.LEFT))

    def test_nan_coordinate_is_invalid(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, -1, 0),
                PoseLandmarkName.LEFT_KNEE: _landmark(math.nan, 0, 0),
                PoseLandmarkName.LEFT_ANKLE: _landmark(1, 0, 0),
            }
        )
        calculator = KneeAngleCalculator()

        measurement = calculator.calculate(frame, side=Side.LEFT)

        self.assertFalse(measurement.is_valid)
        self.assertFalse(calculator.validate(frame, side=Side.LEFT))

    def test_infinite_coordinate_is_invalid(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(math.inf, -1, 0),
                PoseLandmarkName.LEFT_KNEE: _landmark(0, 0, 0),
                PoseLandmarkName.LEFT_ANKLE: _landmark(1, 0, 0),
            }
        )
        self.assertFalse(KneeAngleCalculator().calculate(frame, side=Side.LEFT).is_valid)

    def test_coincident_vertex_and_neighbor_is_invalid(self) -> None:
        # LEFT_KNEE and LEFT_ANKLE at the exact same position -> zero-length
        # vector, not a real angle.
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, -1, 0),
                PoseLandmarkName.LEFT_KNEE: _landmark(0, 0, 0),
                PoseLandmarkName.LEFT_ANKLE: _landmark(0, 0, 0),
            }
        )
        calculator = KneeAngleCalculator()

        measurement = calculator.calculate(frame, side=Side.LEFT)

        self.assertFalse(measurement.is_valid)
        self.assertIsNone(measurement.angle_degrees)

    def test_all_three_coincident_is_invalid(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(2, 2, 2),
                PoseLandmarkName.LEFT_KNEE: _landmark(2, 2, 2),
                PoseLandmarkName.LEFT_ANKLE: _landmark(2, 2, 2),
            }
        )
        self.assertFalse(KneeAngleCalculator().calculate(frame, side=Side.LEFT).is_valid)

    def test_valid_frame_reports_high_confidence(self) -> None:
        measurement = KneeAngleCalculator().calculate(self._valid_frame(), side=Side.LEFT)
        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.confidence, 0.9)

    def test_confidence_is_the_weakest_link_across_required_landmarks(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, -1, 0, visibility=0.95, presence=0.95),
                PoseLandmarkName.LEFT_KNEE: _landmark(0, 0, 0, visibility=0.6, presence=0.99),
                PoseLandmarkName.LEFT_ANKLE: _landmark(1, 0, 0, visibility=0.85, presence=0.85),
            }
        )
        calculator = KneeAngleCalculator()

        measurement = calculator.calculate(frame, side=Side.LEFT)

        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.confidence, 0.6)
        self.assertAlmostEqual(calculator.confidence(frame, side=Side.LEFT), 0.6)


class TrunkInclinationCalculatorTests(unittest.TestCase):
    def test_upright_trunk_is_zero_degrees(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_SHOULDER: _landmark(-1, -2, 0),
                PoseLandmarkName.RIGHT_SHOULDER: _landmark(1, -2, 0),
                PoseLandmarkName.LEFT_HIP: _landmark(-1, 0, 0),
                PoseLandmarkName.RIGHT_HIP: _landmark(1, 0, 0),
            }
        )
        measurement = TrunkInclinationCalculator().calculate(frame)

        self.assertTrue(measurement.is_valid)
        self.assertEqual(measurement.joint_angle_type, JointAngleType.TRUNK_INCLINATION)
        self.assertIsNone(measurement.side)
        self.assertAlmostEqual(measurement.angle_degrees, 0.0)

    def test_45_degree_lean(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_SHOULDER: _landmark(1, -2, 0),
                PoseLandmarkName.RIGHT_SHOULDER: _landmark(3, -2, 0),
                PoseLandmarkName.LEFT_HIP: _landmark(-1, 0, 0),
                PoseLandmarkName.RIGHT_HIP: _landmark(1, 0, 0),
            }
        )
        measurement = TrunkInclinationCalculator().calculate(frame)

        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.angle_degrees, 45.0)

    def test_missing_landmark_is_invalid(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_SHOULDER: _landmark(-1, -2, 0),
                PoseLandmarkName.RIGHT_SHOULDER: _landmark(1, -2, 0),
                PoseLandmarkName.LEFT_HIP: _landmark(-1, 0, 0),
            }
        )
        calculator = TrunkInclinationCalculator()
        self.assertFalse(calculator.validate(frame))
        self.assertFalse(calculator.calculate(frame).is_valid)

    def test_coincident_midpoints_is_invalid(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_SHOULDER: _landmark(0, 0, 0),
                PoseLandmarkName.RIGHT_SHOULDER: _landmark(0, 0, 0),
                PoseLandmarkName.LEFT_HIP: _landmark(0, 0, 0),
                PoseLandmarkName.RIGHT_HIP: _landmark(0, 0, 0),
            }
        )
        self.assertFalse(TrunkInclinationCalculator().calculate(frame).is_valid)

    def test_empty_frame_never_raises(self) -> None:
        measurement = TrunkInclinationCalculator().calculate(_frame({}))
        self.assertFalse(measurement.is_valid)


class RotationCalculatorTests(unittest.TestCase):
    def test_pelvis_rotation_along_x_axis_is_zero(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(-1, 0, 0),
                PoseLandmarkName.RIGHT_HIP: _landmark(1, 0, 0),
            }
        )
        measurement = PelvisRotationCalculator().calculate(frame)

        self.assertTrue(measurement.is_valid)
        self.assertEqual(measurement.joint_angle_type, JointAngleType.PELVIS_ROTATION)
        self.assertIsNone(measurement.side)
        self.assertAlmostEqual(measurement.angle_degrees, 0.0)

    def test_pelvis_rotation_along_z_axis_is_90(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, 0, -1),
                PoseLandmarkName.RIGHT_HIP: _landmark(0, 0, 1),
            }
        )
        measurement = PelvisRotationCalculator().calculate(frame)
        self.assertAlmostEqual(measurement.angle_degrees, 90.0)

    def test_shoulder_rotation_along_x_axis_is_zero(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_SHOULDER: _landmark(-1, 0, 0),
                PoseLandmarkName.RIGHT_SHOULDER: _landmark(1, 0, 0),
            }
        )
        measurement = ShoulderRotationCalculator().calculate(frame)

        self.assertTrue(measurement.is_valid)
        self.assertEqual(measurement.joint_angle_type, JointAngleType.SHOULDER_ROTATION)
        self.assertAlmostEqual(measurement.angle_degrees, 0.0)

    def test_shoulder_rotation_along_z_axis_is_90(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_SHOULDER: _landmark(0, 0, -1),
                PoseLandmarkName.RIGHT_SHOULDER: _landmark(0, 0, 1),
            }
        )
        measurement = ShoulderRotationCalculator().calculate(frame)
        self.assertAlmostEqual(measurement.angle_degrees, 90.0)

    def test_ignores_side_argument_since_not_sided(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(-1, 0, 0),
                PoseLandmarkName.RIGHT_HIP: _landmark(1, 0, 0),
            }
        )
        calculator = PelvisRotationCalculator()
        without_side = calculator.calculate(frame)
        with_side = calculator.calculate(frame, side=Side.LEFT)

        self.assertTrue(without_side.is_valid)
        self.assertTrue(with_side.is_valid)
        self.assertAlmostEqual(without_side.angle_degrees, with_side.angle_degrees)

    def test_coincident_hips_is_invalid(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(1, 1, 1),
                PoseLandmarkName.RIGHT_HIP: _landmark(1, 1, 1),
            }
        )
        self.assertFalse(PelvisRotationCalculator().calculate(frame).is_valid)

    def test_missing_landmark_never_raises(self) -> None:
        frame = _frame({PoseLandmarkName.LEFT_HIP: _landmark(-1, 0, 0)})
        calculator = PelvisRotationCalculator()

        measurement = calculator.calculate(frame)

        self.assertFalse(measurement.is_valid)
        self.assertFalse(calculator.validate(frame))
        self.assertEqual(calculator.confidence(frame), 0.0)


class BenchmarkTests(unittest.TestCase):
    def test_angle_within_range_scores_100(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, -1, 0),
                PoseLandmarkName.LEFT_KNEE: _landmark(0, 0, 0),
                PoseLandmarkName.LEFT_ANKLE: _landmark(1, 0, 0),
            }
        )
        calculator = KneeAngleCalculator()
        measurement = calculator.calculate(frame, side=Side.LEFT)
        spec = BenchmarkSpec(metric_name="knee_angle", band=ScoreBand.INTERMEDIATE, min_value=80.0, max_value=100.0)

        result = calculator.benchmark(measurement, spec)

        self.assertEqual(result.score, 100.0)
        self.assertEqual(result.raw_value, 90.0)
        self.assertEqual(result.band, ScoreBand.INTERMEDIATE)

    def test_angle_outside_range_scores_less_than_100(self) -> None:
        frame = _frame(
            {
                PoseLandmarkName.LEFT_HIP: _landmark(0, -1, 0),
                PoseLandmarkName.LEFT_KNEE: _landmark(0, 0, 0),
                PoseLandmarkName.LEFT_ANKLE: _landmark(1, 0, 0),
            }
        )
        calculator = KneeAngleCalculator()
        measurement = calculator.calculate(frame, side=Side.LEFT)  # 90 degrees
        spec = BenchmarkSpec(metric_name="knee_angle", band=ScoreBand.ADVANCED, min_value=0.0, max_value=45.0)

        result = calculator.benchmark(measurement, spec)

        self.assertLess(result.score, 100.0)
        self.assertGreaterEqual(result.score, 0.0)

    def test_invalid_measurement_scores_zero(self) -> None:
        frame = _frame({})
        calculator = KneeAngleCalculator()
        measurement = calculator.calculate(frame, side=Side.LEFT)
        spec = BenchmarkSpec(metric_name="knee_angle", band=ScoreBand.BEGINNER, min_value=0.0, max_value=180.0)

        result = calculator.benchmark(measurement, spec)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw_value, 0.0)


if __name__ == "__main__":
    unittest.main()
