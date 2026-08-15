"""Tests for WeightTransferCalculator using minimal synthetic frames (trunk
landmarks + both ankles) with hand-computed expected ratios.

Only the trunk landmarks are supplied (no elbows/wrists/knees/foot-index),
so CenterOfMassCalculator's estimate collapses to exactly the trunk segment
center -- avoiding fractional weighted-sum arithmetic in the expected
values here (that arithmetic is already covered by test_center_of_mass.py).
"""

from __future__ import annotations

import unittest

from engine.biomechanics.posture.weight_transfer import WeightTransferCalculator
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.scoring import BenchmarkSpec, ScoreBand
from engine.types.video import FrameTiming

_TIMING = FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0)


def _landmark(x: float, y: float, z: float = 0.0, visibility: float = 0.9, presence: float = 0.9) -> Landmark:
    return Landmark(position=Point3D(x=x, y=y, z=z), visibility=visibility, presence=presence)


def _frame(left_ankle: Landmark | None, right_ankle: Landmark | None) -> LandmarkFrame:
    landmarks: dict[PoseLandmarkName, Landmark] = {
        PoseLandmarkName.LEFT_SHOULDER: _landmark(-1, 0),
        PoseLandmarkName.RIGHT_SHOULDER: _landmark(1, 0),
        PoseLandmarkName.LEFT_HIP: _landmark(-1, 4),
        PoseLandmarkName.RIGHT_HIP: _landmark(1, 4),
    }
    if left_ankle is not None:
        landmarks[PoseLandmarkName.LEFT_ANKLE] = left_ankle
    if right_ankle is not None:
        landmarks[PoseLandmarkName.RIGHT_ANKLE] = right_ankle
    return LandmarkFrame(timing=_TIMING, pose_landmarks=landmarks)


class WeightTransferCalculatorTests(unittest.TestCase):
    def test_symmetric_stance_scores_half(self) -> None:
        # Trunk center is (0, 2, 0); ankles at (-2, 10) and (2, 10) are
        # equidistant from it along x -> right_foot_ratio == 0.5.
        frame = _frame(_landmark(-2, 10), _landmark(2, 10))
        calculator = WeightTransferCalculator()

        measurement = calculator.calculate(frame)

        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.right_foot_ratio, 0.5, places=6)

    def test_asymmetric_stance_matches_hand_computed_ratio(self) -> None:
        # Trunk center (0, 2, 0); left ankle (-3, 10), right ankle (5, 10).
        # stance_vector = (8, 0, 0); com_from_left_ankle = (3, -8, 0).
        # ratio = dot((3,-8,0), (8,0,0)) / 8^2 = 24 / 64 = 0.375.
        frame = _frame(_landmark(-3, 10), _landmark(5, 10))

        measurement = WeightTransferCalculator().calculate(frame)

        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.right_foot_ratio, 0.375, places=6)

    def test_missing_ankle_is_invalid(self) -> None:
        frame = _frame(_landmark(-2, 10), None)
        calculator = WeightTransferCalculator()

        measurement = calculator.calculate(frame)

        self.assertFalse(measurement.is_valid)
        self.assertIsNone(measurement.right_foot_ratio)
        self.assertFalse(calculator.validate(frame))

    def test_coincident_ankles_is_invalid(self) -> None:
        frame = _frame(_landmark(0, 10), _landmark(0, 10))
        self.assertFalse(WeightTransferCalculator().calculate(frame).is_valid)

    def test_low_visibility_ankle_is_invalid(self) -> None:
        frame = _frame(_landmark(-2, 10, visibility=0.1), _landmark(2, 10))
        self.assertFalse(WeightTransferCalculator().calculate(frame).is_valid)

    def test_unresolvable_center_of_mass_is_invalid(self) -> None:
        # No trunk landmarks at all -> CenterOfMassCalculator itself is invalid.
        frame = LandmarkFrame(
            timing=_TIMING,
            pose_landmarks={
                PoseLandmarkName.LEFT_ANKLE: _landmark(-2, 10),
                PoseLandmarkName.RIGHT_ANKLE: _landmark(2, 10),
            },
        )
        calculator = WeightTransferCalculator()

        measurement = calculator.calculate(frame)

        self.assertFalse(measurement.is_valid)
        self.assertEqual(calculator.confidence(frame), 0.0)

    def test_empty_frame_never_raises(self) -> None:
        measurement = WeightTransferCalculator().calculate(LandmarkFrame(timing=_TIMING, pose_landmarks={}))
        self.assertFalse(measurement.is_valid)

    def test_confidence_is_weakest_link(self) -> None:
        frame = _frame(_landmark(-2, 10, presence=0.6), _landmark(2, 10))
        calculator = WeightTransferCalculator()

        self.assertAlmostEqual(calculator.confidence(frame), 0.6)


class WeightTransferStanceStabilityTests(unittest.TestCase):
    """Regression coverage for a real finding from the engine-wide
    biomechanics validation pass: projecting COM onto the ankle-to-ankle
    line divides by stance width squared, so as stance width shrinks toward
    zero the projected direction becomes dominated by landmark noise and the
    resulting ratio blows up unbounded (real observed output before the
    fix: -7.6, +6.99). The old guard only checked stance width against
    EPSILON (1e-9, an absolute divide-by-zero floor) -- a stance a few
    pixels wide clears that easily while still being far too narrow to
    trust a direction from.

    Fixed via a minimum stance-width threshold expressed as a fraction of
    shoulder width (WeightTransferCalculator._DEFAULT_MIN_STANCE_WIDTH_RATIO
    = 0.15, configurable per-instance), below which the measurement is
    reported invalid rather than clamped or silently computed -- consistent
    with how every other "vector too short to trust a direction" case in
    this codebase (ThreePointAngleCalculator, TrunkInclinationCalculator,
    PelvisRotationCalculator, ShoulderRotationCalculator) already behaves.

    All fixtures here share the test module's _frame() helper, whose trunk
    landmarks (shoulders at x=-1/1) give a fixed shoulder_width of 2.0 and
    therefore a fixed default min_stance_width of 0.15 * 2.0 = 0.3.
    """

    def test_extremely_narrow_stance_is_now_invalid_not_unbounded(self) -> None:
        # Same fixture that used to produce right_foot_ratio == -29.5 before
        # the fix: ankles 0.1 apart (stance_width=0.1 < min_stance_width=0.3).
        frame = _frame(_landmark(2.95, 10), _landmark(3.05, 10))

        measurement = WeightTransferCalculator().calculate(frame)

        self.assertFalse(measurement.is_valid)
        self.assertIsNone(measurement.right_foot_ratio)
        self.assertEqual(measurement.confidence, 0.0)

    def test_feet_overlapping_is_invalid(self) -> None:
        # Ankles 0.02 apart -- feet effectively overlapping, well inside the
        # coincident-but-technically-nonzero regime this fix targets.
        frame = _frame(_landmark(3.0, 10), _landmark(3.02, 10))
        self.assertFalse(WeightTransferCalculator().calculate(frame).is_valid)

    def test_identical_ankle_positions_is_invalid(self) -> None:
        # stance_width == 0 exactly: caught by the EPSILON check, before the
        # threshold check is even reached.
        frame = _frame(_landmark(1.5, 10), _landmark(1.5, 10))
        self.assertFalse(WeightTransferCalculator().calculate(frame).is_valid)

    def test_noisy_narrow_stance_stays_invalid_across_jitter(self) -> None:
        # A handful of small, deterministic perturbations around a narrow
        # stance -- simulating detection jitter -- must all still read
        # invalid; none should slip through as a spuriously "valid" reading.
        for dx in (0.0, 0.01, -0.01, 0.02, -0.015):
            with self.subTest(dx=dx):
                frame = _frame(_landmark(-0.05 + dx, 10), _landmark(0.05 + dx, 10))
                self.assertFalse(WeightTransferCalculator().calculate(frame).is_valid)

    def test_noisy_normal_stance_stays_valid_and_stable(self) -> None:
        # Independent small jitter applied to each ankle of a comfortably-
        # wide stance (so stance width itself varies a little, ~3.9-4.1,
        # not just a rigid shared shift) must NOT destabilize the ratio --
        # the fix must not affect well-conditioned stances (regression
        # safety for the pre-existing, correct case).
        jitters = [(0.0, 0.0), (0.03, -0.02), (-0.04, 0.01), (0.02, 0.03), (-0.03, -0.03)]
        ratios = []
        for left_dx, right_dx in jitters:
            frame = _frame(_landmark(-2 + left_dx, 10), _landmark(2 + right_dx, 10))
            measurement = WeightTransferCalculator().calculate(frame)
            self.assertTrue(measurement.is_valid)
            ratios.append(measurement.right_foot_ratio)
        for ratio in ratios:
            self.assertAlmostEqual(ratio, 0.5, places=1)

    def test_confidence_is_zero_exactly_at_the_reliability_threshold(self) -> None:
        # stance_width == min_stance_width (0.3) exactly: valid (the
        # direction is defined), but the reliability ramp is 0 at its own
        # floor, so confidence reports zero trust rather than full trust.
        frame = _frame(_landmark(-0.15, 10), _landmark(0.15, 10))

        measurement = WeightTransferCalculator().calculate(frame)

        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.confidence, 0.0, places=6)

    def test_confidence_ramps_up_smoothly_between_threshold_and_twice_threshold(self) -> None:
        calculator = WeightTransferCalculator()
        # stance_width = 0.3 (1x threshold), 0.45 (1.5x), 0.6 (2x): confidence
        # should climb from 0.0 to a fraction of, up to, full base confidence.
        widths_and_expected = [(0.3, 0.0), (0.45, 0.45), (0.6, 0.9)]
        for width, expected_confidence in widths_and_expected:
            with self.subTest(width=width):
                half = width / 2
                frame = _frame(_landmark(-half, 10), _landmark(half, 10))
                measurement = calculator.calculate(frame)
                self.assertTrue(measurement.is_valid)
                self.assertAlmostEqual(measurement.confidence, expected_confidence, places=6)

    def test_stance_well_above_twice_threshold_reports_full_confidence(self) -> None:
        frame = _frame(_landmark(-2, 10), _landmark(2, 10))  # stance_width=4.0 >> 2*0.3
        measurement = WeightTransferCalculator().calculate(frame)
        self.assertTrue(measurement.is_valid)
        self.assertAlmostEqual(measurement.confidence, 0.9)  # full base confidence, no ramp penalty

    def test_threshold_is_configurable(self) -> None:
        # A stance that's invalid under the default 0.15 ratio becomes valid
        # under a looser, explicitly-configured ratio.
        frame = _frame(_landmark(-0.15, 10), _landmark(0.15, 10))  # stance_width=0.3

        strict = WeightTransferCalculator(min_stance_width_ratio=0.2)
        loose = WeightTransferCalculator(min_stance_width_ratio=0.1)

        self.assertFalse(strict.calculate(frame).is_valid)  # 0.3 < 0.2*2.0=0.4
        self.assertTrue(loose.calculate(frame).is_valid)  # 0.3 >= 0.1*2.0=0.2

    def test_missing_shoulder_reference_is_invalid(self) -> None:
        # No body-scale reference available -> can't judge stance
        # reliability at all, so this is invalid rather than falling back
        # to the old, unsafe EPSILON-only behavior.
        frame = LandmarkFrame(
            timing=_TIMING,
            pose_landmarks={
                PoseLandmarkName.LEFT_HIP: _landmark(-1, 4),
                PoseLandmarkName.RIGHT_HIP: _landmark(1, 4),
                PoseLandmarkName.LEFT_ANKLE: _landmark(-2, 10),
                PoseLandmarkName.RIGHT_ANKLE: _landmark(2, 10),
            },
        )
        # (COM itself also requires both shoulders, so this is already
        # invalid via the COM check -- included here as an explicit,
        # named regression rather than relying on that as incidental.)
        self.assertFalse(WeightTransferCalculator().calculate(frame).is_valid)


class WeightTransferBenchmarkTests(unittest.TestCase):
    def test_ratio_within_range_scores_100(self) -> None:
        frame = _frame(_landmark(-2, 10), _landmark(2, 10))
        calculator = WeightTransferCalculator()
        measurement = calculator.calculate(frame)
        spec = BenchmarkSpec(metric_name="weight_transfer", band=ScoreBand.INTERMEDIATE, min_value=0.4, max_value=0.6)

        result = calculator.benchmark(measurement, spec)

        self.assertEqual(result.score, 100.0)
        self.assertAlmostEqual(result.raw_value, 0.5)

    def test_ratio_outside_range_scores_less_than_100(self) -> None:
        frame = _frame(_landmark(-2, 10), _landmark(2, 10))  # ratio 0.5
        calculator = WeightTransferCalculator()
        measurement = calculator.calculate(frame)
        spec = BenchmarkSpec(metric_name="weight_transfer", band=ScoreBand.ADVANCED, min_value=0.8, max_value=1.0)

        result = calculator.benchmark(measurement, spec)

        self.assertLess(result.score, 100.0)
        self.assertGreaterEqual(result.score, 0.0)

    def test_invalid_measurement_scores_zero(self) -> None:
        calculator = WeightTransferCalculator()
        measurement = calculator.calculate(LandmarkFrame(timing=_TIMING, pose_landmarks={}))
        spec = BenchmarkSpec(metric_name="weight_transfer", band=ScoreBand.BEGINNER, min_value=0.0, max_value=1.0)

        result = calculator.benchmark(measurement, spec)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw_value, 0.0)


if __name__ == "__main__":
    unittest.main()
