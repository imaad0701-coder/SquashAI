"""Unit tests for TemporalPredictor's constant-velocity model."""

from __future__ import annotations

import unittest

from engine.tracking.reconstruction.temporal_predictor import TemporalPredictor
from engine.types.geometry import Point3D, Vector3D
from engine.types.video import FrameTiming


def _timing(frame_index: int, timestamp_ms: float) -> FrameTiming:
    return FrameTiming(frame_index=frame_index, timestamp_ms=timestamp_ms, delta_time_ms=0.0)


class EstimateVelocityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = TemporalPredictor()

    def test_returns_none_with_fewer_than_two_samples(self) -> None:
        self.assertIsNone(self.predictor.estimate_velocity([]))
        self.assertIsNone(self.predictor.estimate_velocity([(_timing(0, 0.0), Point3D(0, 0, 0))]))

    def test_returns_none_when_two_most_recent_share_a_timestamp(self) -> None:
        history = [
            (_timing(0, 100.0), Point3D(0, 0, 0)),
            (_timing(1, 100.0), Point3D(5, 5, 5)),
        ]
        self.assertIsNone(self.predictor.estimate_velocity(history))

    def test_computes_expected_velocity_from_two_points(self) -> None:
        history = [
            (_timing(0, 0.0), Point3D(0.0, 0.0, 0.0)),
            (_timing(1, 100.0), Point3D(10.0, 20.0, 30.0)),
        ]
        estimate = self.predictor.estimate_velocity(history)
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.velocity.x, 100.0, places=6)  # 10px / 0.1s
        self.assertAlmostEqual(estimate.velocity.y, 200.0, places=6)
        self.assertAlmostEqual(estimate.velocity.z, 300.0, places=6)
        self.assertEqual(estimate.reference_position, Point3D(10.0, 20.0, 30.0))
        self.assertEqual(estimate.reference_timestamp_ms, 100.0)

    def test_only_uses_the_two_most_recent_samples(self) -> None:
        # A distant, wildly different earlier sample must not affect the estimate.
        history = [
            (_timing(0, -10_000.0), Point3D(9999.0, 9999.0, 9999.0)),
            (_timing(1, 0.0), Point3D(0.0, 0.0, 0.0)),
            (_timing(2, 100.0), Point3D(10.0, 0.0, 0.0)),
        ]
        estimate = self.predictor.estimate_velocity(history)
        assert estimate is not None
        self.assertAlmostEqual(estimate.velocity.x, 100.0, places=6)
        self.assertAlmostEqual(estimate.velocity.y, 0.0, places=6)

    def test_negative_velocity_for_receding_motion(self) -> None:
        history = [
            (_timing(0, 0.0), Point3D(100.0, 100.0, 0.0)),
            (_timing(1, 100.0), Point3D(90.0, 80.0, 0.0)),
        ]
        estimate = self.predictor.estimate_velocity(history)
        assert estimate is not None
        self.assertAlmostEqual(estimate.velocity.x, -100.0, places=6)
        self.assertAlmostEqual(estimate.velocity.y, -200.0, places=6)


class PredictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = TemporalPredictor()

    def test_predict_extrapolates_forward(self) -> None:
        history = [
            (_timing(0, 0.0), Point3D(0.0, 0.0, 0.0)),
            (_timing(1, 100.0), Point3D(10.0, 20.0, 30.0)),
        ]
        estimate = self.predictor.estimate_velocity(history)
        assert estimate is not None

        predicted = self.predictor.predict(estimate, _timing(2, 200.0))
        self.assertAlmostEqual(predicted.x, 20.0, places=6)
        self.assertAlmostEqual(predicted.y, 40.0, places=6)
        self.assertAlmostEqual(predicted.z, 60.0, places=6)

    def test_predict_at_reference_timestamp_returns_reference_position(self) -> None:
        history = [
            (_timing(0, 0.0), Point3D(0.0, 0.0, 0.0)),
            (_timing(1, 100.0), Point3D(10.0, 20.0, 30.0)),
        ]
        estimate = self.predictor.estimate_velocity(history)
        assert estimate is not None

        predicted = self.predictor.predict(estimate, _timing(1, 100.0))
        self.assertEqual(predicted, Point3D(10.0, 20.0, 30.0))

    def test_predict_handles_zero_velocity(self) -> None:
        history = [
            (_timing(0, 0.0), Point3D(5.0, 5.0, 5.0)),
            (_timing(1, 100.0), Point3D(5.0, 5.0, 5.0)),
        ]
        estimate = self.predictor.estimate_velocity(history)
        assert estimate is not None

        predicted = self.predictor.predict(estimate, _timing(5, 500.0))
        self.assertEqual(predicted, Point3D(5.0, 5.0, 5.0))


class EstimateAccelerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = TemporalPredictor()

    def test_returns_none_with_fewer_than_three_samples(self) -> None:
        history = [(_timing(0, 0.0), Point3D(0, 0, 0)), (_timing(1, 100.0), Point3D(1, 0, 0))]
        self.assertIsNone(self.predictor.estimate_acceleration(history))

    def test_zero_acceleration_for_constant_velocity(self) -> None:
        # Equal steps every 100ms -> velocity constant -> acceleration ~0.
        history = [
            (_timing(0, 0.0), Point3D(0.0, 0.0, 0.0)),
            (_timing(1, 100.0), Point3D(10.0, 0.0, 0.0)),
            (_timing(2, 200.0), Point3D(20.0, 0.0, 0.0)),
        ]
        estimate = self.predictor.estimate_acceleration(history)
        assert estimate is not None
        self.assertAlmostEqual(estimate.acceleration.x, 0.0, places=6)

    def test_computes_expected_deceleration(self) -> None:
        # v1 = (10-0)/0.1 = 100 px/s ; v2 = (15-10)/0.1 = 50 px/s
        # acceleration = (50-100) / 0.1 = -500 px/s^2
        history = [
            (_timing(0, 0.0), Point3D(0.0, 0.0, 0.0)),
            (_timing(1, 100.0), Point3D(10.0, 0.0, 0.0)),
            (_timing(2, 200.0), Point3D(15.0, 0.0, 0.0)),
        ]
        estimate = self.predictor.estimate_acceleration(history)
        assert estimate is not None
        self.assertAlmostEqual(estimate.acceleration.x, -500.0, places=3)

    def test_returns_none_on_degenerate_time_gap(self) -> None:
        history = [
            (_timing(0, 0.0), Point3D(0.0, 0.0, 0.0)),
            (_timing(1, 0.0), Point3D(1.0, 0.0, 0.0)),  # same timestamp as sample 0
            (_timing(2, 100.0), Point3D(2.0, 0.0, 0.0)),
        ]
        self.assertIsNone(self.predictor.estimate_acceleration(history))


class PredictStepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = TemporalPredictor()

    def test_undamped_step_matches_plain_constant_velocity(self) -> None:
        # velocity_damping_rate=1.0 disables damping -- one 100ms step at
        # 100 px/s should move exactly 10px.
        current = Point3D(0.0, 0.0, 0.0)
        velocity = Vector3D(x=100.0, y=0.0, z=0.0)
        result = self.predictor.predict_step(
            current, velocity, dt_seconds=0.1, step_index=1, velocity_damping_rate=1.0
        )
        self.assertAlmostEqual(result.x, 10.0, places=6)

    def test_damping_reduces_displacement_at_later_steps(self) -> None:
        velocity = Vector3D(x=100.0, y=0.0, z=0.0)
        step1 = self.predictor.predict_step(
            Point3D(0.0, 0.0, 0.0), velocity, dt_seconds=0.1, step_index=1, velocity_damping_rate=0.5
        )
        step2 = self.predictor.predict_step(
            step1, velocity, dt_seconds=0.1, step_index=2, velocity_damping_rate=0.5
        )
        step3 = self.predictor.predict_step(
            step2, velocity, dt_seconds=0.1, step_index=3, velocity_damping_rate=0.5
        )

        increment1 = step1.x - 0.0
        increment2 = step2.x - step1.x
        increment3 = step3.x - step2.x

        self.assertAlmostEqual(increment1, 10.0, places=6)  # step_index=1: undamped (0.5^0=1)
        self.assertAlmostEqual(increment2, 5.0, places=6)  # 0.5^1
        self.assertAlmostEqual(increment3, 2.5, places=6)  # 0.5^2
        self.assertGreater(increment1, increment2)
        self.assertGreater(increment2, increment3)

    def test_cumulative_damped_displacement_converges_below_undamped_linear_growth(self) -> None:
        velocity = Vector3D(x=100.0, y=0.0, z=0.0)
        position = Point3D(0.0, 0.0, 0.0)
        for step in range(1, 21):
            position = self.predictor.predict_step(
                position, velocity, dt_seconds=0.1, step_index=step, velocity_damping_rate=0.8
            )
        # Undamped, 20 steps of 10px each would reach x=200. Damped (0.8
        # per step) must land well short of that -- this is the concrete
        # "avoid straight-line overshoot" property.
        self.assertLess(position.x, 200.0)
        # Geometric series bound: 10 * 1/(1-0.8) = 50
        self.assertLess(position.x, 51.0)

    def test_acceleration_term_curves_the_prediction(self) -> None:
        velocity = Vector3D(x=100.0, y=0.0, z=0.0)
        deceleration = Vector3D(x=-500.0, y=0.0, z=0.0)

        without_accel = self.predictor.predict_step(
            Point3D(0.0, 0.0, 0.0), velocity, dt_seconds=0.1, step_index=1, velocity_damping_rate=1.0
        )
        with_accel = self.predictor.predict_step(
            Point3D(0.0, 0.0, 0.0),
            velocity,
            dt_seconds=0.1,
            step_index=1,
            velocity_damping_rate=1.0,
            acceleration=deceleration,
            acceleration_damping_rate=1.0,
        )
        # Deceleration must pull the prediction back relative to the
        # velocity-only version.
        self.assertLess(with_accel.x, without_accel.x)
        # 0.5 * -500 * 0.1^2 = -2.5
        self.assertAlmostEqual(with_accel.x - without_accel.x, -2.5, places=6)


if __name__ == "__main__":
    unittest.main()
