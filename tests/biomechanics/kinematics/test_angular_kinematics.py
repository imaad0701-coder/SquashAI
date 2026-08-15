"""Deterministic tests for AngularVelocityCalculator and
AngularAccelerationCalculator, using synthetic angle trajectories with known
analytical derivatives.
"""

from __future__ import annotations

import math
import unittest

from engine.biomechanics.kinematics.angular_acceleration import AngularAccelerationCalculator
from engine.biomechanics.kinematics.derivatives import DerivativeConfig, DerivativeMethod
from engine.biomechanics.kinematics.rotations import AngularVelocityCalculator
from engine.types.video import FrameTiming


def _timing(index: int, ms: float) -> FrameTiming:
    return FrameTiming(frame_index=index, timestamp_ms=ms, delta_time_ms=0.0)


class AngularVelocityTests(unittest.TestCase):
    def test_linear_angle_trajectory_recovers_exact_angular_velocity(self) -> None:
        # theta(t) = omega * t, omega in degrees/second, irregular spacing.
        omega = 30.0
        timestamps_ms = [0.0, 80.0, 150.0, 400.0]
        samples = [(_timing(i, ms), omega * (ms / 1000.0)) for i, ms in enumerate(timestamps_ms)]

        for method in (DerivativeMethod.FINITE_DIFFERENCE, DerivativeMethod.CENTRAL_DIFFERENCE):
            with self.subTest(method=method):
                config = DerivativeConfig(method=method)
                results = AngularVelocityCalculator().compute(samples, config)
                valid_results = [r for r in results if r.is_valid]
                self.assertTrue(valid_results)
                for result in valid_results:
                    self.assertAlmostEqual(result.angular_velocity_degrees_per_second, omega, places=6)

    def test_validate_false_for_insufficient_samples(self) -> None:
        calculator = AngularVelocityCalculator()
        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)
        self.assertFalse(calculator.validate([(_timing(0, 0.0), 1.0)], config))

    def test_missing_angle_sample_is_skipped_using_real_gap(self) -> None:
        omega = 30.0
        samples = [
            (_timing(0, 0.0), 0.0),
            (_timing(1, 80.0), None),  # occluded joint / invalid angle measurement
            (_timing(2, 250.0), omega * 0.25),
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        results = AngularVelocityCalculator().compute(samples, config)

        self.assertFalse(results[1].is_valid)
        self.assertTrue(results[2].is_valid)
        self.assertAlmostEqual(results[2].angular_velocity_degrees_per_second, omega, places=6)


class AngularAccelerationTests(unittest.TestCase):
    def test_linear_angular_velocity_recovers_exact_angular_acceleration(self) -> None:
        # omega(t) = alpha * t, alpha in degrees/second^2.
        alpha = 15.0
        timestamps_ms = [0.0, 90.0, 220.0, 300.0]
        samples = [(_timing(i, ms), alpha * (ms / 1000.0)) for i, ms in enumerate(timestamps_ms)]
        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)

        results = AngularAccelerationCalculator().compute(samples, config)

        for result in results[1:-1]:
            self.assertTrue(result.is_valid)
            self.assertAlmostEqual(
                result.angular_acceleration_degrees_per_second_squared, alpha, places=6
            )

    def test_never_raises_on_duplicate_timestamps(self) -> None:
        samples = [
            (_timing(0, 0.0), 0.0),
            (_timing(1, 0.0), 5.0),  # zero dt
            (_timing(2, 10.0), 10.0),
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        results = AngularAccelerationCalculator().compute(samples, config)

        for result in results:
            self.assertFalse(result.is_valid)
            self.assertIsNone(result.angular_acceleration_degrees_per_second_squared)

    def test_nan_angle_never_raises(self) -> None:
        samples = [(_timing(0, 0.0), math.nan), (_timing(1, 10.0), 1.0), (_timing(2, 20.0), 2.0)]
        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)

        results = AngularAccelerationCalculator().compute(samples, config)

        self.assertTrue(all(not r.is_valid for r in results))

    def test_savitzky_golay_hook_never_raises(self) -> None:
        samples = [(_timing(i, i * 10.0), float(i)) for i in range(5)]
        config = DerivativeConfig(method=DerivativeMethod.SAVITZKY_GOLAY)

        results = AngularAccelerationCalculator().compute(samples, config)

        self.assertTrue(all(not r.is_valid for r in results))


if __name__ == "__main__":
    unittest.main()
