"""Deterministic tests for VelocityCalculator / AccelerationCalculator /
JerkCalculator, using synthetic position/velocity/acceleration trajectories
with known analytical derivatives.
"""

from __future__ import annotations

import math
import unittest

from engine.biomechanics.kinematics.acceleration import AccelerationCalculator
from engine.biomechanics.kinematics.derivatives import DerivativeConfig, DerivativeMethod
from engine.biomechanics.kinematics.jerk import JerkCalculator
from engine.biomechanics.kinematics.velocity import VelocityCalculator
from engine.types.geometry import Point3D, Vector3D
from engine.types.video import FrameTiming
from engine.utils.geometry import magnitude


def _timing(index: int, ms: float) -> FrameTiming:
    return FrameTiming(frame_index=index, timestamp_ms=ms, delta_time_ms=0.0)


class ConstantVelocityTests(unittest.TestCase):
    SPEED = 50.0  # pixels/second
    TIMESTAMPS_MS = [0.0, 120.0, 260.0, 300.0, 500.0]  # deliberately irregular

    def _position_samples(self) -> list[tuple[FrameTiming, Point3D]]:
        return [
            (_timing(i, ms), Point3D(x=self.SPEED * (ms / 1000.0), y=0.0, z=0.0))
            for i, ms in enumerate(self.TIMESTAMPS_MS)
        ]

    def test_finite_difference_recovers_exact_velocity(self) -> None:
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)
        results = VelocityCalculator().compute(self._position_samples(), config)

        self.assertFalse(results[0].is_valid)
        for result in results[1:]:
            self.assertTrue(result.is_valid)
            self.assertAlmostEqual(result.velocity.x, self.SPEED, places=6)
            self.assertAlmostEqual(result.velocity.y, 0.0, places=9)
            self.assertAlmostEqual(result.velocity.z, 0.0, places=9)

    def test_central_difference_recovers_exact_velocity(self) -> None:
        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)
        results = VelocityCalculator().compute(self._position_samples(), config)

        self.assertFalse(results[0].is_valid)
        self.assertFalse(results[-1].is_valid)
        for result in results[1:-1]:
            self.assertTrue(result.is_valid)
            self.assertAlmostEqual(result.velocity.x, self.SPEED, places=6)

    def test_validate_and_smooth_never_raise(self) -> None:
        calculator = VelocityCalculator()
        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)
        samples = self._position_samples()

        self.assertTrue(calculator.validate(samples, config))
        smoothed = calculator.smooth(samples, config)
        self.assertEqual(len(smoothed), len(samples))


class ConstantAccelerationAndJerkTests(unittest.TestCase):
    def test_acceleration_calculator_recovers_exact_constant_acceleration(self) -> None:
        a = 20.0  # pixels/second^2
        timestamps_ms = [0.0, 90.0, 200.0, 260.0]
        velocity_samples = [
            (_timing(i, ms), Vector3D(x=a * (ms / 1000.0), y=0.0, z=0.0))
            for i, ms in enumerate(timestamps_ms)
        ]
        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)

        results = AccelerationCalculator().compute(velocity_samples, config)

        for result in results[1:-1]:
            self.assertTrue(result.is_valid)
            self.assertAlmostEqual(result.acceleration.x, a, places=6)

    def test_jerk_calculator_is_zero_for_constant_acceleration(self) -> None:
        a = 20.0
        timestamps_ms = [0.0, 90.0, 200.0, 260.0]
        acceleration_samples = [
            (_timing(i, ms), Vector3D(x=a, y=0.0, z=0.0)) for i, ms in enumerate(timestamps_ms)
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        results = JerkCalculator().compute(acceleration_samples, config)

        for result in results[1:]:
            self.assertTrue(result.is_valid)
            self.assertAlmostEqual(result.jerk.x, 0.0, places=9)


class SinusoidalMotionTests(unittest.TestCase):
    def test_velocity_matches_analytic_derivative_of_sine(self) -> None:
        amplitude = 100.0
        angular_frequency = 2 * math.pi  # 1 Hz
        dt_ms = 1.0
        n_samples = 200

        samples = []
        for i in range(n_samples):
            t_seconds = (i * dt_ms) / 1000.0
            samples.append(
                (
                    _timing(i, i * dt_ms),
                    Point3D(x=amplitude * math.sin(angular_frequency * t_seconds), y=0.0, z=0.0),
                )
            )

        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)
        results = VelocityCalculator().compute(samples, config)

        for i in range(50, 150):  # interior points, away from edges
            t_seconds = (i * dt_ms) / 1000.0
            analytic = amplitude * angular_frequency * math.cos(angular_frequency * t_seconds)
            self.assertTrue(results[i].is_valid)
            self.assertAlmostEqual(results[i].velocity.x, analytic, delta=0.1)


class CircularMotionTests(unittest.TestCase):
    def test_speed_is_constant_for_circular_motion(self) -> None:
        radius = 80.0
        angular_frequency = 3.0  # radians/second
        dt_ms = 1.0
        n_samples = 300
        expected_speed = radius * angular_frequency

        samples = []
        for i in range(n_samples):
            t_seconds = (i * dt_ms) / 1000.0
            samples.append(
                (
                    _timing(i, i * dt_ms),
                    Point3D(
                        x=radius * math.cos(angular_frequency * t_seconds),
                        y=radius * math.sin(angular_frequency * t_seconds),
                        z=0.0,
                    ),
                )
            )

        config = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)
        results = VelocityCalculator().compute(samples, config)

        for i in range(50, 250):
            self.assertTrue(results[i].is_valid)
            speed = magnitude(results[i].velocity)
            self.assertAlmostEqual(speed, expected_speed, delta=0.05)


class NumericalStabilityTests(unittest.TestCase):
    def test_duplicate_timestamps_do_not_produce_nan_or_raise(self) -> None:
        samples = [
            (_timing(0, 0.0), Point3D(x=0.0, y=0.0, z=0.0)),
            (_timing(1, 0.0), Point3D(x=1.0, y=0.0, z=0.0)),  # zero dt
            (_timing(2, 10.0), Point3D(x=2.0, y=0.0, z=0.0)),
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        results = VelocityCalculator().compute(samples, config)

        for result in results:
            self.assertFalse(result.is_valid)
            self.assertIsNone(result.velocity)

    def test_nan_position_component_never_raises(self) -> None:
        samples = [
            (_timing(0, 0.0), Point3D(x=math.nan, y=0.0, z=0.0)),
            (_timing(1, 10.0), Point3D(x=1.0, y=0.0, z=0.0)),
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        results = VelocityCalculator().compute(samples, config)

        self.assertTrue(all(not r.is_valid for r in results))

    def test_missing_frame_uses_real_gap(self) -> None:
        v = 50.0
        samples = [
            (_timing(0, 0.0), Point3D(x=0.0, y=0.0, z=0.0)),
            (_timing(1, 100.0), None),  # dropped frame
            (_timing(2, 300.0), Point3D(x=v * 0.3, y=0.0, z=0.0)),
        ]
        config = DerivativeConfig(method=DerivativeMethod.FINITE_DIFFERENCE)

        results = VelocityCalculator().compute(samples, config)

        self.assertFalse(results[1].is_valid)
        self.assertTrue(results[2].is_valid)
        self.assertAlmostEqual(results[2].velocity.x, v, places=6)

    def test_savitzky_golay_hook_never_raises(self) -> None:
        samples = [
            (_timing(i, i * 10.0), Point3D(x=float(i), y=0.0, z=0.0)) for i in range(5)
        ]
        config = DerivativeConfig(method=DerivativeMethod.SAVITZKY_GOLAY)

        results = VelocityCalculator().compute(samples, config)

        self.assertTrue(all(not r.is_valid and r.velocity is None for r in results))


if __name__ == "__main__":
    unittest.main()
