"""Angular velocity: the first derivative of an angle trajectory (e.g. from
engine.biomechanics.posture.knee_angle or any other AngleCalculator
subclass), in degrees/second.

Supersedes the old fps-based RotationCalculator.compute(frames, fps) stub
for the same reason as velocity.py/acceleration.py — constant-spacing
assumption, nothing implemented it. That older shape also operated
directly on raw LandmarkFrame sequences to produce a 3D rotation vector;
this engine instead differentiates the scalar angle trajectories the
joint-angle engine already produces, which is what AngularVelocityCalculator
and AngularAccelerationCalculator are for.
"""

from __future__ import annotations

from engine.biomechanics.kinematics.derivatives import ScalarDerivative
from engine.types.biomechanics import AngularVelocityMeasurement
from engine.types.video import FrameTiming


class AngularVelocityCalculator(ScalarDerivative[AngularVelocityMeasurement]):
    """Differentiates an angle trajectory (degrees) -> angular velocity, in degrees/second."""

    def _build_measurement(
        self, timing: FrameTiming, value: float | None, is_valid: bool
    ) -> AngularVelocityMeasurement:
        return AngularVelocityMeasurement(
            frame_index=timing.frame_index,
            timestamp_ms=timing.timestamp_ms,
            angular_velocity_degrees_per_second=value,
            is_valid=is_valid,
        )
