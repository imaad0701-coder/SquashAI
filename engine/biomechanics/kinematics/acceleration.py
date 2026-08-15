"""Linear acceleration: the first derivative of a velocity trajectory
(equivalently, the second derivative of position), in pixels/second^2.

Supersedes the old fps-based AccelerationCalculator.compute(velocities, fps)
stub for the same reason as velocity.py: that signature assumed constant
spacing, which this engine explicitly rejects, and nothing implemented it.
"""

from __future__ import annotations

from engine.biomechanics.kinematics.derivatives import VectorToVectorDerivative
from engine.types.biomechanics import AccelerationMeasurement
from engine.types.video import FrameTiming
from engine.types.geometry import Vector3D


class AccelerationCalculator(VectorToVectorDerivative[AccelerationMeasurement]):
    """Differentiates a velocity trajectory -> acceleration, in pixels/second^2."""

    def _build_measurement(
        self, timing: FrameTiming, value: Vector3D | None, is_valid: bool
    ) -> AccelerationMeasurement:
        return AccelerationMeasurement(
            frame_index=timing.frame_index,
            timestamp_ms=timing.timestamp_ms,
            acceleration=value,
            is_valid=is_valid,
        )
