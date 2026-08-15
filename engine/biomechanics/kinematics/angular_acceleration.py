"""Angular acceleration: the first derivative of an angular-velocity
trajectory (equivalently, the second derivative of an angle trajectory),
in degrees/second^2.
"""

from __future__ import annotations

from engine.biomechanics.kinematics.derivatives import ScalarDerivative
from engine.types.biomechanics import AngularAccelerationMeasurement
from engine.types.video import FrameTiming


class AngularAccelerationCalculator(ScalarDerivative[AngularAccelerationMeasurement]):
    """Differentiates an angular-velocity trajectory (degrees/second) ->
    angular acceleration, in degrees/second^2."""

    def _build_measurement(
        self, timing: FrameTiming, value: float | None, is_valid: bool
    ) -> AngularAccelerationMeasurement:
        return AngularAccelerationMeasurement(
            frame_index=timing.frame_index,
            timestamp_ms=timing.timestamp_ms,
            angular_acceleration_degrees_per_second_squared=value,
            is_valid=is_valid,
        )
