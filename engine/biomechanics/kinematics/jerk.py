"""Jerk: the first derivative of an acceleration trajectory (equivalently,
the third derivative of position), in pixels/second^3.
"""

from __future__ import annotations

from engine.biomechanics.kinematics.derivatives import VectorToVectorDerivative
from engine.types.biomechanics import JerkMeasurement
from engine.types.geometry import Vector3D
from engine.types.video import FrameTiming


class JerkCalculator(VectorToVectorDerivative[JerkMeasurement]):
    """Differentiates an acceleration trajectory -> jerk, in pixels/second^3."""

    def _build_measurement(
        self, timing: FrameTiming, value: Vector3D | None, is_valid: bool
    ) -> JerkMeasurement:
        return JerkMeasurement(
            frame_index=timing.frame_index,
            timestamp_ms=timing.timestamp_ms,
            jerk=value,
            is_valid=is_valid,
        )
