"""Linear velocity: the first derivative of a position trajectory.

Supersedes the old fps-based VelocityCalculator.compute(frames, fps) stub —
that signature baked in a constant-fps assumption this engine explicitly
rejects. Nothing implemented that shape, so it's replaced outright rather
than kept alongside a contradictory legacy contract.
"""

from __future__ import annotations

from engine.biomechanics.kinematics.derivatives import TrajectoryDerivative
from engine.types.biomechanics import VelocityMeasurement
from engine.types.geometry import Point3D, Vector3D
from engine.types.video import FrameTiming


class PositionDerivative(TrajectoryDerivative[Point3D, VelocityMeasurement]):
    """First derivative of a position trajectory -> velocity, in pixels/second.

    Position is in pixels pending real-world calibration (see
    engine.calibration) — velocity is a rate of that same, currently-
    uncalibrated unit. This is also the shared vector-derivative engine
    reused by AccelerationCalculator and JerkCalculator, which differentiate
    Vector3D-valued velocity/acceleration trajectories with the same math.
    """

    def _to_components(self, value: Point3D) -> tuple[float, float, float]:
        return (value.x, value.y, value.z)

    def _from_components(self, components: tuple[float, ...]) -> Vector3D:
        return Vector3D(x=components[0], y=components[1], z=components[2])

    def _build_measurement(
        self, timing: FrameTiming, value: Vector3D | None, is_valid: bool
    ) -> VelocityMeasurement:
        return VelocityMeasurement(
            frame_index=timing.frame_index,
            timestamp_ms=timing.timestamp_ms,
            velocity=value,
            is_valid=is_valid,
        )


class VelocityCalculator(PositionDerivative):
    """Velocity is, by definition, the first derivative of position."""
