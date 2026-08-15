"""Trunk inclination: angle of the trunk (mid-hip -> mid-shoulder) from
vertical. See angle_calculator.py for the shared calculate/validate/
confidence/benchmark machinery this reuses."""

from __future__ import annotations

from typing import Final

from engine.biomechanics.posture.angle_calculator import AngleCalculator, midpoint
from engine.types.biomechanics import JointAngleType, Side
from engine.types.geometry import Point3D, Vector3D
from engine.types.landmarks import PoseLandmarkName
from engine.utils.geometry import angle_between, magnitude, vector_between
from engine.utils.math_utils import EPSILON

# "Up" in normalized image-space landmark coordinates (Y increases downward).
# Trunk inclination is measured against this, not true world-vertical — that
# would need calibrated world coordinates (see engine.calibration), which
# aren't available yet.
_VERTICAL_REFERENCE: Final[Vector3D] = Vector3D(x=0.0, y=-1.0, z=0.0)


class TrunkInclinationCalculator(AngleCalculator):
    """Not left/right specific: uses both shoulders and both hips. 0 degrees
    is an upright trunk; larger values indicate more forward/lateral lean."""

    joint_angle_type = JointAngleType.TRUNK_INCLINATION

    def _required_landmarks(self, side: Side | None) -> tuple[PoseLandmarkName, ...]:
        return (
            PoseLandmarkName.LEFT_SHOULDER,
            PoseLandmarkName.RIGHT_SHOULDER,
            PoseLandmarkName.LEFT_HIP,
            PoseLandmarkName.RIGHT_HIP,
        )

    def _measure(
        self, positions: dict[PoseLandmarkName, Point3D], required: tuple[PoseLandmarkName, ...]
    ) -> float | None:
        mid_shoulder = midpoint(positions[PoseLandmarkName.LEFT_SHOULDER], positions[PoseLandmarkName.RIGHT_SHOULDER])
        mid_hip = midpoint(positions[PoseLandmarkName.LEFT_HIP], positions[PoseLandmarkName.RIGHT_HIP])

        trunk_vector = vector_between(mid_hip, mid_shoulder)
        if magnitude(trunk_vector) <= EPSILON:
            return None  # coincident shoulder/hip midpoints
        return angle_between(trunk_vector, _VERTICAL_REFERENCE)
