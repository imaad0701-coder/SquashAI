"""Pelvis rotation (aka hip rotation/turn in coaching terms): signed
rotation of the hip line (left-hip -> right-hip) about the vertical axis,
as the heading of its XZ-plane projection.

This is a proxy for pelvis turn relative to the camera's X axis, not a
true court-relative facing angle — that needs calibrated world
coordinates (see engine.calibration), which aren't available yet.
"""

from __future__ import annotations

from engine.biomechanics.posture.angle_calculator import AngleCalculator
from engine.types.biomechanics import JointAngleType, Side
from engine.types.geometry import Point3D
from engine.types.landmarks import PoseLandmarkName
from engine.utils.geometry import heading_angle_degrees, magnitude, vector_between
from engine.utils.math_utils import EPSILON


class PelvisRotationCalculator(AngleCalculator):
    joint_angle_type = JointAngleType.PELVIS_ROTATION

    def _required_landmarks(self, side: Side | None) -> tuple[PoseLandmarkName, ...]:
        return (PoseLandmarkName.LEFT_HIP, PoseLandmarkName.RIGHT_HIP)

    def _measure(
        self, positions: dict[PoseLandmarkName, Point3D], required: tuple[PoseLandmarkName, ...]
    ) -> float | None:
        hip_vector = vector_between(positions[PoseLandmarkName.LEFT_HIP], positions[PoseLandmarkName.RIGHT_HIP])
        if magnitude(hip_vector) <= EPSILON:
            return None
        return heading_angle_degrees(hip_vector)
