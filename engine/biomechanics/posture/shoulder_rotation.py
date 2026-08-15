"""Shoulder rotation: signed rotation of the shoulder line (left-shoulder ->
right-shoulder) about the vertical axis, as the heading of its XZ-plane
projection.

Same camera-relative caveat as pelvis_rotation.py.
"""

from __future__ import annotations

from engine.biomechanics.posture.angle_calculator import AngleCalculator
from engine.types.biomechanics import JointAngleType, Side
from engine.types.geometry import Point3D
from engine.types.landmarks import PoseLandmarkName
from engine.utils.geometry import heading_angle_degrees, magnitude, vector_between
from engine.utils.math_utils import EPSILON


class ShoulderRotationCalculator(AngleCalculator):
    joint_angle_type = JointAngleType.SHOULDER_ROTATION

    def _required_landmarks(self, side: Side | None) -> tuple[PoseLandmarkName, ...]:
        return (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.RIGHT_SHOULDER)

    def _measure(
        self, positions: dict[PoseLandmarkName, Point3D], required: tuple[PoseLandmarkName, ...]
    ) -> float | None:
        shoulder_vector = vector_between(
            positions[PoseLandmarkName.LEFT_SHOULDER], positions[PoseLandmarkName.RIGHT_SHOULDER]
        )
        if magnitude(shoulder_vector) <= EPSILON:
            return None
        return heading_angle_degrees(shoulder_vector)
