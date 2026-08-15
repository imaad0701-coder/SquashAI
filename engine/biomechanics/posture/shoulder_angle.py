"""Shoulder angle: the angle at the shoulder between the trunk (to hip) and
upper arm (to elbow). See angle_calculator.py for the shared
calculate/validate/confidence/benchmark machinery this reuses."""

from __future__ import annotations

from engine.biomechanics.posture.angle_calculator import ThreePointAngleCalculator
from engine.types.biomechanics import JointAngleType, Side
from engine.types.landmarks import PoseLandmarkName


class ShoulderAngleCalculator(ThreePointAngleCalculator):
    joint_angle_type = JointAngleType.SHOULDER

    def _triplet(self, side: Side | None) -> tuple[PoseLandmarkName, PoseLandmarkName, PoseLandmarkName]:
        if side is Side.LEFT:
            return (PoseLandmarkName.LEFT_HIP, PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.LEFT_ELBOW)
        if side is Side.RIGHT:
            return (PoseLandmarkName.RIGHT_HIP, PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_ELBOW)
        raise ValueError(f"ShoulderAngleCalculator requires side=LEFT or RIGHT, got {side!r}")
