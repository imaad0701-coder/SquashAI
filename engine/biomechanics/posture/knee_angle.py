"""Knee angle: the angle at the knee between the thigh (to hip) and shank
(to ankle). See angle_calculator.py for the shared calculate/validate/
confidence/benchmark machinery this reuses."""

from __future__ import annotations

from engine.biomechanics.posture.angle_calculator import ThreePointAngleCalculator
from engine.types.biomechanics import JointAngleType, Side
from engine.types.landmarks import PoseLandmarkName


class KneeAngleCalculator(ThreePointAngleCalculator):
    joint_angle_type = JointAngleType.KNEE

    def _triplet(self, side: Side | None) -> tuple[PoseLandmarkName, PoseLandmarkName, PoseLandmarkName]:
        if side is Side.LEFT:
            return (PoseLandmarkName.LEFT_HIP, PoseLandmarkName.LEFT_KNEE, PoseLandmarkName.LEFT_ANKLE)
        if side is Side.RIGHT:
            return (PoseLandmarkName.RIGHT_HIP, PoseLandmarkName.RIGHT_KNEE, PoseLandmarkName.RIGHT_ANKLE)
        raise ValueError(f"KneeAngleCalculator requires side=LEFT or RIGHT, got {side!r}")
