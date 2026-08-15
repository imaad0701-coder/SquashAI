"""Ankle angle: the angle at the ankle between the shank (to knee) and foot
(to foot index). See angle_calculator.py for the shared calculate/validate/
confidence/benchmark machinery this reuses."""

from __future__ import annotations

from engine.biomechanics.posture.angle_calculator import ThreePointAngleCalculator
from engine.types.biomechanics import JointAngleType, Side
from engine.types.landmarks import PoseLandmarkName


class AnkleAngleCalculator(ThreePointAngleCalculator):
    joint_angle_type = JointAngleType.ANKLE

    def _triplet(self, side: Side | None) -> tuple[PoseLandmarkName, PoseLandmarkName, PoseLandmarkName]:
        if side is Side.LEFT:
            return (PoseLandmarkName.LEFT_KNEE, PoseLandmarkName.LEFT_ANKLE, PoseLandmarkName.LEFT_FOOT_INDEX)
        if side is Side.RIGHT:
            return (PoseLandmarkName.RIGHT_KNEE, PoseLandmarkName.RIGHT_ANKLE, PoseLandmarkName.RIGHT_FOOT_INDEX)
        raise ValueError(f"AnkleAngleCalculator requires side=LEFT or RIGHT, got {side!r}")
