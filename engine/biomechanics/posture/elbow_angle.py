"""Elbow angle: the angle at the elbow between the upper arm (to shoulder)
and forearm (to wrist). See angle_calculator.py for the shared
calculate/validate/confidence/benchmark machinery this reuses."""

from __future__ import annotations

from engine.biomechanics.posture.angle_calculator import ThreePointAngleCalculator
from engine.types.biomechanics import JointAngleType, Side
from engine.types.landmarks import PoseLandmarkName


class ElbowAngleCalculator(ThreePointAngleCalculator):
    joint_angle_type = JointAngleType.ELBOW

    def _triplet(self, side: Side | None) -> tuple[PoseLandmarkName, PoseLandmarkName, PoseLandmarkName]:
        if side is Side.LEFT:
            return (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.LEFT_ELBOW, PoseLandmarkName.LEFT_WRIST)
        if side is Side.RIGHT:
            return (PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_ELBOW, PoseLandmarkName.RIGHT_WRIST)
        raise ValueError(f"ElbowAngleCalculator requires side=LEFT or RIGHT, got {side!r}")
