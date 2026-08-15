"""Hip angle: the angle at the hip between the trunk (to shoulder) and
thigh (to knee). See angle_calculator.py for the shared calculate/validate/
confidence/benchmark machinery this reuses."""

from __future__ import annotations

from engine.biomechanics.posture.angle_calculator import ThreePointAngleCalculator
from engine.types.biomechanics import JointAngleType, Side
from engine.types.landmarks import PoseLandmarkName


class HipAngleCalculator(ThreePointAngleCalculator):
    joint_angle_type = JointAngleType.HIP

    def _triplet(self, side: Side | None) -> tuple[PoseLandmarkName, PoseLandmarkName, PoseLandmarkName]:
        if side is Side.LEFT:
            return (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.LEFT_HIP, PoseLandmarkName.LEFT_KNEE)
        if side is Side.RIGHT:
            return (PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_HIP, PoseLandmarkName.RIGHT_KNEE)
        raise ValueError(f"HipAngleCalculator requires side=LEFT or RIGHT, got {side!r}")
