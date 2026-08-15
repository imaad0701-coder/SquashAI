"""Pose and racket landmark contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from engine.types.geometry import Point3D
from engine.types.video import FrameTiming


class PoseLandmarkName(Enum):
    NOSE = "nose"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_FOOT_INDEX = "left_foot_index"
    RIGHT_FOOT_INDEX = "right_foot_index"


class RacketKeypointName(Enum):
    HEAD_TIP = "head_tip"
    THROAT = "throat"
    GRIP_END = "grip_end"


@dataclass(frozen=True)
class Landmark:
    position: Point3D
    visibility: float
    presence: float


@dataclass(frozen=True)
class LandmarkFrame:
    timing: FrameTiming
    pose_landmarks: dict[PoseLandmarkName, Landmark] = field(default_factory=dict)
    racket_landmarks: dict[RacketKeypointName, Landmark] = field(default_factory=dict)
