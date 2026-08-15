"""Landmark signal-smoothing contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame


class SmoothingMethod(Enum):
    MOVING_AVERAGE = "moving_average"
    SAVITZKY_GOLAY = "savitzky_golay"
    KALMAN = "kalman"


@dataclass(frozen=True)
class SmoothingConfig:
    method: SmoothingMethod
    window_size: int


class Smoother(ABC):
    @abstractmethod
    def smooth(
        self, frames: tuple[LandmarkFrame, ...], config: SmoothingConfig
    ) -> tuple[LandmarkFrame, ...]: ...


_MIN_WEIGHT_MS = 1e-6  # avoids zero total weight for a frame with delta_time_ms == 0.0


class MovingAverageSmoother(Smoother):
    """Trailing-window, time-weighted average of each landmark's position.

    `window_size` caps how many samples back to look, but frames are never
    assumed to be evenly spaced in time: each sample is weighted by its own
    `timing.delta_time_ms` (the real gap since whatever frame preceded it),
    so a sample from a long gap (e.g. after dropped frames) counts for more
    than one from a tightly-packed burst, and variable-fps footage is
    weighted correctly rather than by raw frame count.

    Visibility/presence are passed through from the current frame's own
    reading (unaveraged) — only position is smoothed.
    """

    def smooth(
        self, frames: tuple[LandmarkFrame, ...], config: SmoothingConfig
    ) -> tuple[LandmarkFrame, ...]:
        if config.method is not SmoothingMethod.MOVING_AVERAGE:
            raise ValueError(
                f"MovingAverageSmoother only supports {SmoothingMethod.MOVING_AVERAGE}, "
                f"got {config.method}"
            )
        if config.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {config.window_size}")

        smoothed_frames = []
        for index, current in enumerate(frames):
            window = frames[max(0, index - config.window_size + 1) : index + 1]
            smoothed_landmarks: dict = {}
            for name, landmark in current.pose_landmarks.items():
                samples = [
                    (other.pose_landmarks[name].position, max(other.timing.delta_time_ms, _MIN_WEIGHT_MS))
                    for other in window
                    if name in other.pose_landmarks
                ]
                total_weight = sum(weight for _, weight in samples)
                smoothed_landmarks[name] = Landmark(
                    position=Point3D(
                        x=sum(p.x * w for p, w in samples) / total_weight,
                        y=sum(p.y * w for p, w in samples) / total_weight,
                        z=sum(p.z * w for p, w in samples) / total_weight,
                    ),
                    visibility=landmark.visibility,
                    presence=landmark.presence,
                )
            smoothed_frames.append(
                LandmarkFrame(
                    timing=current.timing,
                    pose_landmarks=smoothed_landmarks,
                    racket_landmarks=current.racket_landmarks,
                )
            )
        return tuple(smoothed_frames)
