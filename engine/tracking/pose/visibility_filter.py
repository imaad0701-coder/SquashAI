"""Filtering of low-confidence pose landmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.types.landmarks import LandmarkFrame


@dataclass(frozen=True)
class VisibilityThresholds:
    min_visibility: float
    min_presence: float


class VisibilityFilter(ABC):
    @abstractmethod
    def filter(self, frame: LandmarkFrame, thresholds: VisibilityThresholds) -> LandmarkFrame: ...


class ThresholdVisibilityFilter(VisibilityFilter):
    def filter(self, frame: LandmarkFrame, thresholds: VisibilityThresholds) -> LandmarkFrame:
        filtered = {
            name: landmark
            for name, landmark in frame.pose_landmarks.items()
            if landmark.visibility >= thresholds.min_visibility
            and landmark.presence >= thresholds.min_presence
        }
        return LandmarkFrame(
            timing=frame.timing, pose_landmarks=filtered, racket_landmarks=frame.racket_landmarks
        )
