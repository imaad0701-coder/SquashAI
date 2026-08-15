"""Per-frame pose estimation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.tracking.base import TrackerConfig
from engine.types.landmarks import LandmarkFrame
from engine.types.video import FrameMeta


@dataclass(frozen=True)
class PoseEstimatorConfig:
    tracker_config: TrackerConfig
    model_complexity: int


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: FrameMeta, config: PoseEstimatorConfig) -> LandmarkFrame: ...
