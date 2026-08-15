"""Per-frame racket tracking contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.tracking.base import TrackerConfig
from engine.types.landmarks import LandmarkFrame
from engine.types.video import FrameMeta


class RacketTracker(ABC):
    @abstractmethod
    def track(self, frame: FrameMeta, config: TrackerConfig) -> LandmarkFrame: ...
