"""Shared tracking contracts used by both pose and racket trackers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from engine.types.video import FrameMeta

T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class TrackerConfig:
    min_detection_confidence: float
    min_tracking_confidence: float
    max_missed_frames: int


class Tracker(Protocol[T_co]):
    def track(self, frame: FrameMeta, config: TrackerConfig) -> T_co: ...
