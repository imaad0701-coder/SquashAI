"""Skeleton rendering contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.types.landmarks import LandmarkFrame
from engine.visualization.drawing import DrawingStyle


class SkeletonRenderer(ABC):
    @abstractmethod
    def render(self, frame: LandmarkFrame, style: DrawingStyle) -> None: ...
