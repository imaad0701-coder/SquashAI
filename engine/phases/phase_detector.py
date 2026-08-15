"""Swing phase segmentation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.types.landmarks import LandmarkFrame
from engine.types.phases import PhaseSegment


class PhaseDetector(ABC):
    @abstractmethod
    def detect(self, frames: tuple[LandmarkFrame, ...]) -> tuple[PhaseSegment, ...]: ...
