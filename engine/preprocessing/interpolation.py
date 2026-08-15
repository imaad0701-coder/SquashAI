"""Landmark gap-filling contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from engine.types.landmarks import LandmarkFrame


class InterpolationMethod(Enum):
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    NEAREST = "nearest"


class Interpolator(ABC):
    @abstractmethod
    def interpolate(
        self, frames: tuple[LandmarkFrame, ...], method: InterpolationMethod
    ) -> tuple[LandmarkFrame, ...]: ...
