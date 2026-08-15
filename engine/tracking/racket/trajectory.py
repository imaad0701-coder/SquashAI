"""Racket trajectory reconstruction contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.types.geometry import Point3D
from engine.types.landmarks import LandmarkFrame


@dataclass(frozen=True)
class RacketTrajectory:
    frame_indices: tuple[int, ...]
    head_positions: tuple[Point3D, ...]


class TrajectoryBuilder(ABC):
    @abstractmethod
    def build(self, frames: tuple[LandmarkFrame, ...]) -> RacketTrajectory: ...
