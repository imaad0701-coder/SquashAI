"""Balance/stability scoring contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.types.landmarks import LandmarkFrame


class BalanceAnalyzer(ABC):
    @abstractmethod
    def score(self, frame: LandmarkFrame) -> float: ...
