"""Phase-duration analysis contracts (moved from biomechanics/)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.types.phases import PhaseLabel, PhaseSegment


@dataclass(frozen=True)
class PhaseTiming:
    label: PhaseLabel
    duration_seconds: float


class TimingAnalyzer(ABC):
    @abstractmethod
    def analyze(self, segments: tuple[PhaseSegment, ...], fps: float) -> tuple[PhaseTiming, ...]: ...
