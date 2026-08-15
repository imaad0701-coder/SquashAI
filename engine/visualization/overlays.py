"""Frame overlay rendering contracts (phase labels, scores, trajectories)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.types.phases import PhaseLabel
from engine.types.scoring import ScoreResult


class OverlayRenderer(ABC):
    @abstractmethod
    def render_phase_label(self, label: PhaseLabel) -> None: ...

    @abstractmethod
    def render_scores(self, scores: tuple[ScoreResult, ...]) -> None: ...
