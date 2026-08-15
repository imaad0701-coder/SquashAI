"""Metric-to-score evaluation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.scoring.benchmarks import BenchmarkSet
from engine.types.biomechanics import SwingMetrics
from engine.types.scoring import ScoreResult


class ScoreEngine(ABC):
    @abstractmethod
    def score(self, metrics: SwingMetrics, benchmarks: BenchmarkSet) -> tuple[ScoreResult, ...]: ...
