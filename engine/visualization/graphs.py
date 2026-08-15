"""Metric graph rendering contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.types.biomechanics import SwingMetrics


class GraphRenderer(ABC):
    @abstractmethod
    def render(self, metrics: SwingMetrics, output_path: str) -> None: ...
