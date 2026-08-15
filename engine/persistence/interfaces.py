"""Repository contracts for reading and writing engine artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engine.types.biomechanics import SwingMetrics
from engine.types.landmarks import LandmarkFrame
from engine.types.results import PipelineResult


@dataclass(frozen=True)
class StorageConfig:
    landmarks_dir: str
    metrics_dir: str
    reports_dir: str


class LandmarkRepository(Protocol):
    def save(self, session_id: str, frames: tuple[LandmarkFrame, ...]) -> None: ...

    def load(self, session_id: str) -> tuple[LandmarkFrame, ...]: ...


class MetricsRepository(Protocol):
    def save(self, session_id: str, metrics: SwingMetrics) -> None: ...

    def load(self, session_id: str) -> SwingMetrics: ...


class ReportRepository(Protocol):
    def save(self, session_id: str, result: PipelineResult) -> None: ...

    def load(self, session_id: str) -> PipelineResult: ...
