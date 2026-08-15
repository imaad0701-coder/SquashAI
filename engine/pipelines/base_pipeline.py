"""Base contract for a single shot-analysis pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.api.interfaces import AnalysisRequest
from engine.types.results import PipelineResult
from engine.types.shots import ShotType


class Pipeline(ABC):
    shot_type: ShotType

    @abstractmethod
    def run(self, request: AnalysisRequest) -> PipelineResult: ...
