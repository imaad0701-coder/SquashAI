"""Pipeline registry contract (renamed from pipeline_registery.py)."""

from __future__ import annotations

from typing import Protocol

from engine.pipelines.base_pipeline import Pipeline
from engine.types.shots import ShotType


class PipelineRegistry(Protocol):
    def register(self, shot_type: ShotType, pipeline: Pipeline) -> None: ...

    def get(self, shot_type: ShotType) -> Pipeline: ...
