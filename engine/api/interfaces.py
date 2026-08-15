"""Contracts for driving the engine from a CLI or service entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engine.types.results import PipelineResult
from engine.types.shots import Handedness, ShotType


@dataclass(frozen=True)
class CLIArgs:
    video_path: str
    shot_type: ShotType
    session_id: str
    output_dir: str
    verbose: bool
    handedness: Handedness


@dataclass(frozen=True)
class AnalysisRequest:
    video_path: str
    shot_type: ShotType
    player_id: str
    session_id: str
    # Required at this API boundary -- callers must make an explicit choice,
    # never omit it -- but the value itself may be None (handedness genuinely
    # unknown at request time). See ShotPipeline.run_with_debug for how a
    # None here degrades racket_side/non_racket_side/side_roles rather than
    # guessing a default.
    handedness: Handedness | None


@dataclass(frozen=True)
class AnalysisResponse:
    request: AnalysisRequest
    result: PipelineResult


class AnalysisRunner(Protocol):
    def run(self, request: AnalysisRequest) -> AnalysisResponse: ...
