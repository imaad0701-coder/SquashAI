"""Validation-only execution paths: run a video through the existing,
already-implemented ShotPipeline (engine.pipelines.shots.shot_pipeline) and
report what it produces. No new pipeline, no new algorithm -- see
engine/pipelines/shots/shot_pipeline.py for what actually runs.

ShotPipeline computes an identical output shape for forehand and backhand
requests (posture, handedness/racket-side metadata included either way --
see ShotPipeline's own module docstring), so run_forehand_pipeline and
run_backhand_pipeline below are both thin wrappers with no shot-type-specific
logic of their own.
"""

from __future__ import annotations

from engine.api.interfaces import AnalysisRequest
from engine.pipelines.shots.shot_pipeline import ShotPipeline
from engine.types.results import PipelineResult
from engine.types.shots import Handedness, ShotType


def run_forehand_pipeline(video_path: str, session_id: str, handedness: Handedness) -> tuple[PipelineResult, dict]:
    pipeline = ShotPipeline(ShotType.FOREHAND)
    request = AnalysisRequest(
        video_path=video_path,
        shot_type=ShotType.FOREHAND,
        player_id="unknown",
        session_id=session_id,
        handedness=handedness,
    )
    return pipeline.run_with_debug(request)


def run_backhand_pipeline(video_path: str, session_id: str, handedness: Handedness) -> tuple[PipelineResult, dict]:
    pipeline = ShotPipeline(ShotType.BACKHAND)
    request = AnalysisRequest(
        video_path=video_path,
        shot_type=ShotType.BACKHAND,
        player_id="unknown",
        session_id=session_id,
        handedness=handedness,
    )
    return pipeline.run_with_debug(request)
