"""Top-level pipeline result contract composing all other result types."""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.types.biomechanics import SwingMetrics
from engine.types.feedback import Recommendation
from engine.types.scoring import ScoreResult
from engine.types.shots import ShotType
from engine.types.video import VideoMetadata


@dataclass(frozen=True)
class PipelineResult:
    shot_type: ShotType
    video: VideoMetadata
    swing_metrics: SwingMetrics
    scores: tuple[ScoreResult, ...] = field(default_factory=tuple)
    recommendations: tuple[Recommendation, ...] = field(default_factory=tuple)
