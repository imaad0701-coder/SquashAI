"""Scoring and benchmark contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScoreBand(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ELITE = "elite"


@dataclass(frozen=True)
class BenchmarkSpec:
    metric_name: str
    band: ScoreBand
    min_value: float
    max_value: float


@dataclass(frozen=True)
class ScoreResult:
    metric_name: str
    raw_value: float
    band: ScoreBand
    score: float
