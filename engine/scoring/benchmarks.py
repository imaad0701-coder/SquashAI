"""Benchmark definitions used by the scoring engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from engine.types.scoring import BenchmarkSpec

DEFAULT_SCORE_MIN: Final[float] = 0.0
DEFAULT_SCORE_MAX: Final[float] = 100.0


@dataclass(frozen=True)
class BenchmarkSet:
    shot_type_label: str
    specs: tuple[BenchmarkSpec, ...]
