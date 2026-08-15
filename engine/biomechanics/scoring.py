"""Shared raw-value-against-range scoring, reused by every biomechanics
metric's benchmark() so a scoring-formula fix only has to happen once.
"""

from __future__ import annotations

from engine.types.scoring import BenchmarkSpec
from engine.utils.math_utils import EPSILON


def score_within_range(raw_value: float, spec: BenchmarkSpec) -> float:
    """100 inside [spec.min_value, spec.max_value], decaying linearly (never
    below 0) per unit of the spec's own range once outside it."""
    if spec.min_value <= raw_value <= spec.max_value:
        return 100.0
    span = max(spec.max_value - spec.min_value, EPSILON)
    deviation = (spec.min_value - raw_value) if raw_value < spec.min_value else (raw_value - spec.max_value)
    return max(0.0, 100.0 - (deviation / span) * 100.0)
