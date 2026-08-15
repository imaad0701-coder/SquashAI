"""Optional comparison against a prior forehand run of the SAME unmodified
pipeline, to answer: do measurements on this video behave differently than
they do on a forehand run of the identical engine?

Reads an existing main.py-produced JSON report (e.g. one of the forehand
runs already sitting in assets/outputs/) read-only, over plain JSON -- never
re-invokes the engine. Recomputes the same per-key stats
validation/diagnostics.py's AngleSeriesSummary produces, purely for a
side-by-side diff.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

from validation.diagnostics import AngleSeriesSummary

DEFAULT_BASELINE_PATH = "assets/outputs/post-fix-2_report.json"


@dataclass(frozen=True)
class BaselineAngleStat:
    key: str
    total_frames: int
    valid_fraction: float
    mean_confidence: float | None
    mean_angle: float | None


@dataclass(frozen=True)
class BaselineComparison:
    baseline_path: str
    baseline_video_path: str
    stats: tuple[BaselineAngleStat, ...]


@dataclass(frozen=True)
class BaselineDelta:
    key: str
    current_valid_fraction: float
    baseline_valid_fraction: float
    valid_fraction_delta: float
    current_mean_confidence: float | None
    baseline_mean_confidence: float | None
    confidence_delta: float | None
    current_mean_angle: float | None
    baseline_mean_angle: float | None


def load_baseline_angle_stats(path: str) -> BaselineComparison:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = []
    for key, measurements in data["angle_measurements"].items():
        valid = [m for m in measurements if m.get("is_valid") and m.get("angle_degrees") is not None]
        confidences = [m["confidence"] for m in valid]
        angles = [m["angle_degrees"] for m in valid]
        stats.append(
            BaselineAngleStat(
                key=key,
                total_frames=len(measurements),
                valid_fraction=(len(valid) / len(measurements) if measurements else 0.0),
                mean_confidence=(statistics.fmean(confidences) if confidences else None),
                mean_angle=(statistics.fmean(angles) if angles else None),
            )
        )
    return BaselineComparison(baseline_path=path, baseline_video_path=data["video"]["path"], stats=tuple(stats))


def diff_against_baseline(
    current_summaries: tuple[AngleSeriesSummary, ...], baseline: BaselineComparison
) -> tuple[BaselineDelta, ...]:
    baseline_by_key = {stat.key: stat for stat in baseline.stats}
    deltas = []
    for summary in current_summaries:
        base = baseline_by_key.get(summary.key)
        if base is None:
            continue
        confidence_delta = (
            summary.mean_confidence - base.mean_confidence
            if summary.mean_confidence is not None and base.mean_confidence is not None
            else None
        )
        deltas.append(
            BaselineDelta(
                key=summary.key,
                current_valid_fraction=summary.valid_fraction,
                baseline_valid_fraction=base.valid_fraction,
                valid_fraction_delta=summary.valid_fraction - base.valid_fraction,
                current_mean_confidence=summary.mean_confidence,
                baseline_mean_confidence=base.mean_confidence,
                confidence_delta=confidence_delta,
                current_mean_angle=summary.mean_angle,
                baseline_mean_angle=base.mean_angle,
            )
        )
    return tuple(deltas)
