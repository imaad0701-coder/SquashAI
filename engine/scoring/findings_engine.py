"""Evaluates the v1 finding rules (engine.scoring.finding_rules) against one
clip: ShotPipeline's debug_report plus the AnalysisResult built from the same
landmark frames. Pure computation over data that already exists -- nothing
here re-runs tracking or phase detection.

Per rule, in order:
  1. Preconditions -> NOT_APPLICABLE: handedness for racket-side rules;
     at least MIN_REPETITIONS swings with a contact frame.
  2. Per-swing trust gates -> each failing swing is excluded, with the gate
     that failed recorded as a SwingExclusion.
  3. Enough usable swings left -> REPORTED; otherwise SUPPRESSED, with the
     repetition gate's observed-vs-required values and every exclusion.

Noise floor (consistency rules only, and only for series with a confidence
channel): the per-sample measurement noise of the series as reported, in
this clip, estimated from second differences over consecutive valid frames
whose confidence passes the same gate the samples do. The series are
computed from positions already passed through ShotPipeline's N-frame
moving average (N = PIPELINE_SMOOTHING_WINDOW). For white per-frame noise
of standard deviation s, an N-frame average leaves s/sqrt(N) on each
reported sample, and its second differences have standard deviation
2s/N -- so the reported-sample noise is sd(second differences) * sqrt(N)/2.
The spread of the second differences is taken robustly (MAD), so short
bursts of fast motion don't dominate it; sustained motion still adds to it,
so the estimate leans high (towards WITHIN_NOISE_FLOOR). It assumes noise
propagates roughly linearly from positions into angles, and it covers
frame-to-frame measurement noise only -- not the contact frame itself
landing a few frames off, which is a separate source of spread it does
not capture. A within-data reference point, not a calibrated noise model.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from engine.api.interfaces import AnalysisResult
from engine.phases.contact_detection import infer_racket_side
from engine.phases.frame_timing import ms_per_frame as measure_ms_per_frame
from engine.phases.frame_timing import ms_to_frames
from engine.scoring import gates
from engine.scoring.finding_rules import (
    ASYMMETRY_RULES,
    CONSISTENCY_RULES,
    CONTACT_WINDOW_POST_MS,
    CONTACT_WINDOW_PRE_MS,
    RULE_SET_VERSION,
    SEQUENCING_RULES,
    SERIES_HAS_CONFIDENCE,
    AsymmetryRule,
    AsymmetrySeries,
    ConsistencyRule,
    MetricSpec,
    SequencingAnchor,
    SequencingRule,
    Series,
    Statistic,
)
from engine.types.findings import (
    AsymmetryValue,
    ConsistencyValue,
    Finding,
    FindingKind,
    FindingOutcome,
    FindingsReport,
    GateCheck,
    NoiseFloorComparison,
    SequencingValue,
    SwingExclusion,
)
from engine.types.phases import SwingPhases
from engine.utils.geometry import magnitude

_HANDEDNESS_NOT_SUPPLIED = "handedness_not_supplied: racket-side rule needs AnalysisRequest.handedness"
_MAD_TO_SIGMA = 1.4826  # median absolute deviation -> standard deviation, for normal data
# Must equal ShotPipeline's default SmoothingConfig.window_size (drift guard:
# tests/scoring/test_findings_engine.py).
PIPELINE_SMOOTHING_WINDOW = 5


@dataclass(frozen=True)
class _Sample:
    value: float | None
    valid: bool
    confidence: float | None  # None when the series has no confidence channel


class _ClipView:
    """Index-aligned access to the debug_report series by frame_index (the
    unit AnalysisResult boundaries are expressed in)."""

    def __init__(self, debug_report: Mapping[str, Any], phase_racket_side: str | None = None) -> None:
        self.phase_racket_side = phase_racket_side
        self.frames = debug_report["landmark_frames"]
        self.angles = debug_report["angle_measurements"]
        self.kinematics = debug_report["kinematics"]
        self.posture = debug_report["posture"]
        self.racket_side: str | None = debug_report.get("racket_side")
        self.position_of = {f.timing.frame_index: i for i, f in enumerate(self.frames)}
        self.timestamps_ms = [f.timing.timestamp_ms for f in self.frames]
        self.ms_per_frame = measure_ms_per_frame([f.timing for f in self.frames])

    def __len__(self) -> int:
        return len(self.frames)

    def sampler(self, series: Series, key: str | None) -> Callable[[int], _Sample]:
        if series is Series.JOINT_ANGLE:
            seq = self.angles[key]
            return lambda p: _Sample(seq[p].angle_degrees, seq[p].is_valid and seq[p].angle_degrees is not None,
                                     seq[p].confidence)
        if series is Series.ANGULAR_VELOCITY:
            seq = self.kinematics[key]["angular_velocity"]
            return lambda p: _Sample(seq[p].angular_velocity_degrees_per_second,
                                     seq[p].is_valid and seq[p].angular_velocity_degrees_per_second is not None, None)
        if series is Series.COM_HEIGHT_RATIO:
            seq = self.posture["center_of_mass"]
            return lambda p: _Sample(seq[p].height_ratio, seq[p].is_valid and seq[p].height_ratio is not None,
                                     seq[p].confidence)
        if series is Series.HEAD_DISPLACEMENT:
            seq = self.posture["head_stability"]
            return lambda p: _Sample(seq[p].normalized_displacement,
                                     seq[p].is_valid and seq[p].normalized_displacement is not None, None)
        raise ValueError(f"unknown series {series}")

    def wrist_speed_sampler(self, side: str) -> Callable[[int], _Sample]:
        seq = self.kinematics[f"{side}_wrist"]["velocity"]

        def sample(p: int) -> _Sample:
            m = seq[p]
            ok = m.is_valid and m.velocity is not None
            return _Sample(magnitude(m.velocity) if ok else None, ok, None)

        return sample

    def contact_window(self, contact_pos: int, start_pos: int | None = None) -> tuple[int, int]:
        pre = ms_to_frames(CONTACT_WINDOW_PRE_MS, self.ms_per_frame)
        post = ms_to_frames(CONTACT_WINDOW_POST_MS, self.ms_per_frame)
        start = contact_pos - pre if start_pos is None else start_pos
        return max(0, start), min(len(self) - 1, contact_pos + post)


# --- per-swing extraction ----------------------------------------------------


def _at(sampler: Callable[[int], _Sample], pos: int, prefix: str) -> tuple[float | None, GateCheck | None]:
    s = sampler(pos)
    if not s.valid:
        return None, gates.sample_valid_gate(False, gate=f"{prefix}sample_valid")
    if s.confidence is not None:
        check = gates.sample_confidence_gate(s.confidence, gate=f"{prefix}sample_confidence")
        if not check.passed:
            return None, check
    return s.value, None


def _peak(view: _ClipView, sampler: Callable[[int], _Sample], window: tuple[int, int],
          prefix: str) -> tuple[int | None, float | None, GateCheck | None]:
    """Returns (peak_position, |value| at peak, failed gate or None)."""
    start, end = window
    samples = [sampler(p) for p in range(start, end + 1)]
    flags = [s.valid for s in samples]
    run_check = gates.invalid_run_gate(flags, view.ms_per_frame, gate=f"{prefix}window_longest_invalid_run_frames")
    if not run_check.passed:
        return None, None, run_check
    best = max((i for i, s in enumerate(samples) if s.valid), key=lambda i: abs(samples[i].value), default=None)
    if best is None:
        return None, None, gates.sample_valid_gate(False, gate=f"{prefix}window_has_valid_sample")
    edge_check = gates.peak_not_at_edge_gate(best, len(samples), gate=f"{prefix}peak_position")
    if not edge_check.passed:
        return None, None, edge_check
    return start + best, abs(samples[best].value), None


def _metric_value(view: _ClipView, sampler: Callable[[int], _Sample], statistic: Statistic, contact_pos: int,
                  prefix: str = "") -> tuple[float | None, GateCheck | None]:
    if statistic is Statistic.AT_CONTACT:
        return _at(sampler, contact_pos, prefix)
    if statistic is Statistic.PEAK_ABS_IN_CONTACT_WINDOW:
        _pos, value, failed = _peak(view, sampler, view.contact_window(contact_pos), prefix)
        return value, failed
    raise ValueError(f"unknown statistic {statistic}")


def _contact_swings(view: _ClipView, swings: tuple[SwingPhases, ...]) -> list[tuple[int, SwingPhases, int]]:
    """(swing_index, swing, contact_position) for swings whose contact frame exists in this clip."""
    out = []
    for i, swing in enumerate(swings):
        if swing.contact_frame is not None and swing.contact_frame in view.position_of:
            out.append((i, swing, view.position_of[swing.contact_frame]))
    return out


def _fewer_than_min_reps(count: int) -> str:
    return f"fewer_than_min_repetitions: {count} swing(s) with a detected contact, need >= {gates.MIN_REPETITIONS}"


# --- noise floor -------------------------------------------------------------


def _noise_floor(view: _ClipView, sampler: Callable[[int], _Sample]) -> float | None:
    def usable(p: int) -> float | None:
        s = sampler(p)
        if not s.valid or s.confidence is None or s.confidence < gates.MIN_SAMPLE_CONFIDENCE:
            return None
        return s.value

    values = [usable(p) for p in range(len(view))]
    second_diffs = [
        values[p + 1] - 2 * values[p] + values[p - 1]
        for p in range(1, len(values) - 1)
        if values[p - 1] is not None and values[p] is not None and values[p + 1] is not None
    ]
    if not second_diffs:
        return None
    center = statistics.median(second_diffs)
    mad = statistics.median(abs(d - center) for d in second_diffs)
    return _MAD_TO_SIGMA * mad * math.sqrt(PIPELINE_SMOOTHING_WINDOW) / 2


# --- rule evaluation -----------------------------------------------------------


def _not_applicable(rule_id: str, kind: FindingKind, description: str, unit: str, reason: str) -> Finding:
    return Finding(rule_id=rule_id, kind=kind, outcome=FindingOutcome.NOT_APPLICABLE, description=description,
                   unit=unit, value=None, not_applicable_reason=reason)


def _suppressed(rule_id: str, kind: FindingKind, description: str, unit: str, usable: int,
                exclusions: list[SwingExclusion]) -> Finding:
    return Finding(rule_id=rule_id, kind=kind, outcome=FindingOutcome.SUPPRESSED, description=description,
                   unit=unit, value=None, gate_failures=(gates.repetitions_gate(usable),),
                   swing_exclusions=tuple(exclusions))


def _racket_side_mismatch(rule_id: str, kind: FindingKind, description: str, unit: str,
                          view: _ClipView) -> Finding | None:
    check = gates.racket_side_agreement_gate(view.racket_side, view.phase_racket_side)
    if check.passed:
        return None
    return Finding(rule_id=rule_id, kind=kind, outcome=FindingOutcome.SUPPRESSED, description=description,
                   unit=unit, value=None, gate_failures=(check,))


def _series_key(metric: MetricSpec, side: str | None) -> str | None:
    if metric.joint is None:
        return None
    return f"{metric.joint}_{side}" if metric.sided else metric.joint


def evaluate_consistency(rule: ConsistencyRule, view: _ClipView, swings: tuple[SwingPhases, ...]) -> Finding:
    kind, metric = FindingKind.CONSISTENCY, rule.metric
    if metric.sided and view.racket_side is None:
        return _not_applicable(rule.rule_id, kind, rule.description, metric.unit, _HANDEDNESS_NOT_SUPPLIED)
    candidates = _contact_swings(view, swings)
    if len(candidates) < gates.MIN_REPETITIONS:
        return _not_applicable(rule.rule_id, kind, rule.description, metric.unit, _fewer_than_min_reps(len(candidates)))
    if metric.sided and (mismatch := _racket_side_mismatch(rule.rule_id, kind, rule.description, metric.unit, view)):
        return mismatch

    sampler = view.sampler(metric.series, _series_key(metric, view.racket_side))
    per_swing: list[tuple[int, float]] = []
    exclusions: list[SwingExclusion] = []
    for swing_index, _swing, contact_pos in candidates:
        value, failed = _metric_value(view, sampler, metric.statistic, contact_pos)
        if failed is not None:
            exclusions.append(SwingExclusion(swing_index, failed))
        else:
            per_swing.append((swing_index, value))

    if len(per_swing) < gates.MIN_REPETITIONS:
        return _suppressed(rule.rule_id, kind, rule.description, metric.unit, len(per_swing), exclusions)

    values = [v for _i, v in per_swing]
    std = statistics.stdev(values)
    noise_floor = comparison = None
    if metric.series in SERIES_HAS_CONFIDENCE:
        noise_floor = _noise_floor(view, sampler)
        if noise_floor is not None:
            comparison = (NoiseFloorComparison.ABOVE_NOISE_FLOOR if std > noise_floor
                          else NoiseFloorComparison.WITHIN_NOISE_FLOOR)
    value = ConsistencyValue(
        repetitions=len(values), mean=statistics.fmean(values), std=std, minimum=min(values), maximum=max(values),
        per_swing=tuple(per_swing), noise_floor=noise_floor, noise_floor_comparison=comparison,
    )
    return Finding(rule_id=rule.rule_id, kind=kind, outcome=FindingOutcome.REPORTED, description=rule.description,
                   unit=metric.unit, value=value, swing_exclusions=tuple(exclusions))


def _asymmetry_sampler(view: _ClipView, series: AsymmetrySeries, key: str) -> Callable[[int], _Sample]:
    mapping = {AsymmetrySeries.JOINT_ANGLE: Series.JOINT_ANGLE, AsymmetrySeries.ANGULAR_VELOCITY: Series.ANGULAR_VELOCITY}
    return view.sampler(mapping[series], key)


def evaluate_asymmetry(rule: AsymmetryRule, view: _ClipView, swings: tuple[SwingPhases, ...]) -> Finding:
    kind = FindingKind.ASYMMETRY
    candidates = _contact_swings(view, swings)
    if len(candidates) < gates.MIN_REPETITIONS:
        return _not_applicable(rule.rule_id, kind, rule.description, rule.unit, _fewer_than_min_reps(len(candidates)))

    left = _asymmetry_sampler(view, rule.left.series, f"{rule.left.joint}_left")
    right = _asymmetry_sampler(view, rule.right.series, f"{rule.right.joint}_right")
    per_swing: list[tuple[int, float, float]] = []
    exclusions: list[SwingExclusion] = []
    for swing_index, _swing, contact_pos in candidates:
        lv, lfail = _metric_value(view, left, rule.left.statistic, contact_pos, prefix="left_")
        rv, rfail = _metric_value(view, right, rule.right.statistic, contact_pos, prefix="right_")
        failed = lfail or rfail
        if failed is not None:
            exclusions.append(SwingExclusion(swing_index, failed))
        else:
            per_swing.append((swing_index, lv, rv))

    if len(per_swing) < gates.MIN_REPETITIONS:
        return _suppressed(rule.rule_id, kind, rule.description, rule.unit, len(per_swing), exclusions)

    diffs = [lv - rv for _i, lv, rv in per_swing]
    value = AsymmetryValue(
        repetitions=len(per_swing),
        left_mean=statistics.fmean(lv for _i, lv, _r in per_swing),
        right_mean=statistics.fmean(rv for _i, _l, rv in per_swing),
        mean_difference=statistics.fmean(diffs),
        difference_std=statistics.stdev(diffs),
        per_swing=tuple(per_swing),
        racket_side=view.racket_side,
    )
    return Finding(rule_id=rule.rule_id, kind=kind, outcome=FindingOutcome.REPORTED, description=rule.description,
                   unit=rule.unit, value=value, swing_exclusions=tuple(exclusions))


def evaluate_sequencing(rule: SequencingRule, view: _ClipView, swings: tuple[SwingPhases, ...]) -> Finding:
    kind, unit = FindingKind.SEQUENCING, "ms"
    if view.racket_side is None:
        return _not_applicable(rule.rule_id, kind, rule.description, unit, _HANDEDNESS_NOT_SUPPLIED)
    candidates = _contact_swings(view, swings)
    if len(candidates) < gates.MIN_REPETITIONS:
        return _not_applicable(rule.rule_id, kind, rule.description, unit, _fewer_than_min_reps(len(candidates)))
    if mismatch := _racket_side_mismatch(rule.rule_id, kind, rule.description, unit, view):
        return mismatch

    elbow = view.sampler(Series.ANGULAR_VELOCITY, f"elbow_{view.racket_side}")
    wrist = view.wrist_speed_sampler(view.racket_side)
    per_swing: list[tuple[int, float]] = []
    exclusions: list[SwingExclusion] = []
    for swing_index, swing, contact_pos in candidates:
        start_pos = None
        if rule.anchor is SequencingAnchor.BACKSWING:
            boundary = swing.backswing
            failed = next((c for c in (gates.boundary_detected_gate(boundary, "backswing_derivation"),
                                       gates.boundary_not_widened_gate(boundary, "backswing_search_range_widened"))
                           if not c.passed), None)
            if failed is None and (boundary.frame_index is None or boundary.frame_index not in view.position_of):
                failed = gates.sample_valid_gate(False, gate="backswing_frame_present")
            if failed is not None:
                exclusions.append(SwingExclusion(swing_index, failed))
                continue
            start_pos = view.position_of[boundary.frame_index]
        window = view.contact_window(contact_pos, start_pos)
        elbow_pos, _ev, efail = _peak(view, elbow, window, prefix="elbow_")
        wrist_pos, _wv, wfail = _peak(view, wrist, window, prefix="wrist_")
        failed = efail or wfail
        if failed is not None:
            exclusions.append(SwingExclusion(swing_index, failed))
            continue
        per_swing.append((swing_index, view.timestamps_ms[wrist_pos] - view.timestamps_ms[elbow_pos]))

    if len(per_swing) < gates.MIN_REPETITIONS:
        return _suppressed(rule.rule_id, kind, rule.description, unit, len(per_swing), exclusions)

    lags = [lag for _i, lag in per_swing]
    value = SequencingValue(
        repetitions=len(lags),
        proximal_first_count=sum(1 for lag in lags if lag > 0),
        distal_first_count=sum(1 for lag in lags if lag < 0),
        same_frame_count=sum(1 for lag in lags if lag == 0),
        median_lag_ms=statistics.median(lags),
        per_swing=tuple(per_swing),
        ms_per_frame=view.ms_per_frame,
    )
    return Finding(rule_id=rule.rule_id, kind=kind, outcome=FindingOutcome.REPORTED, description=rule.description,
                   unit=unit, value=value, swing_exclusions=tuple(exclusions))


def evaluate_findings(debug_report: Mapping[str, Any], analysis_result: AnalysisResult,
                      phase_racket_side: str | None = None) -> FindingsReport:
    """debug_report: ShotPipeline.run_with_debug()'s second return value.
    analysis_result: build_analysis_result() over the same landmark_frames.
    phase_racket_side: the wrist side ("left"/"right") the phase detector used
    to find contacts. Defaults to infer_racket_side over the same frames --
    the exact call KinematicPhaseDetector.detect() and build_analysis_result
    make -- so callers normally leave it out.

    The racket side for racket-side rules always comes from the supplied
    handedness (debug_report["racket_side"]), never from inference: with no
    handedness those rules are NOT_APPLICABLE, and when the supplied side
    disagrees with the phase detector's, they are SUPPRESSED."""
    if phase_racket_side is None:
        phase_racket_side = infer_racket_side(tuple(debug_report["landmark_frames"])).value.removesuffix("_wrist")
    view = _ClipView(debug_report, phase_racket_side)
    swings = analysis_result.swings
    findings = (
        tuple(evaluate_consistency(rule, view, swings) for rule in CONSISTENCY_RULES)
        + tuple(evaluate_asymmetry(rule, view, swings) for rule in ASYMMETRY_RULES)
        + tuple(evaluate_sequencing(rule, view, swings) for rule in SEQUENCING_RULES)
    )
    return FindingsReport(rule_set_version=RULE_SET_VERSION, swing_count=len(swings),
                          racket_side=view.racket_side, ms_per_frame=view.ms_per_frame, findings=findings,
                          phase_racket_side=phase_racket_side)
