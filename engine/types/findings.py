"""Findings contracts: deterministic, within-player facts measured across a
clip's swings. Every finding has exactly one of three outcomes:

  - REPORTED: every trust gate passed; `value` holds the measured numbers.
  - SUPPRESSED: the finding's precondition held, but a trust gate failed --
    `gate_failures` records each failed gate with its observed and required
    values, and `swing_exclusions` records which swings were dropped and why.
    `value` is always None: a suppressed finding never leaks a number.
  - NOT_APPLICABLE: a precondition is absent (fewer than the minimum number
    of repetitions, handedness not supplied for a racket-side rule) --
    `not_applicable_reason` says which. Not a trust failure; there was
    nothing to evaluate.

Values are plain measurements (dispersion numbers, paired differences, peak
orderings). There are deliberately no verdict labels like "consistent" or
"asymmetric": a reader gets the number, and -- only where the underlying
measurement has a confidence channel -- a within-data noise-floor
comparison, never an adjective.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FindingKind(Enum):
    CONSISTENCY = "consistency"
    ASYMMETRY = "asymmetry"
    SEQUENCING = "sequencing"


class FindingOutcome(Enum):
    REPORTED = "reported"
    SUPPRESSED = "suppressed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class GateCheck:
    """One trust gate's result. `comparator` reads left to right as
    "observed <comparator> required" must hold to pass, e.g. observed=2,
    comparator=">=", required=3 fails."""

    gate: str
    observed: float | str
    comparator: str
    required: float | str
    passed: bool


@dataclass(frozen=True)
class SwingExclusion:
    """A single swing dropped from a finding's sample, and the gate it failed.
    swing_index is the position in AnalysisResult.swings."""

    swing_index: int
    check: GateCheck


class NoiseFloorComparison(Enum):
    """Across-swing dispersion compared with the measurement's own estimated
    frame-to-frame noise in this clip. Not a judgement of technique: WITHIN
    means the measured spread is not distinguishable from measurement noise,
    ABOVE means it is larger than that noise."""

    ABOVE_NOISE_FLOOR = "above_noise_floor"
    WITHIN_NOISE_FLOOR = "within_noise_floor"


@dataclass(frozen=True)
class ConsistencyValue:
    repetitions: int
    mean: float
    std: float  # sample standard deviation (ddof=1) across swings
    minimum: float
    maximum: float
    per_swing: tuple[tuple[int, float], ...]  # (swing_index, value)
    # Present only for measurements with a confidence channel (see
    # findings_engine's noise-floor estimate); None otherwise -- never a guess.
    noise_floor: float | None
    noise_floor_comparison: NoiseFloorComparison | None


@dataclass(frozen=True)
class AsymmetryValue:
    repetitions: int
    left_mean: float
    right_mean: float
    mean_difference: float  # left - right, paired per swing
    difference_std: float  # sample std (ddof=1) of the per-swing paired differences
    per_swing: tuple[tuple[int, float, float], ...]  # (swing_index, left, right)
    # Role labels for the two sides, when handedness was supplied; None
    # otherwise. Purely relabelling -- the numbers above are left/right.
    racket_side: str | None


@dataclass(frozen=True)
class SequencingValue:
    repetitions: int
    proximal_first_count: int  # elbow peak strictly before wrist peak
    distal_first_count: int  # wrist peak strictly before elbow peak
    same_frame_count: int
    median_lag_ms: float  # median of (wrist_peak_ms - elbow_peak_ms); positive = elbow first
    per_swing: tuple[tuple[int, float], ...]  # (swing_index, lag_ms)
    # Timing resolution: lags are only known to within one frame. None when
    # the clip's rate couldn't be measured (30fps fallback used for windows).
    ms_per_frame: float | None


FindingValue = ConsistencyValue | AsymmetryValue | SequencingValue


@dataclass(frozen=True)
class Finding:
    rule_id: str
    kind: FindingKind
    outcome: FindingOutcome
    description: str  # what was measured, as a plain fact -- not an interpretation
    unit: str
    value: FindingValue | None  # set iff outcome is REPORTED
    gate_failures: tuple[GateCheck, ...] = ()
    swing_exclusions: tuple[SwingExclusion, ...] = ()
    not_applicable_reason: str | None = None


@dataclass(frozen=True)
class FindingsReport:
    rule_set_version: str
    swing_count: int  # swings in the AnalysisResult, with or without a contact frame
    racket_side: str | None  # from the supplied handedness only; None when not supplied
    ms_per_frame: float | None
    findings: tuple[Finding, ...]
    # The wrist the phase detector actually used to find contact frames
    # (engine.phases.contact_detection.infer_racket_side over the same frames).
    # Racket-side findings are suppressed when it disagrees with racket_side.
    phase_racket_side: str | None = None
