"""Trust gates for the findings engine. Each gate returns a GateCheck that
records the observed and required values, so a suppressed finding always
says exactly what fell short and by how much.

No new tuned thresholds live here. Every value is either the repetition bar
the findings design fixed (3), an existing engine constant reused for the
same meaning it already has elsewhere, or a structural rule (a peak at a
window edge is not a located peak):

  - MIN_REPETITIONS = 3. Kept even though it yields few findings on the
    current corpus -- lowering it would manufacture findings, not find them.
  - MIN_SAMPLE_CONFIDENCE reuses MIN_LANDMARK_VISIBILITY (0.5), the same gate
    the angle calculators and the visibility filter already apply.
  - The longest-invalid-run gate is expressed in milliseconds via Part 0's
    SWING_MIN_WINDOW_MS, converted to frames per clip with
    engine.phases.frame_timing -- never a hardcoded frame count. A gap at
    least as long as the shortest motion the swing detector accepts as a
    swing could hide a whole movement, so a peak found around it can't be
    trusted.
"""

from __future__ import annotations

from typing import Final, Sequence

from engine.biomechanics.posture.angle_calculator import MIN_LANDMARK_VISIBILITY
from engine.phases.contact_detection import SWING_MIN_WINDOW_MS
from engine.phases.frame_timing import ms_to_frames
from engine.types.findings import GateCheck
from engine.types.phases import DerivationMethod, PhaseBoundary

MIN_REPETITIONS: Final[int] = 3
MIN_SAMPLE_CONFIDENCE: Final[float] = MIN_LANDMARK_VISIBILITY
MAX_INVALID_RUN_MS: Final[float] = SWING_MIN_WINDOW_MS


def repetitions_gate(observed: int, gate: str = "usable_repetitions") -> GateCheck:
    return GateCheck(gate=gate, observed=observed, comparator=">=", required=MIN_REPETITIONS,
                     passed=observed >= MIN_REPETITIONS)


def sample_valid_gate(is_valid: bool, gate: str = "sample_valid") -> GateCheck:
    return GateCheck(gate=gate, observed=str(is_valid).lower(), comparator="==", required="true", passed=is_valid)


def sample_confidence_gate(confidence: float, gate: str = "sample_confidence") -> GateCheck:
    return GateCheck(gate=gate, observed=round(confidence, 4), comparator=">=", required=MIN_SAMPLE_CONFIDENCE,
                     passed=confidence >= MIN_SAMPLE_CONFIDENCE)


def longest_invalid_run(valid_flags: Sequence[bool]) -> int:
    longest = current = 0
    for ok in valid_flags:
        current = 0 if ok else current + 1
        longest = max(longest, current)
    return longest


def invalid_run_gate(valid_flags: Sequence[bool], ms_per_frame: float | None, gate: str) -> GateCheck:
    """Passes when the longest run of invalid samples in the window is
    strictly shorter than MAX_INVALID_RUN_MS, in this clip's own frames."""
    limit_frames = ms_to_frames(MAX_INVALID_RUN_MS, ms_per_frame)
    observed = longest_invalid_run(valid_flags)
    return GateCheck(gate=gate, observed=observed, comparator="<", required=limit_frames, passed=observed < limit_frames)


def peak_not_at_edge_gate(peak_position: int, window_length: int, gate: str) -> GateCheck:
    """A maximum on the first or last sample of a window may just be the
    window cutting off a larger value outside it -- not a located peak."""
    at_edge = peak_position == 0 or peak_position == window_length - 1
    return GateCheck(gate=gate, observed="window_edge" if at_edge else "interior", comparator="==",
                     required="interior", passed=not at_edge)


def boundary_detected_gate(boundary: PhaseBoundary, gate: str) -> GateCheck:
    observed = boundary.derivation_method.value
    return GateCheck(gate=gate, observed=observed, comparator="==", required=DerivationMethod.DETECTED.value,
                     passed=boundary.derivation_method is DerivationMethod.DETECTED)


def boundary_not_widened_gate(boundary: PhaseBoundary, gate: str) -> GateCheck:
    return GateCheck(gate=gate, observed=str(boundary.search_range_widened).lower(), comparator="==",
                     required="false", passed=not boundary.search_range_widened)
