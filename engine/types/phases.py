"""Swing phase contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PhaseLabel(Enum):
    READY = "ready"
    BACKSWING = "backswing"
    FORWARD_SWING = "forward_swing"
    CONTACT = "contact"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class PhaseSegment:
    label: PhaseLabel
    start_frame_index: int
    end_frame_index: int


class DerivationMethod(Enum):
    """How a non-contact phase boundary's frame index was arrived at.

    Computed per swing, not fixed per phase, for backswing/follow_through/
    recovery: each has both a real detected path (threshold crossing,
    direction reversal) and an architectural fallback clamp in
    KinematicPhaseDetector._segment_one_swing for when that search comes up
    empty, and which one fired varies swing to swing. prep is the one
    constant case -- it's always range_start, never searched at all, so it
    always reports ARCHITECTURAL. forward_swing is a constant in the other
    direction: always UNRELIABLE, regardless of which of its own two
    internal methods (shoulder-rotation extremum or speed-minimum fallback)
    fired -- both were checked against real human corrections and neither
    is trustworthy (see docs/STATUS.md's engine/phases known limitations,
    2026-08-28/29), so the schema shouldn't imply a confidence the evidence
    doesn't support.
    """

    DETECTED = "detected"
    ARCHITECTURAL = "architectural"
    UNRELIABLE = "unreliable"


class ContactRule(Enum):
    """Mirrors engine.phases.contact_detection.CONTACT_RULES' keys exactly
    (see tests/engine_types/test_phases.py for the drift guard)."""

    PEAK_SPEED = "peak_speed"
    DECELERATION_ONSET = "deceleration_onset"


@dataclass(frozen=True)
class PhaseBoundary:
    frame_index: int | None
    derivation_method: DerivationMethod
    # Whether a neighboring window being filtered out by SWING_MIN_WINDOW_MS
    # widened this boundary's own search range beyond what it would otherwise
    # have been. Not speculative: confirmed 2026-08-29 across 5 of 6 labelled
    # clips (13 windows removed, ranges widened 12-112 frames, the removed
    # window's own speed peak reliably reappearing inside the widened range),
    # and follow_through accuracy on the 35-item human review split cleanly
    # on it (0/3 correct widened vs. 4/4 correct not-widened). See
    # docs/STATUS.md's engine/phases known limitations for the full evidence.
    # No production code populates this from a real clip yet -- that's the
    # detect()-to-AnalysisResult conversion, deferred along with the rest of
    # the backend/API wiring.
    search_range_widened: bool


@dataclass(frozen=True)
class SwingPhases:
    """One swing's full phase picture. Field names match labels/<clip>.json
    schema v2's phase_boundaries keys exactly (tools/label.py's PHASES),
    aside from contact_frame -- see AnalysisResult's docstring for why
    contact is pulled out of the uniform PhaseBoundary shape."""

    prep: PhaseBoundary
    backswing: PhaseBoundary
    forward_swing: PhaseBoundary
    contact_frame: int | None
    follow_through: PhaseBoundary
    recovery: PhaseBoundary
