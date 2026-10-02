"""Assembles a real AnalysisResult from a real KinematicPhaseDetector.detect()
call's output plus the frames it ran against.

Backend/API wiring, built on top of engine/phases and engine/types/phases --
deliberately kept separate from KinematicPhaseDetector itself rather than
extending detect()'s return type, since the existing PhaseDetector.detect()
ABC contract is fixed (group_by_swing's own docstring explains why: multi-
swing structure is recovered outside the contract too, same principle here).

Classifying each boundary's derivation_method means re-deriving which of
_segment_one_swing's own candidate signals actually fired, before any
fallback -- something detect()'s flat PhaseSegment output doesn't expose.
This is done by calling KinematicPhaseDetector's existing (underscore-
prefixed, "private" by convention) static helper methods a second time with
the same inputs: not a reimplementation of the decision logic, but a second
call path into the same tested functions. It IS tightly coupled to
_segment_one_swing's exact branch structure, though, and would need
updating in lockstep if that method's fallback chain changes shape (not
just its threshold constants) -- a real, deliberate maintenance-coupling
trade-off, not an oversight.
"""

from __future__ import annotations

from engine.api.interfaces import AnalysisResult
from engine.phases.contact_detection import (
    CONTACT_RULES,
    contact_candidates_in_windows,
    infer_racket_side,
    segment_swing_windows,
    wrist_speed_series,
)
from engine.phases.phase_detector import KinematicPhaseDetector, group_by_swing
from engine.types.landmarks import LandmarkFrame
from engine.types.phases import ContactRule, DerivationMethod, PhaseBoundary, PhaseLabel, PhaseSegment, SwingPhases

# The "no length-based filtering at all" baseline for detecting whether a
# swing's phase-search range was widened by SWING_MIN_WINDOW_MS removing a
# neighboring window -- the general form of what the neighbor-widening audit
# measured (docs/STATUS.md engine/phases known limitations, 2026-08-29),
# which compared the specific pre-Part-0 value (5 frames) against the
# then-current one (13 frames). Using 1 here instead of hardcoding a frame
# count means this keeps working correctly however SWING_MIN_WINDOW_MS is
# tuned later, and however a given clip's own frame rate converts it,
# rather than silently comparing against a stale constant forever.
_UNFILTERED_MIN_WINDOW_FRAMES = 1


def _build_ranges(frames: tuple[LandmarkFrame, ...], windows) -> list[tuple[int, int]]:
    """Identical construction to KinematicPhaseDetector.detect()'s own
    ranges list -- duplicated rather than imported since detect() builds it
    as a local, not something the class exposes."""
    if not windows:
        return []
    first_frame_index = frames[0].timing.frame_index
    last_frame_index = frames[-1].timing.frame_index
    ranges: list[tuple[int, int]] = []
    for i, window in enumerate(windows):
        range_start = first_frame_index if i == 0 else ranges[-1][1] + 1
        range_end = (window.end_frame + windows[i + 1].start_frame) // 2 if i + 1 < len(windows) else last_frame_index
        ranges.append((range_start, range_end))
    return ranges


def _classify_backswing(speeds, range_start: int, contact_frame: int, threshold: float) -> DerivationMethod:
    found = KinematicPhaseDetector._first_crossing_up(speeds, range_start, contact_frame, threshold)
    return DerivationMethod.DETECTED if found is not None else DerivationMethod.ARCHITECTURAL


def _classify_follow_through(
    speeds, velocities, contact_frame: int, range_end: int, follow_through_speed_fraction: float, pre_contact_peak: float
) -> DerivationMethod:
    # Mirrors _segment_one_swing's own two-tier signal cascade exactly:
    # direction-reversal first, a speed-fraction crossing second, and only
    # if BOTH come up empty (or land at/before contact) does the real code
    # fall back to the architectural min(contact_frame+1, range_end) clamp.
    reversal = KinematicPhaseDetector._direction_reversal_after_contact(velocities, contact_frame, range_end)
    if reversal is not None:
        return DerivationMethod.DETECTED
    contact_peak = pre_contact_peak if pre_contact_peak > 0 else KinematicPhaseDetector._peak_speed_in_range(
        speeds, contact_frame, range_end
    )
    crossing = KinematicPhaseDetector._first_crossing_down(
        speeds, contact_frame, range_end, follow_through_speed_fraction * contact_peak
    )
    if crossing is not None and crossing > contact_frame:
        return DerivationMethod.DETECTED
    return DerivationMethod.ARCHITECTURAL


def _classify_recovery(speeds, follow_through_start: int, range_end: int, threshold: float) -> DerivationMethod:
    crossing = KinematicPhaseDetector._first_crossing_down(speeds, follow_through_start, range_end, threshold)
    if crossing is not None and crossing > follow_through_start:
        return DerivationMethod.DETECTED
    return DerivationMethod.ARCHITECTURAL


def build_analysis_result(
    frames: tuple[LandmarkFrame, ...],
    segments: tuple[PhaseSegment, ...],
    contact_rule: ContactRule,
    rest_speed_fraction: float = 0.2,
    follow_through_speed_fraction: float = 0.7,
) -> AnalysisResult:
    """`frames` must be the exact tuple `segments` was produced from (i.e.
    `segments = KinematicPhaseDetector(contact_rule.value).detect(frames)`)
    -- this recomputes the speed/velocity signals detect() used internally
    to classify each boundary; a mismatched pair would silently produce a
    wrong-but-plausible result, not an error. rest_speed_fraction/
    follow_through_speed_fraction must match whatever detector instance
    produced `segments` (both default to KinematicPhaseDetector's own
    defaults)."""
    swing_groups = group_by_swing(segments)
    if not swing_groups:
        return AnalysisResult(contact_rule=contact_rule, swings=())

    wrist = infer_racket_side(frames)
    speeds = wrist_speed_series(frames, wrist)
    velocities = KinematicPhaseDetector._wrist_velocities(frames, wrist)

    windows_filtered = segment_swing_windows(speeds)
    # min_rest_gap_frames left at its default (None -> derived from
    # SWING_MIN_REST_GAP_MS and this clip's own measured rate) so it matches
    # windows_filtered's rest-gap exactly -- only min_window_frames differs.
    windows_unfiltered = segment_swing_windows(speeds, min_window_frames=_UNFILTERED_MIN_WINDOW_FRAMES)
    ranges_filtered = _build_ranges(frames, windows_filtered)
    ranges_unfiltered = _build_ranges(frames, windows_unfiltered)
    unfiltered_range_by_window_key = {
        (w.start_frame, w.end_frame): ranges_unfiltered[i] for i, w in enumerate(windows_unfiltered)
    }

    contacts_filtered = contact_candidates_in_windows(speeds, windows_filtered, CONTACT_RULES[contact_rule.value])
    widening_by_contact_frame: dict[int, tuple[bool, bool]] = {}
    for i, c in enumerate(contacts_filtered):
        if c is None:
            continue
        window_key = (windows_filtered[i].start_frame, windows_filtered[i].end_frame)
        filtered_range = ranges_filtered[i]
        # A currently-detected window not found in the unfiltered pass would
        # mean length-only filtering somehow changed a surviving window's
        # own boundaries, which segment_swing_windows never does (filtering
        # only ever drops whole windows) -- shouldn't happen, but default to
        # "not widened" rather than raising on a real clip.
        unfiltered_range = unfiltered_range_by_window_key.get(window_key, filtered_range)
        widening_by_contact_frame[c.frame_index] = (
            filtered_range[0] != unfiltered_range[0],  # left_widened
            filtered_range[1] != unfiltered_range[1],  # right_widened
        )

    swings = []
    for group in swing_groups:
        by_label = {seg.label: seg for seg in group}
        contact_frame = by_label[PhaseLabel.CONTACT].start_frame_index
        range_start = by_label[PhaseLabel.READY].start_frame_index
        range_end = by_label[PhaseLabel.RECOVERY].end_frame_index
        follow_through_start = by_label[PhaseLabel.FOLLOW_THROUGH].start_frame_index

        left_widened, right_widened = widening_by_contact_frame.get(contact_frame, (False, False))

        baseline_speed = KinematicPhaseDetector._baseline_speed(speeds, range_start, range_end)
        pre_contact_peak = KinematicPhaseDetector._peak_speed_in_range(speeds, range_start, contact_frame)
        threshold = baseline_speed + rest_speed_fraction * max(pre_contact_peak - baseline_speed, 0.0)

        swings.append(SwingPhases(
            prep=PhaseBoundary(
                frame_index=by_label[PhaseLabel.READY].start_frame_index,
                derivation_method=DerivationMethod.ARCHITECTURAL,  # always -- range_start, never searched
                search_range_widened=left_widened,
            ),
            backswing=PhaseBoundary(
                frame_index=by_label[PhaseLabel.BACKSWING].start_frame_index,
                derivation_method=_classify_backswing(speeds, range_start, contact_frame, threshold),
                search_range_widened=left_widened,
            ),
            forward_swing=PhaseBoundary(
                frame_index=by_label[PhaseLabel.FORWARD_SWING].start_frame_index,
                derivation_method=DerivationMethod.UNRELIABLE,  # always -- see docs/STATUS.md, 2026-08-28/29
                search_range_widened=left_widened,
            ),
            contact_frame=contact_frame,
            follow_through=PhaseBoundary(
                frame_index=follow_through_start,
                derivation_method=_classify_follow_through(
                    speeds, velocities, contact_frame, range_end, follow_through_speed_fraction, pre_contact_peak
                ),
                search_range_widened=right_widened,
            ),
            recovery=PhaseBoundary(
                frame_index=by_label[PhaseLabel.RECOVERY].start_frame_index,
                derivation_method=_classify_recovery(speeds, follow_through_start, range_end, threshold),
                search_range_widened=right_widened,
            ),
        ))

    return AnalysisResult(contact_rule=contact_rule, swings=tuple(swings))
