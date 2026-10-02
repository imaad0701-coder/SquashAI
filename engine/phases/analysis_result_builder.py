"""Assembles an AnalysisResult from KinematicPhaseDetector.detect_swings()
output plus the frames it ran against.

Each boundary's derivation_method comes straight from the detector: its
_segment_one_swing records which signal (or which fallback) placed every
boundary at the moment it decides, and detect_swings() returns that as
DetectedSwing.derivation. Nothing here re-runs the detector's signal
searches or calls its private helpers -- an earlier version did, which
coupled this module to _segment_one_swing's exact branch structure and could
silently drift from what detection actually did.

search_range_widened is computed here, from public functions only: it
compares the detector's real (length-filtered) swing windows with the
windows an unfiltered pass would produce, which is analysis metadata about
the detection, not part of it.
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
from engine.phases.phase_detector import DetectedSwing, build_search_ranges
from engine.types.landmarks import LandmarkFrame
from engine.types.phases import ContactRule, PhaseBoundary, PhaseLabel, SwingPhases

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


def _widening_by_contact_frame(frames: tuple[LandmarkFrame, ...], contact_rule: ContactRule) -> dict[int, tuple[bool, bool]]:
    first, last = frames[0].timing.frame_index, frames[-1].timing.frame_index
    speeds = wrist_speed_series(frames, infer_racket_side(frames))
    windows_filtered = segment_swing_windows(speeds)
    # min_rest_gap_frames left at its default (None -> derived from
    # SWING_MIN_REST_GAP_MS and this clip's own measured rate) so it matches
    # windows_filtered's rest-gap exactly -- only min_window_frames differs.
    windows_unfiltered = segment_swing_windows(speeds, min_window_frames=_UNFILTERED_MIN_WINDOW_FRAMES)
    ranges_filtered = build_search_ranges(first, last, windows_filtered)
    ranges_unfiltered = build_search_ranges(first, last, windows_unfiltered)
    unfiltered_range_by_window_key = {
        (w.start_frame, w.end_frame): ranges_unfiltered[i] for i, w in enumerate(windows_unfiltered)
    }
    widening: dict[int, tuple[bool, bool]] = {}
    for i, c in enumerate(contact_candidates_in_windows(speeds, windows_filtered, CONTACT_RULES[contact_rule.value])):
        if c is None:
            continue
        filtered_range = ranges_filtered[i]
        # A currently-detected window not found in the unfiltered pass would
        # mean length-only filtering somehow changed a surviving window's
        # own boundaries, which segment_swing_windows never does (filtering
        # only ever drops whole windows) -- shouldn't happen, but default to
        # "not widened" rather than raising on a real clip.
        unfiltered_range = unfiltered_range_by_window_key.get(
            (windows_filtered[i].start_frame, windows_filtered[i].end_frame), filtered_range)
        widening[c.frame_index] = (filtered_range[0] != unfiltered_range[0], filtered_range[1] != unfiltered_range[1])
    return widening


def build_analysis_result(
    frames: tuple[LandmarkFrame, ...],
    swings: tuple[DetectedSwing, ...],
    contact_rule: ContactRule,
) -> AnalysisResult:
    """`swings` must come from `KinematicPhaseDetector(contact_rule.value).detect_swings(frames)`
    over these same `frames` (the widening comparison re-derives swing
    windows from them)."""
    if not swings:
        return AnalysisResult(contact_rule=contact_rule, swings=())

    widening = _widening_by_contact_frame(frames, contact_rule)
    out = []
    for swing in swings:
        start = {seg.label: seg.start_frame_index for seg in swing.segments}
        left_widened, right_widened = widening.get(swing.contact_frame, (False, False))

        def boundary(label: PhaseLabel, widened: bool) -> PhaseBoundary:
            return PhaseBoundary(frame_index=start[label], derivation_method=swing.derivation[label],
                                 search_range_widened=widened)

        out.append(SwingPhases(
            prep=boundary(PhaseLabel.READY, left_widened),
            backswing=boundary(PhaseLabel.BACKSWING, left_widened),
            forward_swing=boundary(PhaseLabel.FORWARD_SWING, left_widened),
            contact_frame=swing.contact_frame,
            follow_through=boundary(PhaseLabel.FOLLOW_THROUGH, right_widened),
            recovery=boundary(PhaseLabel.RECOVERY, right_widened),
        ))
    return AnalysisResult(contact_rule=contact_rule, swings=tuple(out))
