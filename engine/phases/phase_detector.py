"""Swing phase segmentation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from engine.biomechanics.kinematics.derivatives import DerivativeConfig, DerivativeMethod
from engine.biomechanics.kinematics.velocity import VelocityCalculator
from engine.biomechanics.posture.shoulder_rotation import ShoulderRotationCalculator
from engine.phases.contact_detection import (
    CONTACT_RULES,
    contact_candidates_in_windows,
    infer_racket_side,
    segment_swing_windows,
    wrist_speed_series,
)
from engine.types.biomechanics import VelocityMeasurement
from engine.types.landmarks import LandmarkFrame
from engine.types.phases import DerivationMethod, PhaseLabel, PhaseSegment
from engine.utils.geometry import dot_product, magnitude


class PhaseDetector(ABC):
    @abstractmethod
    def detect(self, frames: tuple[LandmarkFrame, ...]) -> tuple[PhaseSegment, ...]: ...


def group_by_swing(segments: tuple[PhaseSegment, ...]) -> tuple[tuple[PhaseSegment, ...], ...]:
    """detect() stays exactly ABC-compliant -- one flat tuple[PhaseSegment,
    ...], not a tuple-of-tuples -- even for a multi-swing clip: it
    concatenates each detected swing's 6 segments back to back, in
    chronological order. Multi-swing structure is recovered here, outside
    the contract, rather than by changing detect()'s return type: every
    recurrence of PhaseLabel.READY after the first segment marks the start
    of another swing, since READY is always exactly the first segment of
    whatever swing produced it."""
    groups: list[list[PhaseSegment]] = []
    for segment in segments:
        if not groups or segment.label == PhaseLabel.READY:
            groups.append([])
        groups[-1].append(segment)
    return tuple(tuple(g) for g in groups)


_DERIVATIVE_CONFIG = DerivativeConfig(method=DerivativeMethod.CENTRAL_DIFFERENCE)

# Boundary order matches the label schema this is validated against
# (tools/label.py / labels/<clip_id>.json), modulo naming: that schema calls
# the first phase "prep", the engine's own PhaseLabel calls it READY -- same
# phase, different name coined in a separate pass. tools/eval_phases.py maps
# between them; not reconciled here since PhaseLabel is the existing,
# untouched contract this implements against.
_PHASE_ORDER: tuple[PhaseLabel, ...] = (
    PhaseLabel.READY,
    PhaseLabel.BACKSWING,
    PhaseLabel.FORWARD_SWING,
    PhaseLabel.CONTACT,
    PhaseLabel.FOLLOW_THROUGH,
    PhaseLabel.RECOVERY,
)


@dataclass(frozen=True)
class DetectedSwing:
    """One swing as KinematicPhaseDetector found it: its 6 segments (in
    phase order) plus how each non-contact boundary was derived, recorded at
    the moment _segment_one_swing chose it -- computed once, by the code that
    made the decision, not reverse-engineered afterwards.

    derivation keys: READY, BACKSWING, FORWARD_SWING, FOLLOW_THROUGH,
    RECOVERY (contact is always the detected primary signal). A boundary is
    DETECTED when its own signal search produced the pre-clamp value,
    ARCHITECTURAL when a fallback (range edge / contact+1) did. READY is
    always ARCHITECTURAL (range_start, never searched). FORWARD_SWING is
    always UNRELIABLE whichever of its two methods fired -- a policy from
    validation against human review, not a property of the run (see
    DerivationMethod's docstring)."""

    segments: tuple[PhaseSegment, ...]
    range_start: int
    range_end: int
    contact_frame: int
    derivation: Mapping[PhaseLabel, DerivationMethod]


def build_search_ranges(first_frame_index: int, last_frame_index: int, windows) -> list[tuple[int, int]]:
    """Partitions a clip into one contiguous, non-overlapping search range
    per swing window -- each swing gets exclusive claim to the frames around
    it, out to the midpoint with its neighbor or the clip edge at either end.
    Public so callers needing the same construction for other window sets
    (analysis_result_builder's unfiltered-window comparison) use this exact
    code rather than a copy."""
    ranges: list[tuple[int, int]] = []
    for i, window in enumerate(windows):
        range_start = first_frame_index if i == 0 else ranges[-1][1] + 1
        range_end = (window.end_frame + windows[i + 1].start_frame) // 2 if i + 1 < len(windows) else last_frame_index
        ranges.append((range_start, range_end))
    return ranges


class KinematicPhaseDetector(PhaseDetector):
    """Concrete PhaseDetector, multi-swing aware. Contact candidates come
    from racket-side wrist kinematics (engine.phases.contact_detection),
    now found independently per candidate swing window
    (segment_swing_windows) rather than once globally -- a clip containing
    several swings no longer has its rules picking extrema from different
    repetitions of the swing against each other. The other five boundaries
    per swing are placed relative to that swing's own contact, searched only
    within that swing's own share of the clip (the midpoint to its
    neighboring windows, or the clip edges for the first/last swing), using
    two signal types:

      - wrist speed thresholds, treated as a "departure from / return to
        rest" crossing (READY->BACKSWING, FOLLOW_THROUGH->RECOVERY) and a
        directional velocity zero-crossing along the contact-moment
        direction (FOLLOW_THROUGH start);
      - the shoulder-rotation extremum between the start of the backswing
        and contact (top of the backswing / peak trunk coil), for
        BACKSWING->FORWARD_SWING.

    detect() still returns a single flat tuple[PhaseSegment, ...] -- the
    existing ABC contract is unchanged -- with every detected swing's 6
    segments concatenated in order; see group_by_swing() to split it back
    into one group per swing.

    Untuned: rest_speed_fraction/follow_through_speed_fraction below, and
    the window-segmentation constants in contact_detection.py, are starting
    guesses, not fit against labels/. See tools/eval_phases.py for where
    they stand against whatever's labelled before touching them.
    """

    def __init__(
        self,
        contact_rule: str = "peak_speed",
        rest_speed_fraction: float = 0.2,
        follow_through_speed_fraction: float = 0.7,
    ) -> None:
        if contact_rule not in CONTACT_RULES:
            raise ValueError(f"Unknown contact_rule {contact_rule!r}; choose from {sorted(CONTACT_RULES)}")
        self._contact_rule = contact_rule
        self._rest_speed_fraction = rest_speed_fraction
        self._follow_through_speed_fraction = follow_through_speed_fraction

    def detect(self, frames: tuple[LandmarkFrame, ...]) -> tuple[PhaseSegment, ...]:
        return tuple(segment for swing in self.detect_swings(frames) for segment in swing.segments)

    def detect_swings(self, frames: tuple[LandmarkFrame, ...]) -> tuple[DetectedSwing, ...]:
        """The structured form of detect(): one DetectedSwing per swing,
        carrying each boundary's derivation method. detect() is exactly this,
        flattened -- one code path, so the two can't disagree."""
        if not frames:
            return ()

        wrist = infer_racket_side(frames)
        speeds = wrist_speed_series(frames, wrist)
        windows = segment_swing_windows(speeds)
        if not windows:
            return ()

        velocities = self._wrist_velocities(frames, wrist)
        rotations = ShoulderRotationCalculator()
        rotation_by_index = {f.timing.frame_index: rotations.calculate(f) for f in frames}

        ranges = build_search_ranges(frames[0].timing.frame_index, frames[-1].timing.frame_index, windows)
        contacts = contact_candidates_in_windows(speeds, windows, CONTACT_RULES[self._contact_rule])

        swings: list[DetectedSwing] = []
        for (range_start, range_end), contact in zip(ranges, contacts):
            if contact is None:
                continue  # this window produced no usable candidate -- no swing recorded for it
            swings.append(
                self._segment_one_swing(speeds, velocities, rotation_by_index, range_start, range_end, contact.frame_index)
            )
        return tuple(swings)

    def _segment_one_swing(
        self, speeds, velocities, rotation_by_index, range_start: int, range_end: int, contact_frame: int
    ) -> DetectedSwing:
        baseline_speed = self._baseline_speed(speeds, range_start, range_end)
        pre_contact_peak = self._peak_speed_in_range(speeds, range_start, contact_frame)
        threshold = baseline_speed + self._rest_speed_fraction * max(pre_contact_peak - baseline_speed, 0.0)

        derivation = {PhaseLabel.READY: DerivationMethod.ARCHITECTURAL, PhaseLabel.FORWARD_SWING: DerivationMethod.UNRELIABLE}

        backswing_start = self._first_crossing_up(speeds, range_start, contact_frame, threshold)
        derivation[PhaseLabel.BACKSWING] = DerivationMethod.DETECTED
        if backswing_start is None:
            backswing_start = range_start
            derivation[PhaseLabel.BACKSWING] = DerivationMethod.ARCHITECTURAL

        forward_swing_start = self._shoulder_rotation_extremum(rotation_by_index, backswing_start, contact_frame)
        if forward_swing_start is None:
            forward_swing_start = self._min_speed_frame(speeds, backswing_start, contact_frame)
        if forward_swing_start is None or forward_swing_start < backswing_start:
            forward_swing_start = backswing_start

        follow_through_start = self._direction_reversal_after_contact(velocities, contact_frame, range_end)
        if follow_through_start is None:
            contact_peak = pre_contact_peak if pre_contact_peak > 0 else self._peak_speed_in_range(
                speeds, contact_frame, range_end
            )
            follow_through_start = self._first_crossing_down(
                speeds, contact_frame, range_end, self._follow_through_speed_fraction * contact_peak
            )
        derivation[PhaseLabel.FOLLOW_THROUGH] = DerivationMethod.DETECTED
        if follow_through_start is None or follow_through_start <= contact_frame:
            follow_through_start = min(contact_frame + 1, range_end)
            derivation[PhaseLabel.FOLLOW_THROUGH] = DerivationMethod.ARCHITECTURAL

        recovery_start = self._first_crossing_down(speeds, follow_through_start, range_end, threshold)
        derivation[PhaseLabel.RECOVERY] = DerivationMethod.DETECTED
        if recovery_start is None or recovery_start <= follow_through_start:
            recovery_start = range_end
            derivation[PhaseLabel.RECOVERY] = DerivationMethod.ARCHITECTURAL

        starts = {
            PhaseLabel.READY: range_start,
            PhaseLabel.BACKSWING: max(backswing_start, range_start),
            PhaseLabel.FORWARD_SWING: max(forward_swing_start, range_start),
            PhaseLabel.CONTACT: contact_frame,
            PhaseLabel.FOLLOW_THROUGH: follow_through_start,
            PhaseLabel.RECOVERY: recovery_start,
        }
        # Enforce monotonic, non-decreasing boundaries within this swing's
        # own range -- upstream heuristics can disagree on ordering in
        # noisy/short windows; later phases never get pulled earlier than an
        # already-placed one, and never spill past this swing's own range.
        ordered_starts = []
        running_max = range_start
        for label in _PHASE_ORDER:
            start = min(max(starts[label], running_max), range_end)
            ordered_starts.append(start)
            running_max = start

        segments = []
        for i, label in enumerate(_PHASE_ORDER):
            start = ordered_starts[i]
            end = (ordered_starts[i + 1] - 1) if i + 1 < len(_PHASE_ORDER) else range_end
            # end can land below start when two boundaries collapse onto the
            # same frame (e.g. no room was found between two neighbors) --
            # left as a genuinely empty (zero-frame) segment rather than
            # clamped up to steal a frame from the next phase, which would
            # break strict contiguity (later.start == earlier.end + 1).
            segments.append(PhaseSegment(label=label, start_frame_index=start, end_frame_index=end))
        return DetectedSwing(segments=tuple(segments), range_start=range_start, range_end=range_end,
                             contact_frame=contact_frame, derivation=MappingProxyType(derivation))

    @staticmethod
    def _wrist_velocities(frames: tuple[LandmarkFrame, ...], wrist) -> tuple[VelocityMeasurement, ...]:
        position_samples = [
            (f.timing, f.pose_landmarks[wrist].position if wrist in f.pose_landmarks else None) for f in frames
        ]
        return VelocityCalculator().compute(position_samples, _DERIVATIVE_CONFIG)

    @staticmethod
    def _baseline_speed(speeds, range_start: int, range_end: int) -> float:
        window_end = min(range_start + 10, range_end)
        early = [s.speed for s in speeds if s.speed is not None and range_start <= s.frame_index <= window_end]
        return min(early) if early else 0.0

    @staticmethod
    def _peak_speed_in_range(speeds, start: int, end: int) -> float:
        values = [s.speed for s in speeds if s.speed is not None and start <= s.frame_index <= end]
        return max(values, default=0.0)

    @staticmethod
    def _min_speed_frame(speeds, start: int, end: int) -> int | None:
        candidates = [s for s in speeds if s.speed is not None and start <= s.frame_index <= end]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.speed).frame_index

    @staticmethod
    def _first_crossing_up(speeds, start: int, end: int, threshold: float) -> int | None:
        for s in speeds:
            if start <= s.frame_index <= end and s.speed is not None and s.speed >= threshold:
                return s.frame_index
        return None

    @staticmethod
    def _first_crossing_down(speeds, start: int, end: int, threshold: float) -> int | None:
        for s in speeds:
            if start < s.frame_index <= end and s.speed is not None and s.speed <= threshold:
                return s.frame_index
        return None

    @staticmethod
    def _shoulder_rotation_extremum(rotation_by_index: dict, start: int, end: int) -> int | None:
        baseline = rotation_by_index.get(start)
        if baseline is None or not baseline.is_valid or baseline.angle_degrees is None:
            return None
        best_frame = None
        best_deviation = -1.0
        for frame_index, measurement in rotation_by_index.items():
            if not (start <= frame_index <= end):
                continue
            if not measurement.is_valid or measurement.angle_degrees is None:
                continue
            deviation = abs(measurement.angle_degrees - baseline.angle_degrees)
            if deviation > best_deviation:
                best_deviation = deviation
                best_frame = frame_index
        return best_frame

    @staticmethod
    def _direction_reversal_after_contact(
        velocities: tuple[VelocityMeasurement, ...], contact_frame: int, range_end: int
    ) -> int | None:
        contact_velocity = next(
            (v for v in velocities if v.frame_index == contact_frame and v.is_valid and v.velocity is not None), None
        )
        if contact_velocity is None:
            return None
        direction_mag = magnitude(contact_velocity.velocity)
        if direction_mag <= 0:
            return None
        was_positive = False
        for v in velocities:
            if v.frame_index <= contact_frame or v.frame_index > range_end or not v.is_valid or v.velocity is None:
                continue
            projection = dot_product(v.velocity, contact_velocity.velocity) / direction_mag
            if v.frame_index == contact_frame + 1:
                was_positive = projection > 0
                continue
            if was_positive and projection <= 0:
                return v.frame_index
            was_positive = projection > 0
        return None
