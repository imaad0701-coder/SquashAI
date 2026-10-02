"""Tests for engine.phases.phase_detector.KinematicPhaseDetector."""

from __future__ import annotations

import unittest

from engine.phases.phase_detector import KinematicPhaseDetector, group_by_swing
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.phases import PhaseLabel
from engine.types.video import FrameTiming

_FPS = 30.0
_DT_MS = 1000.0 / _FPS
_SIX_PHASES = (
    PhaseLabel.READY,
    PhaseLabel.BACKSWING,
    PhaseLabel.FORWARD_SWING,
    PhaseLabel.CONTACT,
    PhaseLabel.FOLLOW_THROUGH,
    PhaseLabel.RECOVERY,
)


def _landmark(x: float, y: float = 0.0, z: float = 0.0) -> Landmark:
    return Landmark(position=Point3D(x=x, y=y, z=z), visibility=0.9, presence=0.9)


def _frames_with_swings(
    n: int, swing_ranges: tuple[tuple[int, int], ...], rest_step: float = 0.2, swing_step: float = 30.0
) -> tuple[LandmarkFrame, ...]:
    """A racket-side (right) wrist that creeps slowly at rest and moves much
    faster (constant velocity) during each (start, end) range in
    swing_ranges -- a clean, sustained rectangular speed pulse per swing,
    which is what segment_swing_windows needs to see (a single-frame spike
    gets correctly filtered as noise by SWING_MIN_WINDOW_MS, so the
    fixture has to look like a real swing, not an instantaneous jump). Left
    wrist barely moves, so it's unambiguously not the racket side. Shoulders
    drift slightly so ShoulderRotationCalculator has something to compute."""
    frames = []
    right_x = 0.0
    for i in range(n):
        in_swing = any(start <= i <= end for start, end in swing_ranges)
        right_x += swing_step if in_swing else rest_step
        timing = FrameTiming(frame_index=i, timestamp_ms=i * _DT_MS, delta_time_ms=_DT_MS)
        pose_landmarks = {
            PoseLandmarkName.LEFT_WRIST: _landmark(x=float(i) * 0.05),
            PoseLandmarkName.RIGHT_WRIST: _landmark(x=right_x),
            PoseLandmarkName.LEFT_SHOULDER: _landmark(x=-20.0, z=float(i) * 0.5),
            PoseLandmarkName.RIGHT_SHOULDER: _landmark(x=20.0, z=-float(i) * 0.5),
        }
        frames.append(LandmarkFrame(timing=timing, pose_landmarks=pose_landmarks))
    return tuple(frames)


def _single_swing_frames(n: int, swing_start: int, swing_end: int) -> tuple[LandmarkFrame, ...]:
    return _frames_with_swings(n, ((swing_start, swing_end),))


class KinematicPhaseDetectorContractTests(unittest.TestCase):
    def test_empty_frames_returns_empty_tuple(self) -> None:
        self.assertEqual(KinematicPhaseDetector().detect(()), ())

    def test_unknown_contact_rule_raises(self) -> None:
        with self.assertRaises(ValueError):
            KinematicPhaseDetector(contact_rule="not_a_real_rule")

    def test_returns_all_six_phases_in_order_for_a_single_swing(self) -> None:
        frames = _single_swing_frames(60, swing_start=20, swing_end=35)
        segments = KinematicPhaseDetector().detect(frames)
        self.assertEqual([s.label for s in segments], list(_SIX_PHASES))

    def test_segments_are_contiguous_and_cover_the_whole_clip(self) -> None:
        # Strict contiguity (later.start == earlier.end + 1) must hold even
        # when a phase ends up empty (start > end) because two boundaries
        # collapsed onto the same frame -- an empty segment is preferable to
        # clamping it non-empty, which would silently steal a frame from the
        # next phase and break contiguity instead.
        frames = _single_swing_frames(60, swing_start=20, swing_end=35)
        segments = KinematicPhaseDetector().detect(frames)

        self.assertEqual(segments[0].start_frame_index, 0)
        self.assertEqual(segments[-1].end_frame_index, 59)
        for earlier, later in zip(segments, segments[1:]):
            self.assertEqual(later.start_frame_index, earlier.end_frame_index + 1)

    def test_no_sustained_motion_returns_empty_tuple(self) -> None:
        # A single-frame jump (not sustained) must NOT register as a swing --
        # this is segment_swing_windows' noise filter (SWING_MIN_WINDOW_MS)
        # doing its job, not a detection failure.
        frames = _single_swing_frames(40, swing_start=20, swing_end=20)
        segments = KinematicPhaseDetector().detect(frames)
        self.assertEqual(segments, ())

    def test_contact_lands_inside_the_swing_window(self) -> None:
        frames = _single_swing_frames(60, swing_start=20, swing_end=35)
        segments = KinematicPhaseDetector().detect(frames)
        contact = next(s for s in segments if s.label == PhaseLabel.CONTACT)
        self.assertTrue(18 <= contact.start_frame_index <= 37, contact.start_frame_index)

    def test_deceleration_onset_rule_also_produces_a_complete_valid_segmentation(self) -> None:
        frames = _single_swing_frames(60, swing_start=20, swing_end=35)
        segments = KinematicPhaseDetector(contact_rule="deceleration_onset").detect(frames)
        self.assertEqual(len(segments), 6)
        self.assertEqual(segments[0].start_frame_index, 0)
        self.assertEqual(segments[-1].end_frame_index, 59)

    def test_no_detectable_motion_returns_empty_tuple_rather_than_raising(self) -> None:
        frames = tuple(
            LandmarkFrame(
                timing=FrameTiming(frame_index=i, timestamp_ms=i * _DT_MS, delta_time_ms=_DT_MS),
                pose_landmarks={},
            )
            for i in range(10)
        )
        self.assertEqual(KinematicPhaseDetector().detect(frames), ())


class MultiSwingTests(unittest.TestCase):
    """Two clean swings separated by a rest gap well past
    SWING_MIN_REST_GAP_MS (500ms = 15 frames at 30fps) -- must be detected as two independent
    swings, each internally well-formed, not one blended segmentation."""

    def _two_swing_frames(self) -> tuple[LandmarkFrame, ...]:
        return _frames_with_swings(160, ((10, 25), (90, 105)))

    def test_detects_two_swings(self) -> None:
        segments = KinematicPhaseDetector().detect(self._two_swing_frames())
        swings = group_by_swing(segments)
        self.assertEqual(len(swings), 2)

    def test_each_swing_has_all_six_phases_in_order(self) -> None:
        segments = KinematicPhaseDetector().detect(self._two_swing_frames())
        for swing in group_by_swing(segments):
            self.assertEqual([s.label for s in swing], list(_SIX_PHASES))

    def test_contact_frames_land_in_their_own_windows_not_swapped(self) -> None:
        segments = KinematicPhaseDetector().detect(self._two_swing_frames())
        swings = group_by_swing(segments)
        contact_0 = next(s for s in swings[0] if s.label == PhaseLabel.CONTACT).start_frame_index
        contact_1 = next(s for s in swings[1] if s.label == PhaseLabel.CONTACT).start_frame_index
        self.assertTrue(5 <= contact_0 <= 30, contact_0)
        self.assertTrue(85 <= contact_1 <= 110, contact_1)

    def test_all_segments_across_both_swings_stay_contiguous_and_cover_the_clip(self) -> None:
        frames = self._two_swing_frames()
        segments = KinematicPhaseDetector().detect(frames)
        self.assertEqual(segments[0].start_frame_index, 0)
        self.assertEqual(segments[-1].end_frame_index, frames[-1].timing.frame_index)
        for earlier, later in zip(segments, segments[1:]):
            self.assertEqual(later.start_frame_index, earlier.end_frame_index + 1)


class NativeDerivationTests(unittest.TestCase):
    """detect_swings() carries each boundary's derivation method, recorded by
    _segment_one_swing when it decides -- the structured source of truth
    analysis_result_builder now consumes instead of re-running the
    detector's private signal searches."""

    def _two_swing_frames(self) -> tuple[LandmarkFrame, ...]:
        return _frames_with_swings(160, ((10, 25), (90, 105)))

    def test_detect_is_exactly_flattened_detect_swings(self) -> None:
        frames = self._two_swing_frames()
        for rule in ("peak_speed", "deceleration_onset"):
            detector = KinematicPhaseDetector(contact_rule=rule)
            flattened = tuple(seg for swing in detector.detect_swings(frames) for seg in swing.segments)
            self.assertEqual(detector.detect(frames), flattened)

    def test_every_swing_records_a_derivation_for_each_non_contact_boundary(self) -> None:
        from engine.types.phases import DerivationMethod

        for swing in KinematicPhaseDetector().detect_swings(self._two_swing_frames()):
            self.assertEqual(set(swing.derivation),
                             {PhaseLabel.READY, PhaseLabel.BACKSWING, PhaseLabel.FORWARD_SWING,
                              PhaseLabel.FOLLOW_THROUGH, PhaseLabel.RECOVERY})
            self.assertIs(swing.derivation[PhaseLabel.READY], DerivationMethod.ARCHITECTURAL)
            self.assertIs(swing.derivation[PhaseLabel.FORWARD_SWING], DerivationMethod.UNRELIABLE)
            self.assertEqual(swing.contact_frame,
                             next(s for s in swing.segments if s.label == PhaseLabel.CONTACT).start_frame_index)

    def test_clean_pulse_backswing_is_detected(self) -> None:
        from engine.types.phases import DerivationMethod

        swing = KinematicPhaseDetector().detect_swings(_single_swing_frames(80, 20, 40))[0]
        self.assertIs(swing.derivation[PhaseLabel.BACKSWING], DerivationMethod.DETECTED)

    def test_builder_never_calls_detector_private_methods(self) -> None:
        import inspect

        import engine.phases.analysis_result_builder as builder

        source = inspect.getsource(builder)
        self.assertNotIn("KinematicPhaseDetector._", source)
        self.assertNotIn("_segment_one_swing(", source)


class GroupBySwingTests(unittest.TestCase):
    def test_empty_input_returns_empty_tuple(self) -> None:
        self.assertEqual(group_by_swing(()), ())

    def test_splits_on_every_ready_recurrence(self) -> None:
        from engine.types.phases import PhaseSegment

        segments = (
            PhaseSegment(PhaseLabel.READY, 0, 1),
            PhaseSegment(PhaseLabel.BACKSWING, 2, 3),
            PhaseSegment(PhaseLabel.READY, 4, 5),
            PhaseSegment(PhaseLabel.BACKSWING, 6, 7),
        )
        groups = group_by_swing(segments)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0], segments[:2])
        self.assertEqual(groups[1], segments[2:])


if __name__ == "__main__":
    unittest.main()
