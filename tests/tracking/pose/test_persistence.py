"""Tests for MissedFramePersistence."""

from __future__ import annotations

import unittest

from engine.tracking.pose.persistence import (
    ExponentialConfidenceDecay,
    LinearConfidenceDecay,
    MissedFramePersistence,
)
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import FrameTiming

_NOSE_LANDMARK = Landmark(position=Point3D(x=1.0, y=2.0, z=3.0), visibility=0.9, presence=0.9)


def _frame(index: int, with_nose: bool) -> LandmarkFrame:
    pose_landmarks = {PoseLandmarkName.NOSE: _NOSE_LANDMARK} if with_nose else {}
    timing = FrameTiming(frame_index=index, timestamp_ms=index * 100.0, delta_time_ms=0.0 if index == 0 else 100.0)
    return LandmarkFrame(timing=timing, pose_landmarks=pose_landmarks)


class MissedFramePersistenceTests(unittest.TestCase):
    def test_carries_gap_forward_within_budget_then_drops_beyond_it(self) -> None:
        frames = tuple(
            _frame(i, with_nose)
            for i, with_nose in enumerate([True, False, False, False, True])
        )

        result = MissedFramePersistence().apply(frames, max_missed_frames=2)

        self.assertIn(PoseLandmarkName.NOSE, result[0].pose_landmarks)  # real detection
        self.assertIn(PoseLandmarkName.NOSE, result[1].pose_landmarks)  # gap 1: carried
        self.assertIn(PoseLandmarkName.NOSE, result[2].pose_landmarks)  # gap 2: carried
        self.assertNotIn(PoseLandmarkName.NOSE, result[3].pose_landmarks)  # gap 3: budget exceeded
        self.assertIn(PoseLandmarkName.NOSE, result[4].pose_landmarks)  # real detection again

        # Position carries forward unchanged; confidence decays with each
        # consecutive held frame (default ExponentialConfidenceDecay, rate=0.75).
        held_1 = result[1].pose_landmarks[PoseLandmarkName.NOSE]
        held_2 = result[2].pose_landmarks[PoseLandmarkName.NOSE]
        self.assertEqual(held_1.position, _NOSE_LANDMARK.position)
        self.assertAlmostEqual(held_1.visibility, 0.9 * 0.75)
        self.assertAlmostEqual(held_1.presence, 0.9 * 0.75)
        self.assertEqual(held_2.position, _NOSE_LANDMARK.position)
        self.assertAlmostEqual(held_2.visibility, 0.9 * 0.75**2)
        self.assertAlmostEqual(held_2.presence, 0.9 * 0.75**2)

    def test_gap_counter_resets_when_detection_reappears(self) -> None:
        frames = tuple(
            _frame(i, with_nose)
            for i, with_nose in enumerate([True, False, False, False, True, False])
        )

        result = MissedFramePersistence().apply(frames, max_missed_frames=2)

        # frame5 follows immediately after a fresh real detection at frame4,
        # so its single-frame gap should be filled even though frame3's gap
        # (a different, earlier run of misses) had already exceeded budget.
        self.assertIn(PoseLandmarkName.NOSE, result[5].pose_landmarks)

    def test_landmark_never_seen_stays_absent_throughout(self) -> None:
        frames = tuple(_frame(i, with_nose=False) for i in range(4))

        result = MissedFramePersistence().apply(frames, max_missed_frames=5)

        for landmark_frame in result:
            self.assertNotIn(PoseLandmarkName.NOSE, landmark_frame.pose_landmarks)

    def test_zero_max_missed_frames_never_carries_forward(self) -> None:
        frames = tuple(_frame(i, with_nose) for i, with_nose in enumerate([True, False]))

        result = MissedFramePersistence().apply(frames, max_missed_frames=0)

        self.assertNotIn(PoseLandmarkName.NOSE, result[1].pose_landmarks)

    def test_preserves_timing_metadata(self) -> None:
        frames = (_frame(3, with_nose=True),)

        result = MissedFramePersistence().apply(frames, max_missed_frames=1)

        self.assertEqual(result[0].timing, frames[0].timing)


class ConfidenceDecayModelTests(unittest.TestCase):
    def test_exponential_decay_is_zero_frames_held_is_identity(self) -> None:
        self.assertEqual(ExponentialConfidenceDecay(rate=0.75).decay(0.9, 0), 0.9)

    def test_exponential_decay_compounds_per_frame_held(self) -> None:
        model = ExponentialConfidenceDecay(rate=0.75)
        self.assertAlmostEqual(model.decay(0.9, 1), 0.9 * 0.75)
        self.assertAlmostEqual(model.decay(0.9, 2), 0.9 * 0.75**2)
        self.assertAlmostEqual(model.decay(0.9, 5), 0.9 * 0.75**5)

    def test_exponential_decay_never_recovers_above_base(self) -> None:
        model = ExponentialConfidenceDecay(rate=0.75)
        prev = 0.9
        for k in range(1, 10):
            cur = model.decay(0.9, k)
            self.assertLessEqual(cur, prev)
            prev = cur

    def test_linear_decay_subtracts_a_flat_penalty_per_frame(self) -> None:
        model = LinearConfidenceDecay(per_frame_penalty=0.15)
        self.assertAlmostEqual(model.decay(0.9, 1), 0.75)
        self.assertAlmostEqual(model.decay(0.9, 2), 0.6)

    def test_linear_decay_floors_at_zero_rather_than_going_negative(self) -> None:
        model = LinearConfidenceDecay(per_frame_penalty=0.15)
        self.assertEqual(model.decay(0.9, 100), 0.0)


class MissedFramePersistenceConfidenceDecayTests(unittest.TestCase):
    """A held landmark's confidence must decay with each consecutive frame
    it's carried, so a stale, unconfirmed guess is no longer indistinguishable
    from a fresh detection to anything downstream reading confidence."""

    def test_default_model_is_exponential_and_strictly_decreasing_across_a_hold(self) -> None:
        frames = tuple(
            _frame(i, with_nose) for i, with_nose in enumerate([True, False, False, False, False])
        )

        result = MissedFramePersistence().apply(frames, max_missed_frames=4)

        confidences = [result[i].pose_landmarks[PoseLandmarkName.NOSE].visibility for i in range(5)]
        self.assertEqual(confidences[0], 0.9)  # fresh detection: unchanged
        for earlier, later in zip(confidences, confidences[1:]):
            self.assertLess(later, earlier)

    def test_confidence_resets_to_full_when_a_fresh_detection_reappears(self) -> None:
        frames = tuple(
            _frame(i, with_nose) for i, with_nose in enumerate([True, False, False, True])
        )

        result = MissedFramePersistence().apply(frames, max_missed_frames=3)

        self.assertEqual(result[3].pose_landmarks[PoseLandmarkName.NOSE].visibility, 0.9)

    def test_position_is_unaffected_by_decay(self) -> None:
        frames = tuple(_frame(i, with_nose) for i, with_nose in enumerate([True, False, False]))

        result = MissedFramePersistence().apply(frames, max_missed_frames=2)

        for i in (1, 2):
            self.assertEqual(result[i].pose_landmarks[PoseLandmarkName.NOSE].position, _NOSE_LANDMARK.position)

    def test_custom_decay_model_is_honored(self) -> None:
        frames = tuple(_frame(i, with_nose) for i, with_nose in enumerate([True, False]))

        result = MissedFramePersistence(decay_model=LinearConfidenceDecay(per_frame_penalty=0.15)).apply(
            frames, max_missed_frames=1
        )

        self.assertAlmostEqual(result[1].pose_landmarks[PoseLandmarkName.NOSE].visibility, 0.9 - 0.15)

    def test_decay_eventually_crosses_the_downstream_visibility_threshold(self) -> None:
        # Documents the design intent from ExponentialConfidenceDecay's
        # docstring: with the default rate (0.75) and a typical real
        # detection confidence (~0.9), decay crosses the 0.5 threshold every
        # downstream joint-angle/CoM/weight-transfer calculator enforces
        # (MIN_LANDMARK_VISIBILITY in angle_calculator.py) within a few held
        # frames -- well before a typical max_missed_frames budget (5) ends.
        frames = tuple(
            _frame(i, with_nose) for i, with_nose in enumerate([True] + [False] * 5)
        )

        result = MissedFramePersistence().apply(frames, max_missed_frames=5)

        below_threshold_frame = next(
            i for i in range(1, 6) if result[i].pose_landmarks[PoseLandmarkName.NOSE].visibility < 0.5
        )
        self.assertLessEqual(below_threshold_frame, 5)


class MissedFramePersistenceReappearanceAfterGapTests(unittest.TestCase):
    """A landmark reappearing right after exhausting the hold budget (>=1
    frame of genuine absence, not just a run that stayed within budget) must
    be de-rated on that first reappearance frame, regardless of its own
    reported visibility -- see docs/bugs/discontinuity-detector-gap.md."""

    def test_reappearance_after_exceeding_budget_is_derated_by_the_same_curve(self) -> None:
        # max_missed_frames=1: frames 1,2 missing exceeds the 1-frame budget
        # (frame1 held, frame2 exhausts it -- genuine absence), frame3 is a
        # fresh reappearance.
        frames = tuple(
            _frame(i, with_nose) for i, with_nose in enumerate([True, False, False, True])
        )

        result = MissedFramePersistence().apply(frames, max_missed_frames=1)

        self.assertNotIn(PoseLandmarkName.NOSE, result[2].pose_landmarks)  # confirms a genuine gap occurred
        reappeared = result[3].pose_landmarks[PoseLandmarkName.NOSE]
        # Same decay curve as a held frame, continued one step past the
        # budget (max_missed_frames + 1 = 2), applied to the fresh (0.9)
        # reading -- not the pre-existing 0.9 value passed through.
        self.assertAlmostEqual(reappeared.visibility, 0.9 * 0.75**2)
        self.assertAlmostEqual(reappeared.presence, 0.9 * 0.75**2)

    def test_reappearance_after_exceeding_budget_crosses_threshold_even_with_high_raw_confidence(self) -> None:
        # Production default max_missed_frames=5, and a near-maximal raw
        # confidence (0.99) -- the exact case the bug report's real-video
        # reproduction hit (visibility 0.50-0.98 read as trustworthy).
        # "Regardless of the model's own reported visibility" means this
        # must land below the 0.5 threshold even here.
        high_confidence_nose = Landmark(position=Point3D(x=1.0, y=2.0, z=3.0), visibility=0.99, presence=0.99)
        timing0 = FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0)
        frame0 = LandmarkFrame(timing=timing0, pose_landmarks={PoseLandmarkName.NOSE: high_confidence_nose})
        gap_frames = [
            LandmarkFrame(
                timing=FrameTiming(frame_index=i, timestamp_ms=i * 100.0, delta_time_ms=100.0), pose_landmarks={}
            )
            for i in range(1, 8)  # 7 consecutive misses, exceeds a 5-frame budget
        ]
        reappear = LandmarkFrame(
            timing=FrameTiming(frame_index=8, timestamp_ms=800.0, delta_time_ms=100.0),
            pose_landmarks={PoseLandmarkName.NOSE: high_confidence_nose},
        )
        frames = (frame0, *gap_frames, reappear)

        result = MissedFramePersistence().apply(frames, max_missed_frames=5)

        self.assertNotIn(PoseLandmarkName.NOSE, result[6].pose_landmarks)  # confirms budget was exceeded
        self.assertLess(result[8].pose_landmarks[PoseLandmarkName.NOSE].visibility, 0.5)

    def test_reappearance_within_budget_is_unaffected(self) -> None:
        # Gap of 2 frames against a budget of 3 -- every intervening frame
        # was successfully held, never genuinely absent, so this is NOT the
        # gap-exceeding-budget case and must keep the pre-existing
        # full-confidence-reset behavior.
        frames = tuple(
            _frame(i, with_nose) for i, with_nose in enumerate([True, False, False, True])
        )

        result = MissedFramePersistence().apply(frames, max_missed_frames=3)

        self.assertIn(PoseLandmarkName.NOSE, result[1].pose_landmarks)  # held, not absent
        self.assertIn(PoseLandmarkName.NOSE, result[2].pose_landmarks)  # held, not absent
        self.assertEqual(result[3].pose_landmarks[PoseLandmarkName.NOSE].visibility, 0.9)

    def test_reappearance_position_is_the_fresh_reading_not_the_stale_one(self) -> None:
        stale = Landmark(position=Point3D(x=1.0, y=2.0, z=3.0), visibility=0.9, presence=0.9)
        fresh = Landmark(position=Point3D(x=50.0, y=60.0, z=70.0), visibility=0.9, presence=0.9)
        frames = (
            LandmarkFrame(
                timing=FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0),
                pose_landmarks={PoseLandmarkName.NOSE: stale},
            ),
            LandmarkFrame(
                timing=FrameTiming(frame_index=1, timestamp_ms=100.0, delta_time_ms=100.0), pose_landmarks={}
            ),
            LandmarkFrame(
                timing=FrameTiming(frame_index=2, timestamp_ms=200.0, delta_time_ms=100.0), pose_landmarks={}
            ),
            LandmarkFrame(
                timing=FrameTiming(frame_index=3, timestamp_ms=300.0, delta_time_ms=100.0),
                pose_landmarks={PoseLandmarkName.NOSE: fresh},
            ),
        )

        result = MissedFramePersistence().apply(frames, max_missed_frames=1)

        # Position is the fresh detection's own reading, not carried over
        # from the stale pre-gap position -- only confidence is de-rated.
        self.assertEqual(result[3].pose_landmarks[PoseLandmarkName.NOSE].position, fresh.position)

    def test_derated_reappearance_does_not_compound_on_a_later_hold(self) -> None:
        # ConfidenceDecayModel's contract: decay is always computed against
        # the pristine last-real-detection value, never a previously-decayed
        # one. A frame held immediately after a de-rated reappearance must
        # decay from the reappearance's own *raw* 0.9, not from its
        # already-lowered output value.
        frames = tuple(
            _frame(i, with_nose) for i, with_nose in enumerate([True, False, False, True, False])
        )

        result = MissedFramePersistence().apply(frames, max_missed_frames=1)

        held_after_reappearance = result[4].pose_landmarks[PoseLandmarkName.NOSE]
        self.assertAlmostEqual(held_after_reappearance.visibility, 0.9 * 0.75)


if __name__ == "__main__":
    unittest.main()
