"""Tests for ThresholdVisibilityFilter."""

from __future__ import annotations

import unittest

from engine.tracking.pose.visibility_filter import ThresholdVisibilityFilter, VisibilityThresholds
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import FrameTiming


def _landmark(visibility: float, presence: float) -> Landmark:
    return Landmark(position=Point3D(x=0.0, y=0.0, z=0.0), visibility=visibility, presence=presence)


def _timing(index: int = 0) -> FrameTiming:
    return FrameTiming(frame_index=index, timestamp_ms=index * 100.0, delta_time_ms=0.0)


class ThresholdVisibilityFilterTests(unittest.TestCase):
    def test_keeps_landmarks_meeting_both_thresholds(self) -> None:
        frame = LandmarkFrame(
            timing=_timing(),
            pose_landmarks={PoseLandmarkName.NOSE: _landmark(0.9, 0.9)},
        )
        thresholds = VisibilityThresholds(min_visibility=0.5, min_presence=0.5)

        result = ThresholdVisibilityFilter().filter(frame, thresholds)

        self.assertIn(PoseLandmarkName.NOSE, result.pose_landmarks)

    def test_drops_landmark_below_visibility_threshold(self) -> None:
        frame = LandmarkFrame(
            timing=_timing(),
            pose_landmarks={PoseLandmarkName.NOSE: _landmark(0.1, 0.9)},
        )
        thresholds = VisibilityThresholds(min_visibility=0.5, min_presence=0.5)

        result = ThresholdVisibilityFilter().filter(frame, thresholds)

        self.assertNotIn(PoseLandmarkName.NOSE, result.pose_landmarks)

    def test_drops_landmark_below_presence_threshold(self) -> None:
        frame = LandmarkFrame(
            timing=_timing(),
            pose_landmarks={PoseLandmarkName.NOSE: _landmark(0.9, 0.1)},
        )
        thresholds = VisibilityThresholds(min_visibility=0.5, min_presence=0.5)

        result = ThresholdVisibilityFilter().filter(frame, thresholds)

        self.assertNotIn(PoseLandmarkName.NOSE, result.pose_landmarks)

    def test_preserves_timing_metadata_and_racket_landmarks(self) -> None:
        timing = _timing(index=7)
        frame = LandmarkFrame(timing=timing, pose_landmarks={}, racket_landmarks={})
        thresholds = VisibilityThresholds(min_visibility=0.5, min_presence=0.5)

        result = ThresholdVisibilityFilter().filter(frame, thresholds)

        self.assertEqual(result.timing, timing)
        self.assertEqual(result.racket_landmarks, {})


if __name__ == "__main__":
    unittest.main()
