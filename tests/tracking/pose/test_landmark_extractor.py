"""Tests for PoseLandmarkExtractor."""

from __future__ import annotations

import unittest

from engine.exceptions import PoseDetectionError
from engine.tracking.pose.landmark_extractor import PoseLandmarkExtractor
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import FrameTiming

_TIMING = FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0)


class PoseLandmarkExtractorTests(unittest.TestCase):
    def test_extract_returns_present_landmark(self) -> None:
        landmark = Landmark(position=Point3D(x=1.0, y=2.0, z=3.0), visibility=0.9, presence=0.8)
        frame = LandmarkFrame(
            timing=_TIMING,
            pose_landmarks={PoseLandmarkName.NOSE: landmark},
        )

        extractor = PoseLandmarkExtractor()
        self.assertEqual(extractor.extract(frame, PoseLandmarkName.NOSE), landmark)

    def test_extract_raises_for_missing_landmark(self) -> None:
        frame = LandmarkFrame(timing=_TIMING, pose_landmarks={})

        extractor = PoseLandmarkExtractor()
        with self.assertRaises(PoseDetectionError):
            extractor.extract(frame, PoseLandmarkName.NOSE)


if __name__ == "__main__":
    unittest.main()
