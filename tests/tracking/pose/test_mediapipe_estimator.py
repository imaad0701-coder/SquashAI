"""Tests for MediaPipePoseEstimator.

Uses a fake PoseDetector so none of this requires mediapipe/numpy to be
installed. CreateMediaPipePoseDetectorTests is the exception: mediapipe is
actually installed in this environment (pinned to 0.10.9, which still has
the legacy Solutions API — newer wheels dropped it, see requirements.txt),
so those tests exercise the real "not installed" error path via mocking
rather than relying on it being genuinely absent, plus one real integration
smoke test.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from typing import Sequence
from unittest.mock import patch

from engine.exceptions import PoseDetectionError
from engine.preprocessing.frame_iterator import FrameImage
from engine.tracking.base import TrackerConfig
from engine.tracking.pose.mediapipe_estimator import (
    MediaPipePoseEstimator,
    create_mediapipe_pose_detector,
)
from engine.tracking.pose.pose_estimator import PoseEstimatorConfig
from engine.types.landmarks import PoseLandmarkName
from engine.types.video import FrameMeta, FrameTiming


@dataclass(frozen=True)
class _FakeRawLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass(frozen=True)
class _FakeRawResult:
    landmarks: Sequence[_FakeRawLandmark] | None


class _FakeDetector:
    def __init__(self, results: list[_FakeRawResult]) -> None:
        self._results = list(results)
        self.processed_images: list[FrameImage] = []

    def process(self, image: FrameImage) -> _FakeRawResult:
        self.processed_images.append(image)
        return self._results.pop(0)


def _config() -> PoseEstimatorConfig:
    return PoseEstimatorConfig(
        tracker_config=TrackerConfig(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, max_missed_frames=3
        ),
        model_complexity=1,
    )


def _full_body_landmarks() -> list[_FakeRawLandmark]:
    # 33 landmarks; x encodes the index so mapping can be verified precisely.
    return [
        _FakeRawLandmark(x=index / 100.0, y=1.0, z=2.0, visibility=0.9, presence=0.8)
        for index in range(33)
    ]


def _image() -> FrameImage:
    return FrameImage(width=1, height=1, channels=3, data=b"\x00\x00\x00")


class MediaPipePoseEstimatorTests(unittest.TestCase):
    def test_estimate_sequence_maps_landmark_indices_correctly(self) -> None:
        detector = _FakeDetector([_FakeRawResult(landmarks=_full_body_landmarks())])
        estimator = MediaPipePoseEstimator(detector, _config())
        timing = FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0)

        results = list(estimator.estimate_sequence([(timing, _image())]))

        self.assertEqual(len(results), 1)
        landmark_frame = results[0]
        self.assertEqual(landmark_frame.timing, timing)
        self.assertEqual(landmark_frame.racket_landmarks, {})

        expected_indices = {
            PoseLandmarkName.NOSE: 0,
            PoseLandmarkName.LEFT_SHOULDER: 11,
            PoseLandmarkName.RIGHT_SHOULDER: 12,
            PoseLandmarkName.LEFT_ELBOW: 13,
            PoseLandmarkName.RIGHT_ELBOW: 14,
            PoseLandmarkName.LEFT_WRIST: 15,
            PoseLandmarkName.RIGHT_WRIST: 16,
            PoseLandmarkName.LEFT_HIP: 23,
            PoseLandmarkName.RIGHT_HIP: 24,
            PoseLandmarkName.LEFT_KNEE: 25,
            PoseLandmarkName.RIGHT_KNEE: 26,
            PoseLandmarkName.LEFT_ANKLE: 27,
            PoseLandmarkName.RIGHT_ANKLE: 28,
            PoseLandmarkName.LEFT_FOOT_INDEX: 31,
            PoseLandmarkName.RIGHT_FOOT_INDEX: 32,
        }
        self.assertEqual(set(landmark_frame.pose_landmarks), set(expected_indices))
        for name, index in expected_indices.items():
            landmark = landmark_frame.pose_landmarks[name]
            self.assertAlmostEqual(landmark.position.x, index / 100.0)
            self.assertEqual(landmark.visibility, 0.9)
            self.assertEqual(landmark.presence, 0.8)

    def test_estimate_sequence_yields_empty_dict_when_nothing_detected(self) -> None:
        detector = _FakeDetector([_FakeRawResult(landmarks=None)])
        estimator = MediaPipePoseEstimator(detector, _config())
        timing = FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0)

        results = list(estimator.estimate_sequence([(timing, _image())]))

        self.assertEqual(results[0].pose_landmarks, {})
        self.assertEqual(results[0].racket_landmarks, {})

    def test_estimate_sequence_processes_frames_in_order(self) -> None:
        detector = _FakeDetector(
            [_FakeRawResult(landmarks=None), _FakeRawResult(landmarks=None)]
        )
        estimator = MediaPipePoseEstimator(detector, _config())
        image_a, image_b = _image(), _image()
        frames = [
            (FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0), image_a),
            (FrameTiming(frame_index=1, timestamp_ms=100.0, delta_time_ms=100.0), image_b),
        ]

        list(estimator.estimate_sequence(frames))

        self.assertEqual(detector.processed_images, [image_a, image_b])

    def test_estimate_uses_frame_image_provider(self) -> None:
        detector = _FakeDetector([_FakeRawResult(landmarks=None)])
        frame_meta = FrameMeta(index=5, timestamp_seconds=0.5)
        provided_image = _image()

        def provider(frame: FrameMeta) -> FrameImage:
            self.assertEqual(frame, frame_meta)
            return provided_image

        estimator = MediaPipePoseEstimator(detector, _config(), frame_image_provider=provider)

        result = estimator.estimate(frame_meta, _config())

        self.assertEqual(
            result.timing,
            FrameTiming(
                frame_index=frame_meta.index,
                timestamp_ms=frame_meta.timestamp_seconds * 1000.0,
                delta_time_ms=0.0,
            ),
        )
        self.assertEqual(detector.processed_images, [provided_image])

    def test_estimate_without_provider_raises(self) -> None:
        detector = _FakeDetector([])
        estimator = MediaPipePoseEstimator(detector, _config())
        frame_meta = FrameMeta(index=0, timestamp_seconds=0.0)

        with self.assertRaises(PoseDetectionError):
            estimator.estimate(frame_meta, _config())


class CreateMediaPipePoseDetectorTests(unittest.TestCase):
    def test_raises_clear_error_when_mediapipe_not_installed(self) -> None:
        # Simulates absence rather than relying on it: mediapipe really is
        # installed in this environment (see module docstring).
        with patch.dict(sys.modules, {"mediapipe": None}):
            with self.assertRaises(PoseDetectionError):
                create_mediapipe_pose_detector(_config())

    def test_real_mediapipe_creates_a_working_detector(self) -> None:
        # Genuine integration smoke test: mediapipe==0.10.9 is installed
        # here and does still expose the legacy Solutions API.
        detector = create_mediapipe_pose_detector(_config())
        blank_image = FrameImage(width=32, height=24, channels=3, data=bytes(32 * 24 * 3))

        result = detector.process(blank_image)

        self.assertIsNone(result.landmarks)  # nothing to detect in a blank frame


if __name__ == "__main__":
    unittest.main()
