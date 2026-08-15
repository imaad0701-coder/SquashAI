"""Tests for MediaPipePoseDetectorAdapter's pixel denormalization.

Unlike the rest of this test package, this genuinely needs numpy — that's a
real dependency of the adapter itself, not something to fake around (it's
"the actual mediapipe/numpy integration boundary", per its own docstring).
Skipped if numpy isn't installed, so the rest of the suite stays runnable
without it.
"""

from __future__ import annotations

import unittest

try:
    import numpy  # noqa: F401

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from engine.preprocessing.frame_iterator import FrameImage
from engine.tracking.pose.mediapipe_estimator import MediaPipePoseDetectorAdapter


class _FakeMediaPipeLandmark:
    def __init__(self, x: float, y: float, z: float, visibility: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class _FakeLandmarkList:
    def __init__(self, landmarks: list[_FakeMediaPipeLandmark]) -> None:
        self.landmark = landmarks


class _FakePoseLandmarksResult:
    def __init__(self, pose_landmarks: _FakeLandmarkList | None) -> None:
        self.pose_landmarks = pose_landmarks


class _FakeMediaPipePose:
    def __init__(self, result: _FakePoseLandmarksResult) -> None:
        self._result = result

    def process(self, array: object) -> _FakePoseLandmarksResult:
        return self._result


@unittest.skipUnless(_HAS_NUMPY, "requires numpy")
class MediaPipePoseDetectorAdapterTests(unittest.TestCase):
    def test_denormalizes_landmark_coordinates_to_pixel_space(self) -> None:
        width, height = 640, 480
        raw_landmark = _FakeMediaPipeLandmark(x=0.5, y=0.25, z=0.1, visibility=0.9)
        adapter = MediaPipePoseDetectorAdapter(
            _FakeMediaPipePose(_FakePoseLandmarksResult(_FakeLandmarkList([raw_landmark])))
        )

        image = FrameImage(width=width, height=height, channels=3, data=bytes(width * height * 3))
        result = adapter.process(image)

        self.assertEqual(len(result.landmarks), 1)
        landmark = result.landmarks[0]
        self.assertAlmostEqual(landmark.x, 0.5 * width)
        self.assertAlmostEqual(landmark.y, 0.25 * height)
        self.assertAlmostEqual(landmark.z, 0.1 * width)
        self.assertEqual(landmark.visibility, 0.9)

    def test_presence_uses_visibility_since_legacy_api_never_sets_it(self) -> None:
        # mediapipe's legacy Solutions API (unlike the Tasks API) never
        # populates `presence` — it always reads back as 0.0 regardless of
        # detection quality. Using visibility as its stand-in is what keeps
        # ThresholdVisibilityFilter's min_presence check from wiping out
        # every landmark on every real detection.
        raw_landmark = _FakeMediaPipeLandmark(x=0.1, y=0.2, z=0.0, visibility=0.73)
        adapter = MediaPipePoseDetectorAdapter(
            _FakeMediaPipePose(_FakePoseLandmarksResult(_FakeLandmarkList([raw_landmark])))
        )

        image = FrameImage(width=100, height=100, channels=3, data=bytes(100 * 100 * 3))
        result = adapter.process(image)

        self.assertEqual(result.landmarks[0].presence, 0.73)

    def test_no_detection_returns_none_landmarks(self) -> None:
        adapter = MediaPipePoseDetectorAdapter(_FakeMediaPipePose(_FakePoseLandmarksResult(pose_landmarks=None)))

        image = FrameImage(width=10, height=10, channels=3, data=bytes(10 * 10 * 3))
        result = adapter.process(image)

        self.assertIsNone(result.landmarks)


if __name__ == "__main__":
    unittest.main()
