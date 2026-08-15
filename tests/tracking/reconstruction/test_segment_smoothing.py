"""Tests for smooth_predicted_segments: must only ever touch PREDICTED
positions, never FRESH/HELD/MISSING, and never reach across a run boundary
into a neighboring state's value."""

from __future__ import annotations

import unittest

from engine.tracking.reconstruction.confidence_state import LandmarkState
from engine.tracking.reconstruction.landmark_reconstructor import ReconstructedLandmarkFrame
from engine.tracking.reconstruction.segment_smoothing import smooth_predicted_segments
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, PoseLandmarkName
from engine.types.video import FrameTiming

_NAME = PoseLandmarkName.RIGHT_WRIST


def _timing(index: int) -> FrameTiming:
    return FrameTiming(frame_index=index, timestamp_ms=index * 33.333, delta_time_ms=33.333)


def _landmark(x: float, confidence: float = 0.5) -> Landmark:
    return Landmark(position=Point3D(x, 0.0, 0.0), visibility=confidence, presence=confidence)


def _seq(states_and_x: list[tuple[LandmarkState, float | None]]) -> tuple[ReconstructedLandmarkFrame, ...]:
    frames = []
    for i, (state, x) in enumerate(states_and_x):
        pose_landmarks = {} if x is None else {_NAME: _landmark(x)}
        frames.append(
            ReconstructedLandmarkFrame(timing=_timing(i), pose_landmarks=pose_landmarks, states={_NAME: state})
        )
    return tuple(frames)


class SmoothPredictedSegmentsTests(unittest.TestCase):
    def test_window_of_one_or_less_is_a_no_op(self) -> None:
        frames = _seq([(LandmarkState.FRESH, 0.0), (LandmarkState.PREDICTED, 10.0)])
        self.assertEqual(smooth_predicted_segments(frames, window=1), frames)
        self.assertEqual(smooth_predicted_segments(frames, window=0), frames)

    def test_empty_sequence_does_not_raise(self) -> None:
        self.assertEqual(smooth_predicted_segments((), window=3), ())

    def test_fresh_frames_are_never_modified(self) -> None:
        frames = _seq(
            [
                (LandmarkState.FRESH, 0.0),
                (LandmarkState.PREDICTED, 100.0),  # a wild outlier
                (LandmarkState.FRESH, 1.0),
            ]
        )
        smoothed = smooth_predicted_segments(frames, window=3)

        self.assertIs(smoothed[0].pose_landmarks[_NAME], frames[0].pose_landmarks[_NAME])
        self.assertIs(smoothed[2].pose_landmarks[_NAME], frames[2].pose_landmarks[_NAME])

    def test_held_and_missing_frames_are_never_modified(self) -> None:
        frames = _seq(
            [
                (LandmarkState.FRESH, 0.0),
                (LandmarkState.HELD, 0.0),
                (LandmarkState.MISSING, None),
            ]
        )
        smoothed = smooth_predicted_segments(frames, window=3)

        self.assertIs(smoothed[1].pose_landmarks[_NAME], frames[1].pose_landmarks[_NAME])
        self.assertNotIn(_NAME, smoothed[2].pose_landmarks)

    def test_predicted_run_is_averaged_within_itself(self) -> None:
        # A run of three PREDICTED frames: 0, 10, 20 -- centered average
        # with window=3 should smooth the middle one using all three, and
        # the edges using whatever's available within the run.
        frames = _seq(
            [
                (LandmarkState.FRESH, -10.0),  # must not contribute to the average
                (LandmarkState.PREDICTED, 0.0),
                (LandmarkState.PREDICTED, 10.0),
                (LandmarkState.PREDICTED, 20.0),
                (LandmarkState.FRESH, 100.0),  # must not contribute to the average
            ]
        )
        smoothed = smooth_predicted_segments(frames, window=3)

        # middle of the run: average of (0, 10, 20) = 10 (unchanged here,
        # but computed from the run's own values, not the neighboring FRESH ones)
        self.assertAlmostEqual(smoothed[2].pose_landmarks[_NAME].position.x, 10.0, places=6)
        # left edge of run: only has itself + the next in-run neighbor -> avg(0, 10) = 5
        self.assertAlmostEqual(smoothed[1].pose_landmarks[_NAME].position.x, 5.0, places=6)
        # right edge of run: avg(10, 20) = 15
        self.assertAlmostEqual(smoothed[3].pose_landmarks[_NAME].position.x, 15.0, places=6)

    def test_confidence_and_state_are_preserved_exactly(self) -> None:
        frames = _seq([(LandmarkState.FRESH, 0.0), (LandmarkState.PREDICTED, 10.0), (LandmarkState.PREDICTED, 20.0)])
        original_confidence = frames[1].pose_landmarks[_NAME].visibility

        smoothed = smooth_predicted_segments(frames, window=3)

        self.assertEqual(smoothed[1].states[_NAME], LandmarkState.PREDICTED)
        self.assertAlmostEqual(smoothed[1].pose_landmarks[_NAME].visibility, original_confidence, places=6)

    def test_two_separate_predicted_runs_are_smoothed_independently(self) -> None:
        frames = _seq(
            [
                (LandmarkState.PREDICTED, 0.0),
                (LandmarkState.PREDICTED, 100.0),  # end of run 1
                (LandmarkState.FRESH, 50.0),  # boundary -- must not leak into either run
                (LandmarkState.PREDICTED, 1000.0),  # start of run 2
                (LandmarkState.PREDICTED, 1010.0),
            ]
        )
        smoothed = smooth_predicted_segments(frames, window=3)

        # Run 1's smoothed values must stay in run-1's own numeric range, nowhere near run 2's.
        self.assertLess(smoothed[0].pose_landmarks[_NAME].position.x, 200.0)
        self.assertLess(smoothed[1].pose_landmarks[_NAME].position.x, 200.0)
        # Run 2's smoothed values must stay in run-2's own numeric range.
        self.assertGreater(smoothed[3].pose_landmarks[_NAME].position.x, 900.0)
        self.assertGreater(smoothed[4].pose_landmarks[_NAME].position.x, 900.0)


if __name__ == "__main__":
    unittest.main()
