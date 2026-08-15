"""Tests for summarize_reconstruction: pure aggregation over
ReconstructedLandmarkFrame sequences, no reconstruction logic here."""

from __future__ import annotations

import unittest

from engine.tracking.reconstruction.confidence_state import LandmarkState
from engine.tracking.reconstruction.diagnostics import summarize_reconstruction
from engine.tracking.reconstruction.landmark_reconstructor import ReconstructedLandmarkFrame
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, PoseLandmarkName
from engine.types.video import FrameTiming


def _timing(index: int) -> FrameTiming:
    return FrameTiming(frame_index=index, timestamp_ms=index * 33.333, delta_time_ms=33.333)


def _landmark(confidence: float) -> Landmark:
    return Landmark(position=Point3D(0.0, 0.0, 0.0), visibility=confidence, presence=confidence)


class SummarizeReconstructionTests(unittest.TestCase):
    def test_fractions_match_known_state_sequence(self) -> None:
        # RIGHT_WRIST: FRESH, FRESH, PREDICTED, HELD, MISSING (5 frames)
        name = PoseLandmarkName.RIGHT_WRIST
        state_sequence = [
            LandmarkState.FRESH,
            LandmarkState.FRESH,
            LandmarkState.PREDICTED,
            LandmarkState.HELD,
            LandmarkState.MISSING,
        ]
        frames = []
        for i, state in enumerate(state_sequence):
            landmarks = {} if state == LandmarkState.MISSING else {name: _landmark(0.8)}
            frames.append(ReconstructedLandmarkFrame(timing=_timing(i), pose_landmarks=landmarks, states={name: state}))

        summary = summarize_reconstruction(tuple(frames))[name.value]

        self.assertEqual(summary.total_frames, 5)
        self.assertAlmostEqual(summary.fresh_fraction, 2 / 5)
        self.assertAlmostEqual(summary.predicted_fraction, 1 / 5)
        self.assertAlmostEqual(summary.held_fraction, 1 / 5)
        self.assertAlmostEqual(summary.missing_fraction, 1 / 5)
        self.assertAlmostEqual(
            summary.fresh_fraction + summary.predicted_fraction + summary.held_fraction + summary.missing_fraction,
            1.0,
        )

    def test_mean_confidence_only_averages_non_missing_frames(self) -> None:
        name = PoseLandmarkName.LEFT_WRIST
        frames = [
            ReconstructedLandmarkFrame(
                timing=_timing(0), pose_landmarks={name: _landmark(1.0)}, states={name: LandmarkState.FRESH}
            ),
            ReconstructedLandmarkFrame(
                timing=_timing(1), pose_landmarks={name: _landmark(0.5)}, states={name: LandmarkState.PREDICTED}
            ),
            ReconstructedLandmarkFrame(timing=_timing(2), pose_landmarks={}, states={name: LandmarkState.MISSING}),
        ]

        summary = summarize_reconstruction(tuple(frames))[name.value]
        self.assertAlmostEqual(summary.mean_confidence, (1.0 + 0.5) / 2)

    def test_landmark_with_no_frames_present_reports_all_missing(self) -> None:
        frames = (ReconstructedLandmarkFrame(timing=_timing(0), pose_landmarks={}, states={}),)
        summary = summarize_reconstruction(frames)[PoseLandmarkName.NOSE.value]
        self.assertEqual(summary.missing_fraction, 1.0)
        self.assertEqual(summary.fresh_fraction, 0.0)
        self.assertEqual(summary.mean_confidence, 0.0)

    def test_empty_sequence_does_not_raise(self) -> None:
        summary = summarize_reconstruction(())[PoseLandmarkName.NOSE.value]
        self.assertEqual(summary.total_frames, 0)
        self.assertEqual(summary.fresh_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
