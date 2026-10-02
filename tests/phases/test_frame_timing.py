"""Tests for engine.phases.frame_timing."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from engine.phases.frame_timing import ms_per_frame, ms_to_frames


@dataclass(frozen=True)
class _Sample:
    frame_index: int
    timestamp_ms: float


class MsPerFrameTests(unittest.TestCase):
    def test_exact_30fps(self) -> None:
        samples = [_Sample(i, i * (1000.0 / 30.0)) for i in range(10)]
        self.assertAlmostEqual(ms_per_frame(samples), 1000.0 / 30.0, places=9)

    def test_exact_59895fps_measured_like_sample_forehand1(self) -> None:
        rate = 1000.0 / 59.895
        samples = [_Sample(i, i * rate) for i in range(20)]
        self.assertAlmostEqual(ms_per_frame(samples), rate, places=6)

    def test_uses_only_first_and_last_ignoring_gaps_in_between(self) -> None:
        # frame_index-based, not list-position-based -- a missing middle
        # sample (e.g. an invalid frame skipped upstream) must not skew it.
        samples = [_Sample(0, 0.0), _Sample(50, 50 * (1000.0 / 30.0))]
        self.assertAlmostEqual(ms_per_frame(samples), 1000.0 / 30.0, places=9)

    def test_fewer_than_two_samples_returns_none(self) -> None:
        self.assertIsNone(ms_per_frame([]))
        self.assertIsNone(ms_per_frame([_Sample(0, 0.0)]))

    def test_zero_frame_span_returns_none(self) -> None:
        # Same frame_index at both ends -- can't derive a rate, not a divide-by-zero.
        self.assertIsNone(ms_per_frame([_Sample(5, 0.0), _Sample(5, 100.0)]))

    def test_non_positive_timestamp_span_returns_none(self) -> None:
        self.assertIsNone(ms_per_frame([_Sample(0, 100.0), _Sample(10, 100.0)]))
        self.assertIsNone(ms_per_frame([_Sample(0, 100.0), _Sample(10, 50.0)]))


class MsToFramesTests(unittest.TestCase):
    def test_round_trips_exactly_at_30fps_for_the_tuned_constants(self) -> None:
        rate = 1000.0 / 30.0
        self.assertEqual(ms_to_frames(500.0, rate), 15)   # SWING_MIN_REST_GAP_MS
        self.assertEqual(ms_to_frames(1000.0, rate), 30)  # MAX_MATCH_DISTANCE_MS

    def test_rounds_to_nearest_frame(self) -> None:
        self.assertEqual(ms_to_frames(100.0, 30.0), 3)   # 3.33 -> 3
        self.assertEqual(ms_to_frames(100.0, 20.0), 5)   # exact

    def test_none_rate_falls_back_to_30fps(self) -> None:
        self.assertEqual(ms_to_frames(500.0, None), ms_to_frames(500.0, 1000.0 / 30.0))


if __name__ == "__main__":
    unittest.main()
