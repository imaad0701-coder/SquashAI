"""Tests for FFprobeFrameTimingSource and FrameSynchronizer."""

from __future__ import annotations

import subprocess
import unittest

from engine.exceptions import FrameTimingError
from engine.preprocessing.frame_sync import FFprobeFrameTimingSource, FrameSynchronizer
from engine.types.video import FrameTiming


def _completed(stdout: bytes, returncode: int = 0, stderr: bytes = b"") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class FFprobeFrameTimingSourceTests(unittest.TestCase):
    def test_parses_one_timestamp_per_line(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(b"0.000000\n0.033367\n0.066733\n")

        source = FFprobeFrameTimingSource(runner=fake_runner)
        timestamps = source.probe_frame_timestamps("clip.mp4")

        self.assertEqual(timestamps, {0: 0.0, 1: 0.033367, 2: 0.066733})

    def test_treats_na_as_unresolved(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(b"0.000000\nN/A\n0.1\n")

        source = FFprobeFrameTimingSource(runner=fake_runner)
        timestamps = source.probe_frame_timestamps("clip.mp4")

        self.assertEqual(timestamps, {0: 0.0, 1: None, 2: 0.1})

    def test_trailing_comma_from_frame_side_data_is_handled(self) -> None:
        # Regression test: ffprobe's csv writer appends an extra (often
        # empty) trailing column whenever a frame carries ANY side_data_list
        # entry, even one never requested via -show_entries. Confirmed
        # against a real ffmpeg 8.1.2 encode where frame 0 alone carried an
        # empty side-data blob, producing "0.000000," instead of "0.000000"
        # -- float("0.000000,") raised ValueError and crashed every single
        # real pipeline run during multi-video QA testing.
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(b"0.000000,\n0.033333\n0.066667\n")

        source = FFprobeFrameTimingSource(runner=fake_runner)
        timestamps = source.probe_frame_timestamps("clip.mp4")

        self.assertEqual(timestamps, {0: 0.0, 1: 0.033333, 2: 0.066667})

    def test_na_with_trailing_comma_is_still_treated_as_unresolved(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(b"0.0\nN/A,\n0.1\n")

        source = FFprobeFrameTimingSource(runner=fake_runner)
        timestamps = source.probe_frame_timestamps("clip.mp4")

        self.assertEqual(timestamps, {0: 0.0, 1: None, 2: 0.1})

    def test_raises_on_nonzero_returncode(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(b"", returncode=1, stderr=b"boom")

        with self.assertRaises(FrameTimingError):
            FFprobeFrameTimingSource(runner=fake_runner).probe_frame_timestamps("clip.mp4")

    def test_raises_when_no_frames_reported(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(b"")

        with self.assertRaises(FrameTimingError):
            FFprobeFrameTimingSource(runner=fake_runner).probe_frame_timestamps("clip.mp4")


class FrameSynchronizerTests(unittest.TestCase):
    def test_exposes_frame_index_timestamp_ms_delta_time_ms(self) -> None:
        synchronizer = FrameSynchronizer({0: 0.0, 1: 0.1})

        timeline = synchronizer.build_timeline([0, 1])

        self.assertEqual(
            timeline,
            (
                FrameTiming(frame_index=0, timestamp_ms=0.0, delta_time_ms=0.0),
                FrameTiming(frame_index=1, timestamp_ms=100.0, delta_time_ms=100.0),
            ),
        )

    def test_variable_fps_produces_real_unequal_deltas(self) -> None:
        # Real-world variable-frame-rate spacing: 20ms, then 50ms, then 10ms.
        synchronizer = FrameSynchronizer({0: 0.0, 1: 0.02, 2: 0.07, 3: 0.08})

        timeline = synchronizer.build_timeline([0, 1, 2, 3])

        deltas = [t.delta_time_ms for t in timeline]
        self.assertAlmostEqual(deltas[0], 0.0)
        self.assertAlmostEqual(deltas[1], 20.0)
        self.assertAlmostEqual(deltas[2], 50.0)
        self.assertAlmostEqual(deltas[3], 10.0)

    def test_dropped_frame_is_linearly_interpolated_between_neighbors(self) -> None:
        # frame 1 was dropped/corrupted: ffprobe reported it as None.
        synchronizer = FrameSynchronizer({0: 0.0, 1: None, 2: 0.2})

        timeline = synchronizer.build_timeline([0, 1, 2])

        self.assertAlmostEqual(timeline[1].timestamp_ms, 100.0)
        self.assertAlmostEqual(timeline[1].delta_time_ms, 100.0)
        self.assertAlmostEqual(timeline[2].delta_time_ms, 100.0)

    def test_dropped_frame_missing_entirely_is_still_interpolated(self) -> None:
        # frame 1 isn't in the map at all (not even as None) — same handling.
        synchronizer = FrameSynchronizer({0: 0.0, 2: 0.2})

        timeline = synchronizer.build_timeline([0, 1, 2])

        self.assertAlmostEqual(timeline[1].timestamp_ms, 100.0)

    def test_extrapolates_forward_using_local_rate(self) -> None:
        # Only frames 0 and 1 are known; frame 2 must be extrapolated using
        # the real local rate between them (0.1s/frame), not a global fps.
        synchronizer = FrameSynchronizer({0: 0.0, 1: 0.1})

        timeline = synchronizer.build_timeline([0, 1, 2])

        self.assertAlmostEqual(timeline[2].timestamp_ms, 200.0)
        self.assertAlmostEqual(timeline[2].delta_time_ms, 100.0)

    def test_extrapolates_flat_when_only_one_known_point_on_that_side(self) -> None:
        # Only frame 5 is known at all; frame 3 (before it) has no second
        # point to derive a rate from, so it holds frame 5's timestamp flat.
        synchronizer = FrameSynchronizer({5: 1.0})

        timeline = synchronizer.build_timeline([3, 5])

        self.assertAlmostEqual(timeline[0].timestamp_ms, 1000.0)
        self.assertAlmostEqual(timeline[1].timestamp_ms, 1000.0)

    def test_raises_when_no_known_timestamps_at_all(self) -> None:
        synchronizer = FrameSynchronizer({0: None, 1: None})

        with self.assertRaises(FrameTimingError):
            synchronizer.build_timeline([0, 1])

    def test_empty_frame_indices_returns_empty_tuple(self) -> None:
        synchronizer = FrameSynchronizer({0: 0.0})

        self.assertEqual(synchronizer.build_timeline([]), ())


if __name__ == "__main__":
    unittest.main()
