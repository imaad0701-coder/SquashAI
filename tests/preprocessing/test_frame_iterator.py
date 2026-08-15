"""Tests for VideoFrameIterator and its resize/rotate helpers."""

from __future__ import annotations

import unittest

from engine.exceptions import CorruptedFrameError
from engine.preprocessing.frame_iterator import (
    FrameImage,
    FrameIteratorConfig,
    VideoFrameIterator,
    resize_nearest,
    rotate_frame,
)
from engine.types.video import FrameMeta, VideoFormat, VideoMetadata


def _metadata(width: int, height: int, fps: float = 10.0) -> VideoMetadata:
    return VideoMetadata(
        path="clip.mp4",
        fmt=VideoFormat.MP4,
        fps=fps,
        width=width,
        height=height,
        frame_count=1000,
        duration_seconds=100.0,
    )


class _ScriptedStream:
    """Returns each scripted chunk verbatim on successive .read() calls,
    regardless of the requested size, then empty bytes forever."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def read(self, n: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, stream: _ScriptedStream) -> None:
        self.stdout = stream
        self.stderr = None
        self.waited = False

    def wait(self) -> int:
        self.waited = True
        return 0


class _FakeFrameReader:
    def __init__(self, stream: _ScriptedStream) -> None:
        self._stream = stream
        self.opened_with: tuple | None = None

    def open_stream(self, video_path: str, width: int, height: int, pix_fmt: str = "rgb24") -> _FakeProcess:
        self.opened_with = (video_path, width, height, pix_fmt)
        return _FakeProcess(self._stream)


class ResizeNearestTests(unittest.TestCase):
    def test_upsample_2x2_to_4x4_repeats_nearest_pixel(self) -> None:
        # 2x2, single channel, values laid out row-major: [0, 1, 2, 3]
        image = FrameImage(width=2, height=2, channels=1, data=bytes([0, 1, 2, 3]))

        resized = resize_nearest(image, 4, 4)

        self.assertEqual(resized.width, 4)
        self.assertEqual(resized.height, 4)
        expected_rows = [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ]
        expected = bytes(v for row in expected_rows for v in row)
        self.assertEqual(resized.data, expected)

    def test_downsample_4x4_to_2x2_picks_nearest_pixel(self) -> None:
        image = FrameImage(width=4, height=4, channels=1, data=bytes(range(16)))

        resized = resize_nearest(image, 2, 2)

        self.assertEqual(resized.width, 2)
        self.assertEqual(resized.height, 2)
        # source pixel (0,0)=0, (2,0)=2, (0,2)=8, (2,2)=10
        self.assertEqual(resized.data, bytes([0, 2, 8, 10]))

    def test_same_resolution_returns_same_image(self) -> None:
        image = FrameImage(width=2, height=2, channels=1, data=bytes([1, 2, 3, 4]))
        self.assertIs(resize_nearest(image, 2, 2), image)

    def test_rejects_non_positive_target(self) -> None:
        image = FrameImage(width=2, height=2, channels=1, data=bytes([1, 2, 3, 4]))
        with self.assertRaises(ValueError):
            resize_nearest(image, 0, 2)


class RotateFrameTests(unittest.TestCase):
    def _grid(self) -> FrameImage:
        # 2 wide x 3 tall, single channel: rows [0,1] [2,3] [4,5]
        return FrameImage(width=2, height=3, channels=1, data=bytes([0, 1, 2, 3, 4, 5]))

    def test_rotate_0_is_identity(self) -> None:
        image = self._grid()
        self.assertIs(rotate_frame(image, 0), image)

    def test_rotate_90_clockwise(self) -> None:
        rotated = rotate_frame(self._grid(), 90)
        # 2x3 -> 3x2, clockwise: top row of result reads bottom-to-top of first column
        self.assertEqual(rotated.width, 3)
        self.assertEqual(rotated.height, 2)
        self.assertEqual(rotated.data, bytes([4, 2, 0, 5, 3, 1]))

    def test_rotate_180(self) -> None:
        rotated = rotate_frame(self._grid(), 180)
        self.assertEqual(rotated.width, 2)
        self.assertEqual(rotated.height, 3)
        self.assertEqual(rotated.data, bytes([5, 4, 3, 2, 1, 0]))

    def test_rotate_270(self) -> None:
        rotated = rotate_frame(self._grid(), 270)
        self.assertEqual(rotated.width, 3)
        self.assertEqual(rotated.height, 2)
        self.assertEqual(rotated.data, bytes([1, 3, 5, 0, 2, 4]))

    def test_rejects_unsupported_angle(self) -> None:
        with self.assertRaises(ValueError):
            rotate_frame(self._grid(), 45)


class VideoFrameIteratorTests(unittest.TestCase):
    def test_yields_only_requested_frames_in_order(self) -> None:
        frame0 = bytes(range(0, 12))
        frame1 = bytes(range(100, 112))
        frame2 = bytes(range(200, 212))
        stream = _ScriptedStream([frame0, frame1, frame2])
        reader = _FakeFrameReader(stream)
        iterator = VideoFrameIterator(reader)
        metadata = _metadata(width=2, height=2)
        requested = (FrameMeta(index=0, timestamp_seconds=0.0), FrameMeta(index=2, timestamp_seconds=0.2))

        results = list(iterator.iter_frames("clip.mp4", metadata, requested))

        self.assertEqual([meta.index for meta, _ in results], [0, 2])
        self.assertEqual(results[0][1].data, frame0)
        self.assertEqual(results[1][1].data, frame2)
        self.assertEqual(reader.opened_with, ("clip.mp4", 2, 2, "rgb24"))
        self.assertTrue(stream.closed)

    def test_applies_rotation_and_resolution_normalization(self) -> None:
        frame0 = bytes(range(0, 12))  # 2x2 rgb
        stream = _ScriptedStream([frame0])
        reader = _FakeFrameReader(stream)
        iterator = VideoFrameIterator(reader)
        metadata = _metadata(width=2, height=2)
        requested = (FrameMeta(index=0, timestamp_seconds=0.0),)

        results = list(
            iterator.iter_frames(
                "clip.mp4", metadata, requested, target_resolution=(4, 4), rotation_degrees=90
            )
        )

        self.assertEqual(len(results), 1)
        _, image = results[0]

        raw = FrameImage(width=2, height=2, channels=3, data=frame0)
        expected = resize_nearest(rotate_frame(raw, 90), 4, 4)
        self.assertEqual(image, expected)

    def test_stops_cleanly_at_end_of_stream(self) -> None:
        frame0 = bytes(range(0, 12))
        stream = _ScriptedStream([frame0])  # only one frame available
        reader = _FakeFrameReader(stream)
        iterator = VideoFrameIterator(reader)
        metadata = _metadata(width=2, height=2)
        requested = (
            FrameMeta(index=0, timestamp_seconds=0.0),
            FrameMeta(index=1, timestamp_seconds=0.1),
        )

        results = list(iterator.iter_frames("clip.mp4", metadata, requested))

        self.assertEqual([meta.index for meta, _ in results], [0])

    def test_skips_corrupted_frame_when_configured(self) -> None:
        frame0 = bytes(range(0, 12))
        truncated = bytes(range(50, 56))  # only 6 of 12 bytes
        frame2 = bytes(range(200, 212))
        stream = _ScriptedStream([frame0, truncated, b"", frame2])
        reader = _FakeFrameReader(stream)
        config = FrameIteratorConfig(skip_corrupted=True, max_consecutive_corrupted=5)
        iterator = VideoFrameIterator(reader, config)
        metadata = _metadata(width=2, height=2)
        requested = tuple(FrameMeta(index=i, timestamp_seconds=i / 10.0) for i in range(3))

        results = list(iterator.iter_frames("clip.mp4", metadata, requested))

        self.assertEqual([meta.index for meta, _ in results], [0, 2])

    def test_raises_on_corrupted_frame_when_not_configured_to_skip(self) -> None:
        frame0 = bytes(range(0, 12))
        truncated = bytes(range(50, 56))
        stream = _ScriptedStream([frame0, truncated, b""])
        reader = _FakeFrameReader(stream)
        config = FrameIteratorConfig(skip_corrupted=False)
        iterator = VideoFrameIterator(reader, config)
        metadata = _metadata(width=2, height=2)
        requested = tuple(FrameMeta(index=i, timestamp_seconds=i / 10.0) for i in range(2))

        with self.assertRaises(CorruptedFrameError):
            list(iterator.iter_frames("clip.mp4", metadata, requested))

    def test_raises_when_consecutive_corrupted_exceeds_threshold(self) -> None:
        frame0 = bytes(range(0, 12))
        truncated = bytes(range(50, 56))
        frame3 = bytes(range(200, 212))
        # frame0 good, then two corrupted frames in a row, then a good one
        stream = _ScriptedStream([frame0, truncated, b"", truncated, b"", frame3])
        reader = _FakeFrameReader(stream)
        config = FrameIteratorConfig(skip_corrupted=True, max_consecutive_corrupted=1)
        iterator = VideoFrameIterator(reader, config)
        metadata = _metadata(width=2, height=2)
        requested = tuple(FrameMeta(index=i, timestamp_seconds=i / 10.0) for i in range(4))

        with self.assertRaises(CorruptedFrameError):
            list(iterator.iter_frames("clip.mp4", metadata, requested))

    def test_empty_frame_request_yields_nothing(self) -> None:
        stream = _ScriptedStream([])
        reader = _FakeFrameReader(stream)
        iterator = VideoFrameIterator(reader)
        metadata = _metadata(width=2, height=2)

        results = list(iterator.iter_frames("clip.mp4", metadata, ()))

        self.assertEqual(results, [])
        # nothing was requested, so the stream should never even be opened
        self.assertIsNone(reader.opened_with)


if __name__ == "__main__":
    unittest.main()
