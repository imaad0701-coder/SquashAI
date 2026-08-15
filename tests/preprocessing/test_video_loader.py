"""Tests for FFmpegVideoLoader."""

from __future__ import annotations

import unittest

from engine.preprocessing.ffmpeg_wrapper import ProbeResult
from engine.preprocessing.video_loader import FFmpegVideoLoader, VideoLoaderConfig
from engine.types.video import VideoFormat


class _FakeProbe:
    """Duck-typed stand-in for FFprobeWrapper (only .probe() is used)."""

    def __init__(self, result: ProbeResult) -> None:
        self._result = result

    def probe(self, video_path: str) -> ProbeResult:
        return self._result


class FFmpegVideoLoaderTests(unittest.TestCase):
    def _probe_result(self, **overrides: object) -> ProbeResult:
        defaults = dict(
            fmt=VideoFormat.MP4,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=300,
            duration_seconds=10.0,
            rotation_degrees=0,
        )
        defaults.update(overrides)
        return ProbeResult(**defaults)

    def test_load_metadata_uses_probed_values(self) -> None:
        loader = FFmpegVideoLoader(probe=_FakeProbe(self._probe_result()))
        config = VideoLoaderConfig(source_path="clip.mp4", target_fps=None, max_resolution=None)

        metadata = loader.load_metadata(config)

        self.assertEqual(metadata.path, "clip.mp4")
        self.assertEqual(metadata.fmt, VideoFormat.MP4)
        self.assertEqual(metadata.fps, 30.0)
        self.assertEqual(metadata.width, 1920)
        self.assertEqual(metadata.height, 1080)
        self.assertEqual(metadata.frame_count, 300)
        self.assertEqual(metadata.duration_seconds, 10.0)

    def test_load_metadata_honors_target_fps_override(self) -> None:
        loader = FFmpegVideoLoader(probe=_FakeProbe(self._probe_result(fps=60.0)))
        config = VideoLoaderConfig(source_path="clip.mp4", target_fps=24.0, max_resolution=None)

        metadata = loader.load_metadata(config)

        self.assertEqual(metadata.fps, 24.0)

    def test_load_rotation_returns_probed_rotation(self) -> None:
        loader = FFmpegVideoLoader(probe=_FakeProbe(self._probe_result(rotation_degrees=90)))
        config = VideoLoaderConfig(source_path="clip.mp4", target_fps=None, max_resolution=None)

        self.assertEqual(loader.load_rotation(config), 90)


if __name__ == "__main__":
    unittest.main()
