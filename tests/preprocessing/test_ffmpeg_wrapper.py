"""Tests for the ffmpeg/ffprobe subprocess wrapper.

All tests inject a fake runner/process factory so they never require the
real ffmpeg/ffprobe binaries to be installed.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from engine.exceptions import VideoLoadError
from engine.preprocessing.ffmpeg_wrapper import FFmpegFrameReader, FFprobeWrapper
from engine.types.video import VideoFormat


def _completed(stdout: bytes, returncode: int = 0, stderr: bytes = b"") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _payload(
    width: str | int = "64",
    height: str | int = "48",
    r_frame_rate: str = "30/1",
    nb_frames: str | None = "60",
    duration: str = "2.000000",
    tags: dict | None = None,
    side_data_list: list | None = None,
) -> dict:
    stream: dict = {
        "codec_type": "video",
        "width": width,
        "height": height,
        "r_frame_rate": r_frame_rate,
    }
    if nb_frames is not None:
        stream["nb_frames"] = nb_frames
    if tags is not None:
        stream["tags"] = tags
    if side_data_list is not None:
        stream["side_data_list"] = side_data_list
    return {
        "format": {"duration": duration, "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [{"codec_type": "audio"}, stream],
    }


class FFprobeWrapperTests(unittest.TestCase):
    def test_probe_returns_expected_metadata(self) -> None:
        runner_calls = []

        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            runner_calls.append(args)
            return _completed(json.dumps(_payload()).encode("utf-8"))

        wrapper = FFprobeWrapper(runner=fake_runner)
        result = wrapper.probe("clip.mp4")

        self.assertEqual(result.fmt, VideoFormat.MP4)
        self.assertEqual(result.fps, 30.0)
        self.assertEqual(result.width, 64)
        self.assertEqual(result.height, 48)
        self.assertEqual(result.frame_count, 60)
        self.assertEqual(result.duration_seconds, 2.0)
        self.assertEqual(result.rotation_degrees, 0)
        self.assertEqual(runner_calls[0][-1], "clip.mp4")

    def test_probe_computes_frame_count_when_nb_frames_missing(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(json.dumps(_payload(nb_frames=None, duration="2.0", r_frame_rate="30/1")).encode())

        result = FFprobeWrapper(runner=fake_runner).probe("clip.mov")
        self.assertEqual(result.frame_count, 60)

    def test_probe_parses_rotation_from_tags(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(json.dumps(_payload(tags={"rotate": "90"})).encode())

        result = FFprobeWrapper(runner=fake_runner).probe("clip.mp4")
        self.assertEqual(result.rotation_degrees, 90)

    def test_probe_parses_rotation_from_side_data_list(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            payload = _payload(side_data_list=[{"side_data_type": "Display Matrix", "rotation": -90.0}])
            return _completed(json.dumps(payload).encode())

        result = FFprobeWrapper(runner=fake_runner).probe("clip.mp4")
        self.assertEqual(result.rotation_degrees, 270)

    def test_probe_raises_on_nonzero_returncode(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(b"", returncode=1, stderr=b"no such file")

        with self.assertRaises(VideoLoadError):
            FFprobeWrapper(runner=fake_runner).probe("missing.mp4")

    def test_probe_raises_on_invalid_json(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(b"not json")

        with self.assertRaises(VideoLoadError):
            FFprobeWrapper(runner=fake_runner).probe("clip.mp4")

    def test_probe_raises_when_no_video_stream(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            payload = {"format": {"duration": "1.0"}, "streams": [{"codec_type": "audio"}]}
            return _completed(json.dumps(payload).encode())

        with self.assertRaises(VideoLoadError):
            FFprobeWrapper(runner=fake_runner).probe("clip.mp4")

    def test_probe_raises_on_unsupported_extension(self) -> None:
        def fake_runner(args: list[str]) -> subprocess.CompletedProcess:
            return _completed(json.dumps(_payload()).encode())

        with self.assertRaises(VideoLoadError):
            FFprobeWrapper(runner=fake_runner).probe("clip.xyz")


class FFmpegFrameReaderTests(unittest.TestCase):
    def test_open_stream_builds_expected_args_and_delegates_to_factory(self) -> None:
        captured_args: list[list[str]] = []

        class _FakeProcess:
            stdout = None
            stderr = None

            def wait(self) -> int:
                return 0

        def factory(args: list[str]) -> _FakeProcess:
            captured_args.append(args)
            return _FakeProcess()

        reader = FFmpegFrameReader(ffmpeg_path="ffmpeg", process_factory=factory)
        reader.open_stream("clip.mp4", width=64, height=48, pix_fmt="rgb24")

        self.assertEqual(
            captured_args[0],
            [
                "ffmpeg",
                "-v", "error",
                "-i", "clip.mp4",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", "64x48",
                "-",
            ],
        )

    def test_default_process_factory_discards_stderr_not_pipes_it(self) -> None:
        # Regression test: piping stderr without ever draining it is a
        # classic subprocess deadlock (VideoFrameIterator reads stdout
        # synchronously and never touches stderr until process.wait()).
        # DEVNULL removes the pipe -- and the possibility of it filling up
        # and blocking ffmpeg -- entirely. Covers the real (non-injected)
        # process factory, which the other tests in this file bypass.
        with patch("engine.preprocessing.ffmpeg_wrapper.shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("engine.preprocessing.ffmpeg_wrapper.subprocess.Popen") as mock_popen:
                reader = FFmpegFrameReader()  # no process_factory override: exercises the real default
                reader.open_stream("clip.mp4", width=64, height=48)

        _args, kwargs = mock_popen.call_args
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)


if __name__ == "__main__":
    unittest.main()
