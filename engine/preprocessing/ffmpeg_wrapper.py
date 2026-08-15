"""Concrete subprocess-based wrapper around the ffmpeg/ffprobe binaries.

No third-party dependency (e.g. ffmpeg-python, opencv) is used: metadata is
obtained by parsing ``ffprobe -show_format -show_streams`` JSON output, and
raw pixel data is obtained by reading ``ffmpeg ... -f rawvideo`` from a
subprocess's stdout. Both the probe runner and the frame-stream process are
injectable so callers (and tests) never need the real binaries installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from engine.exceptions import VideoLoadError
from engine.types.video import VideoFormat

ProbeRunner = Callable[[list[str]], "subprocess.CompletedProcess[bytes]"]


class FrameStream(Protocol):
    """Minimal duck-typed surface of subprocess.Popen this module relies on."""

    stdout: object | None
    stderr: object | None

    def wait(self) -> int: ...


ProcessFactory = Callable[[list[str]], FrameStream]


@dataclass(frozen=True)
class ProbeResult:
    fmt: VideoFormat
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    rotation_degrees: int


def _default_probe_runner(args: list[str]) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.run(args, capture_output=True, check=False)


def _default_process_factory(args: list[str]) -> FrameStream:
    # stderr is intentionally discarded, not piped: VideoFrameIterator reads
    # stdout synchronously, chunk by chunk, and nothing anywhere reads
    # stderr until process.wait() at the very end. Piping stderr without
    # ever draining it is a classic deadlock: if ffmpeg writes enough to
    # stderr to fill the OS pipe buffer (very plausible on a real "-v error"
    # run against a corrupted/truncated file -- exactly the case
    # VideoFrameIterator's own corrupted-frame handling anticipates), ffmpeg
    # blocks writing to stderr while we're simultaneously blocked reading
    # stdout, and neither side ever proceeds. DEVNULL removes the pipe
    # (and the possibility of it filling) entirely; nothing in this codebase
    # relies on reading this process's stderr.
    return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


_EXTENSION_TO_FORMAT: dict[str, VideoFormat] = {
    "mp4": VideoFormat.MP4,
    "mov": VideoFormat.MOV,
    "avi": VideoFormat.AVI,
    "mkv": VideoFormat.MKV,
}


def _video_format_from_path(video_path: str) -> VideoFormat:
    suffix = Path(video_path).suffix.lower().lstrip(".")
    try:
        return _EXTENSION_TO_FORMAT[suffix]
    except KeyError as exc:
        raise VideoLoadError(f"Unsupported video extension '{suffix}' for {video_path}") from exc


def _parse_frame_rate(rate: str) -> float:
    if "/" in rate:
        numerator, _, denominator = rate.partition("/")
        denom_value = float(denominator)
        if denom_value == 0:
            raise VideoLoadError(f"Invalid frame rate fraction: {rate}")
        return float(numerator) / denom_value
    return float(rate)


def _parse_rotation(stream: dict) -> int:
    tags = stream.get("tags", {})
    if "rotate" in tags:
        try:
            return int(tags["rotate"]) % 360
        except (TypeError, ValueError):
            return 0
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            try:
                return int(float(side_data["rotation"])) % 360
            except (TypeError, ValueError):
                continue
    return 0


def _find_video_stream(payload: dict, video_path: str) -> dict:
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise VideoLoadError(f"No video stream found in {video_path}")


def _parse_probe_payload(payload: dict, video_path: str) -> ProbeResult:
    stream = _find_video_stream(payload, video_path)

    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VideoLoadError(f"Missing/invalid width or height in probe of {video_path}") from exc

    fps = _parse_frame_rate(stream.get("r_frame_rate", "0/0")) if stream.get("r_frame_rate") else 0.0
    if fps <= 0:
        fps = _parse_frame_rate(stream.get("avg_frame_rate", "0/0")) if stream.get("avg_frame_rate") else 0.0
    if fps <= 0:
        raise VideoLoadError(f"Could not determine frame rate for {video_path}")

    try:
        duration_seconds = float(payload.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration_seconds = 0.0

    nb_frames = stream.get("nb_frames")
    if nb_frames is not None:
        try:
            frame_count = int(nb_frames)
        except (TypeError, ValueError):
            frame_count = round(duration_seconds * fps)
    else:
        frame_count = round(duration_seconds * fps)

    if frame_count <= 0:
        raise VideoLoadError(f"Could not determine frame count for {video_path}")

    return ProbeResult(
        fmt=_video_format_from_path(video_path),
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        rotation_degrees=_parse_rotation(stream),
    )


class FFprobeWrapper:
    """Extracts video metadata by shelling out to ffprobe."""

    def __init__(
        self,
        ffprobe_path: str = "ffprobe",
        runner: ProbeRunner | None = None,
    ) -> None:
        self._ffprobe_path = ffprobe_path
        self._runner = runner or _default_probe_runner
        self._uses_real_binary = runner is None

    def probe(self, video_path: str) -> ProbeResult:
        if self._uses_real_binary and shutil.which(self._ffprobe_path) is None:
            raise VideoLoadError(f"ffprobe binary not found on PATH: {self._ffprobe_path}")

        args = [
            self._ffprobe_path,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        result = self._runner(args)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            raise VideoLoadError(f"ffprobe failed for {video_path}: {stderr.strip()}")

        try:
            payload = json.loads(result.stdout.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VideoLoadError(f"ffprobe returned invalid JSON for {video_path}") from exc

        return _parse_probe_payload(payload, video_path)


class FFmpegFrameReader:
    """Opens a raw-pixel decode stream for a video by shelling out to ffmpeg."""

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        process_factory: ProcessFactory | None = None,
    ) -> None:
        self._ffmpeg_path = ffmpeg_path
        self._process_factory = process_factory or _default_process_factory
        self._uses_real_binary = process_factory is None

    def open_stream(
        self,
        video_path: str,
        width: int,
        height: int,
        pix_fmt: str = "rgb24",
    ) -> FrameStream:
        if self._uses_real_binary and shutil.which(self._ffmpeg_path) is None:
            raise VideoLoadError(f"ffmpeg binary not found on PATH: {self._ffmpeg_path}")

        args = [
            self._ffmpeg_path,
            "-v", "error",
            "-i", video_path,
            "-f", "rawvideo",
            "-pix_fmt", pix_fmt,
            "-s", f"{width}x{height}",
            "-",
        ]
        return self._process_factory(args)
