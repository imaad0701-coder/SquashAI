"""Frame synchronization and timing.

Produces the *exact* per-frame timing (FrameTiming) that tracking outputs
carry, from the video's real per-frame presentation timestamps — never from
a constant 1/fps assumption. This is what makes variable-fps footage and
dropped frames safe to reason about: every FrameTiming.delta_time_ms is the
actual gap since whatever frame preceded it, not a fixed step.

Two pieces:
- FFprobeFrameTimingSource: reads real per-frame pts_time values via ffprobe
  (lazily/injectably, like engine.preprocessing.ffmpeg_wrapper). Some frames
  may come back as unresolved (ffprobe reports "N/A" for a corrupted frame).
- FrameSynchronizer: pure logic that turns a (possibly incomplete) map of
  frame_index -> real seconds into a dense FrameTiming timeline, linearly
  interpolating/extrapolating any gaps from the nearest known real points.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Callable, Mapping, Sequence

from engine.exceptions import FrameTimingError
from engine.types.video import FrameTiming

ProbeRunner = Callable[[list[str]], "subprocess.CompletedProcess[bytes]"]


def _default_probe_runner(args: list[str]) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.run(args, capture_output=True, check=False)


class FFprobeFrameTimingSource:
    """Reads each frame's real presentation timestamp (pts_time) via ffprobe."""

    def __init__(
        self,
        ffprobe_path: str = "ffprobe",
        runner: ProbeRunner | None = None,
    ) -> None:
        self._ffprobe_path = ffprobe_path
        self._runner = runner or _default_probe_runner
        self._uses_real_binary = runner is None

    def probe_frame_timestamps(self, video_path: str) -> dict[int, float | None]:
        if self._uses_real_binary and shutil.which(self._ffprobe_path) is None:
            raise FrameTimingError(f"ffprobe binary not found on PATH: {self._ffprobe_path}")

        args = [
            self._ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pts_time",
            "-of", "csv=p=0",
            video_path,
        ]
        result = self._runner(args)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            raise FrameTimingError(f"ffprobe frame-timing probe failed for {video_path}: {stderr.strip()}")

        try:
            text = result.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FrameTimingError(f"ffprobe returned undecodable output for {video_path}") from exc

        timestamps: dict[int, float | None] = {}
        for index, line in enumerate(line for line in text.splitlines() if line.strip()):
            # ffprobe's csv writer appends an extra (often empty) trailing
            # column whenever a frame carries ANY side_data_list entry, even
            # one that was never requested via -show_entries -- confirmed
            # against a real ffmpeg 8.1.2 encode where frame 0 alone carried
            # an empty side-data blob, producing "0.000000," instead of
            # "0.000000". Real camera/phone-recorded videos routinely attach
            # side data (rotation matrices, HDR metadata, ...) to frames, so
            # this isn't specific to synthetic test files. Only the single
            # requested field (pts_time) is ever first, so taking just the
            # first comma-separated value is robust to any such trailing
            # columns without needing to know how many there are.
            value = line.strip().split(",", 1)[0]
            timestamps[index] = None if value in ("N/A", "") else float(value)

        if not timestamps:
            raise FrameTimingError(f"No frame timestamps found for {video_path}")

        return timestamps


class FrameSynchronizer:
    """Turns a (possibly incomplete) frame_index -> seconds map into a dense
    FrameTiming timeline, filling gaps by interpolation/extrapolation.

    Never assumes constant spacing: every delta_time_ms is computed from the
    actual timestamps of whichever frames end up adjacent in the requested
    sequence, and any interpolated/extrapolated point is derived from the
    nearest real, known timestamps rather than a global average rate.
    """

    def __init__(self, frame_timestamps_seconds: Mapping[int, float | None]) -> None:
        self._frame_timestamps_seconds = dict(frame_timestamps_seconds)

    def build_timeline(self, frame_indices: Sequence[int]) -> tuple[FrameTiming, ...]:
        if not frame_indices:
            return ()

        resolved = self._resolve(frame_indices)

        timings: list[FrameTiming] = []
        previous_ms: float | None = None
        for frame_index in frame_indices:
            timestamp_ms = resolved[frame_index] * 1000.0
            delta_time_ms = 0.0 if previous_ms is None else timestamp_ms - previous_ms
            timings.append(
                FrameTiming(frame_index=frame_index, timestamp_ms=timestamp_ms, delta_time_ms=delta_time_ms)
            )
            previous_ms = timestamp_ms
        return tuple(timings)

    def _resolve(self, frame_indices: Sequence[int]) -> dict[int, float]:
        known_indices = sorted(
            index for index, seconds in self._frame_timestamps_seconds.items() if seconds is not None
        )
        if not known_indices:
            raise FrameTimingError("No known frame timestamps available to synchronize against")

        resolved: dict[int, float] = {}
        for frame_index in frame_indices:
            direct = self._frame_timestamps_seconds.get(frame_index)
            resolved[frame_index] = (
                direct if direct is not None else self._interpolate_or_extrapolate(frame_index, known_indices)
            )
        return resolved

    def _interpolate_or_extrapolate(self, frame_index: int, known_indices: list[int]) -> float:
        before = max((index for index in known_indices if index < frame_index), default=None)
        after = min((index for index in known_indices if index > frame_index), default=None)

        if before is not None and after is not None:
            before_seconds = self._frame_timestamps_seconds[before]
            after_seconds = self._frame_timestamps_seconds[after]
            ratio = (frame_index - before) / (after - before)
            return before_seconds + ratio * (after_seconds - before_seconds)

        if before is not None:
            return self._extrapolate(frame_index, anchor=before, known_indices=known_indices, forward=True)
        if after is not None:
            return self._extrapolate(frame_index, anchor=after, known_indices=known_indices, forward=False)

        raise FrameTimingError(f"Cannot resolve timestamp for frame {frame_index}: no bracketing data")

    def _extrapolate(
        self, frame_index: int, anchor: int, known_indices: list[int], forward: bool
    ) -> float:
        # Derive a local rate from the two nearest known points on the same
        # side, rather than assuming any fixed global fps.
        if forward:
            second = max((index for index in known_indices if index < anchor), default=None)
        else:
            second = min((index for index in known_indices if index > anchor), default=None)

        anchor_seconds = self._frame_timestamps_seconds[anchor]
        if second is None:
            return anchor_seconds  # only one known point on this side: hold it flat

        second_seconds = self._frame_timestamps_seconds[second]
        rate_seconds_per_index = (anchor_seconds - second_seconds) / (anchor - second)
        return anchor_seconds + rate_seconds_per_index * (frame_index - anchor)
