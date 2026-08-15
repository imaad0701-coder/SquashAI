"""Video and frame metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VideoFormat(Enum):
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fmt: VideoFormat
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class FrameMeta:
    index: int
    timestamp_seconds: float


@dataclass(frozen=True)
class FrameTiming:
    """Exact, synchronized per-frame timing (see engine.preprocessing.frame_sync).

    Unlike FrameMeta.timestamp_seconds (a constant-fps estimate used only to
    pick which frames to decode), timestamp_ms is derived from the video's
    real per-frame presentation timestamps, and delta_time_ms is the actual
    gap from whichever frame preceded it in a given sequence — never a fixed
    1/fps step. This is what tracking outputs (LandmarkFrame) carry.
    """

    frame_index: int
    timestamp_ms: float
    delta_time_ms: float
