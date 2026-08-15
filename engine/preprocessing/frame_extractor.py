"""Frame extraction contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.types.video import FrameMeta, VideoMetadata


@dataclass(frozen=True)
class FrameExtractionConfig:
    stride: int
    start_frame_index: int
    end_frame_index: int | None


class FrameExtractor(ABC):
    @abstractmethod
    def extract(self, config: FrameExtractionConfig) -> tuple[FrameMeta, ...]: ...


class SampledFrameExtractor(FrameExtractor):
    """Builds the sampled frame list (with timestamps) for a known video.

    FrameMeta.timestamp_seconds here is a constant-fps *estimate*, only used
    to decide which frame indices to decode. It is not the "exact" timestamp
    — see engine.preprocessing.frame_sync.FrameSynchronizer, which derives
    real per-frame timing for LandmarkFrame.timing from the video itself.
    """

    def __init__(self, metadata: VideoMetadata) -> None:
        self._metadata = metadata

    def extract(self, config: FrameExtractionConfig) -> tuple[FrameMeta, ...]:
        if config.stride <= 0:
            raise ValueError(f"stride must be positive, got {config.stride}")
        if config.start_frame_index < 0:
            raise ValueError(f"start_frame_index must be >= 0, got {config.start_frame_index}")

        last_index = self._metadata.frame_count - 1
        end = last_index if config.end_frame_index is None else min(config.end_frame_index, last_index)

        return tuple(
            FrameMeta(index=index, timestamp_seconds=index / self._metadata.fps)
            for index in range(config.start_frame_index, end + 1, config.stride)
        )
