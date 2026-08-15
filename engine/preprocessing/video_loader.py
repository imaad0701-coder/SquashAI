"""Video loading contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.preprocessing.ffmpeg_wrapper import FFprobeWrapper
from engine.types.video import VideoMetadata


@dataclass(frozen=True)
class VideoLoaderConfig:
    """
    target_fps and max_resolution are declared here but NOT currently wired
    to real resampling/resizing during decode -- found during a code-review
    pass. target_fps only overwrites VideoMetadata.fps (a label), and
    max_resolution isn't read anywhere at all; the actual frame stream still
    decodes at the source video's native fps and resolution regardless of
    either value. ForehandPipeline always passes both as None, so this has
    no effect on any current pipeline output, but a caller setting either to
    resample/resize would get silently ignored, not an error. Implementing
    real fps/resolution conversion is a feature addition, out of scope for
    a bug-fix pass -- this docstring exists so the gap is documented instead
    of silently assumed to work.
    """

    source_path: str
    target_fps: float | None
    max_resolution: tuple[int, int] | None


class VideoLoader(ABC):
    @abstractmethod
    def load_metadata(self, config: VideoLoaderConfig) -> VideoMetadata: ...


class FFmpegVideoLoader(VideoLoader):
    """Loads video metadata via ffprobe."""

    def __init__(self, probe: FFprobeWrapper | None = None) -> None:
        self._probe = probe or FFprobeWrapper()

    def load_metadata(self, config: VideoLoaderConfig) -> VideoMetadata:
        result = self._probe.probe(config.source_path)
        return VideoMetadata(
            path=config.source_path,
            fmt=result.fmt,
            fps=config.target_fps if config.target_fps else result.fps,
            width=result.width,
            height=result.height,
            frame_count=result.frame_count,
            duration_seconds=result.duration_seconds,
        )

    def load_rotation(self, config: VideoLoaderConfig) -> int:
        """Source rotation in degrees, for orientation correction.

        VideoMetadata (the shared contract) has no rotation field, so this
        is exposed as a supplementary lookup rather than folded in there.
        """
        return self._probe.probe(config.source_path).rotation_degrees
