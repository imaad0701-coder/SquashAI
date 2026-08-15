"""Exception hierarchy shared across the engine."""

from __future__ import annotations


class SquashAIError(Exception):
    """Base class for all engine-raised errors."""


class VideoLoadError(SquashAIError):
    """Raised when a source video cannot be opened or decoded."""


class CorruptedFrameError(VideoLoadError):
    """Raised when a decoded frame's byte stream is truncated or malformed."""


class FrameTimingError(VideoLoadError):
    """Raised when per-frame presentation timestamps cannot be resolved."""


class CalibrationError(SquashAIError):
    """Raised when camera calibration data is missing or invalid."""


class PoseDetectionError(SquashAIError):
    """Raised when pose/racket tracking fails for a frame or clip."""


class PhaseDetectionError(SquashAIError):
    """Raised when swing phase segmentation fails."""


class PipelineError(SquashAIError):
    """Raised when a shot pipeline cannot complete its run."""


class ScoringError(SquashAIError):
    """Raised when scoring cannot be computed against benchmarks."""


class RepositoryError(SquashAIError):
    """Raised when a persistence read/write operation fails."""
