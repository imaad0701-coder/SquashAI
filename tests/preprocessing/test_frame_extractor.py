"""Tests for SampledFrameExtractor."""

from __future__ import annotations

import unittest

from engine.preprocessing.frame_extractor import FrameExtractionConfig, SampledFrameExtractor
from engine.types.video import VideoFormat, VideoMetadata


def _metadata(**overrides: object) -> VideoMetadata:
    defaults = dict(
        path="clip.mp4",
        fmt=VideoFormat.MP4,
        fps=30.0,
        width=1920,
        height=1080,
        frame_count=100,
        duration_seconds=100 / 30,
    )
    defaults.update(overrides)
    return VideoMetadata(**defaults)


class SampledFrameExtractorTests(unittest.TestCase):
    def test_extract_full_range_with_stride_one(self) -> None:
        extractor = SampledFrameExtractor(_metadata(frame_count=5, fps=10.0))
        config = FrameExtractionConfig(stride=1, start_frame_index=0, end_frame_index=None)

        frames = extractor.extract(config)

        self.assertEqual([f.index for f in frames], [0, 1, 2, 3, 4])
        self.assertEqual([f.timestamp_seconds for f in frames], [0.0, 0.1, 0.2, 0.3, 0.4])

    def test_extract_applies_stride(self) -> None:
        extractor = SampledFrameExtractor(_metadata(frame_count=10, fps=10.0))
        config = FrameExtractionConfig(stride=3, start_frame_index=0, end_frame_index=None)

        frames = extractor.extract(config)

        self.assertEqual([f.index for f in frames], [0, 3, 6, 9])

    def test_extract_honors_explicit_end_frame_index(self) -> None:
        extractor = SampledFrameExtractor(_metadata(frame_count=100, fps=25.0))
        config = FrameExtractionConfig(stride=1, start_frame_index=2, end_frame_index=4)

        frames = extractor.extract(config)

        self.assertEqual([f.index for f in frames], [2, 3, 4])

    def test_extract_clamps_end_frame_index_to_video_length(self) -> None:
        extractor = SampledFrameExtractor(_metadata(frame_count=5, fps=10.0))
        config = FrameExtractionConfig(stride=1, start_frame_index=0, end_frame_index=999)

        frames = extractor.extract(config)

        self.assertEqual([f.index for f in frames], [0, 1, 2, 3, 4])

    def test_extract_rejects_non_positive_stride(self) -> None:
        extractor = SampledFrameExtractor(_metadata())
        config = FrameExtractionConfig(stride=0, start_frame_index=0, end_frame_index=None)

        with self.assertRaises(ValueError):
            extractor.extract(config)

    def test_extract_rejects_negative_start_index(self) -> None:
        extractor = SampledFrameExtractor(_metadata())
        config = FrameExtractionConfig(stride=1, start_frame_index=-1, end_frame_index=None)

        with self.assertRaises(ValueError):
            extractor.extract(config)


if __name__ == "__main__":
    unittest.main()
