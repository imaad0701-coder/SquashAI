"""Decoded-frame iteration: resolution normalization, orientation correction,
and corrupted-frame handling on top of the raw ffmpeg byte stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from engine.exceptions import CorruptedFrameError
from engine.preprocessing.ffmpeg_wrapper import FFmpegFrameReader
from engine.types.video import FrameMeta, VideoMetadata

_CHANNELS = 3  # rgb24


@dataclass(frozen=True)
class FrameImage:
    width: int
    height: int
    channels: int
    data: bytes


@dataclass(frozen=True)
class FrameIteratorConfig:
    pix_fmt: str = "rgb24"
    skip_corrupted: bool = True
    max_consecutive_corrupted: int = 5


def resize_nearest(image: FrameImage, target_width: int, target_height: int) -> FrameImage:
    """Nearest-neighbor resample to (target_width, target_height)."""
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Target resolution must be positive, got ({target_width}, {target_height})")
    if target_width == image.width and target_height == image.height:
        return image

    src_w, src_h, ch = image.width, image.height, image.channels
    out = bytearray(target_width * target_height * ch)
    for ty in range(target_height):
        sy = min(src_h - 1, (ty * src_h) // target_height)
        row_base = sy * src_w
        for tx in range(target_width):
            sx = min(src_w - 1, (tx * src_w) // target_width)
            src_offset = (row_base + sx) * ch
            dst_offset = (ty * target_width + tx) * ch
            out[dst_offset : dst_offset + ch] = image.data[src_offset : src_offset + ch]
    return FrameImage(width=target_width, height=target_height, channels=ch, data=bytes(out))


def rotate_frame(image: FrameImage, degrees: int) -> FrameImage:
    """Rotate a frame clockwise by 0/90/180/270 degrees to correct orientation."""
    normalized = degrees % 360
    if normalized == 0:
        return image
    if normalized not in (90, 180, 270):
        raise ValueError(f"Unsupported rotation angle: {degrees}")

    w, h, ch = image.width, image.height, image.channels
    out_w, out_h = (w, h) if normalized == 180 else (h, w)
    out = bytearray(out_w * out_h * ch)

    for y in range(h):
        row_base = y * w
        for x in range(w):
            src_offset = (row_base + x) * ch
            pixel = image.data[src_offset : src_offset + ch]
            if normalized == 90:
                nx, ny = h - 1 - y, x
            elif normalized == 180:
                nx, ny = w - 1 - x, h - 1 - y
            else:  # 270
                nx, ny = y, w - 1 - x
            dst_offset = (ny * out_w + nx) * ch
            out[dst_offset : dst_offset + ch] = pixel

    return FrameImage(width=out_w, height=out_h, channels=ch, data=bytes(out))


def _read_exact(stream: object, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)  # type: ignore[attr-defined]
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class VideoFrameIterator:
    """Decodes only the requested frames from a video, in ascending order."""

    def __init__(
        self,
        frame_reader: FFmpegFrameReader,
        config: FrameIteratorConfig | None = None,
    ) -> None:
        self._frame_reader = frame_reader
        self._config = config or FrameIteratorConfig()

    def iter_frames(
        self,
        video_path: str,
        metadata: VideoMetadata,
        frames: tuple[FrameMeta, ...],
        target_resolution: tuple[int, int] | None = None,
        rotation_degrees: int = 0,
    ) -> Iterator[tuple[FrameMeta, FrameImage]]:
        if not frames:
            return

        requested_by_index = {frame.index: frame for frame in frames}
        max_index = max(requested_by_index)
        frame_size = metadata.width * metadata.height * _CHANNELS

        process = self._frame_reader.open_stream(
            video_path, metadata.width, metadata.height, self._config.pix_fmt
        )
        stdout = process.stdout
        if stdout is None:
            raise CorruptedFrameError(f"ffmpeg process for {video_path} produced no stdout stream")

        consecutive_corrupted = 0
        try:
            for index in range(max_index + 1):
                chunk = _read_exact(stdout, frame_size)

                if len(chunk) == 0:
                    break  # clean end of stream

                if len(chunk) != frame_size:
                    consecutive_corrupted += 1
                    if (
                        not self._config.skip_corrupted
                        or consecutive_corrupted > self._config.max_consecutive_corrupted
                    ):
                        raise CorruptedFrameError(
                            f"Corrupted frame at index {index} in {video_path}: "
                            f"expected {frame_size} bytes, got {len(chunk)}"
                        )
                    continue

                consecutive_corrupted = 0
                if index not in requested_by_index:
                    continue

                image = FrameImage(
                    width=metadata.width, height=metadata.height, channels=_CHANNELS, data=chunk
                )
                if rotation_degrees:
                    image = rotate_frame(image, rotation_degrees)
                if target_resolution is not None:
                    image = resize_nearest(image, *target_resolution)

                yield requested_by_index[index], image
        finally:
            stdout.close()  # type: ignore[attr-defined]
            process.wait()
