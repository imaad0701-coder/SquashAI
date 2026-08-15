"""Landmark-overlay video rendering for the validation harness. Draws
already-computed LandmarkFrame positions (from a harness.py pipeline run)
back onto the source video.

Re-decodes the source through the EXACT same preprocessing path the
pipeline itself used (FFmpegVideoLoader + FFmpegFrameReader +
VideoFrameIterator + SampledFrameExtractor, all existing/unmodified), so
overlay pixel coordinates line up exactly with the pixel space the pipeline
actually measured landmarks in -- rotation included. Pure visualization: no
engine logic, no new measurements, nothing recomputed.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from engine.preprocessing.ffmpeg_wrapper import FFmpegFrameReader
from engine.preprocessing.frame_extractor import FrameExtractionConfig, SampledFrameExtractor
from engine.preprocessing.frame_iterator import FrameIteratorConfig, VideoFrameIterator
from engine.preprocessing.video_loader import FFmpegVideoLoader, VideoLoaderConfig
from engine.types.landmarks import LandmarkFrame, PoseLandmarkName

# Standard body-graph edges over this engine's 15-point universal landmark
# subset (see engine/types/landmarks.py) -- purely a drawing convenience,
# not a biomechanics measurement.
_SKELETON_EDGES: tuple[tuple[PoseLandmarkName, PoseLandmarkName], ...] = (
    (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.RIGHT_SHOULDER),
    (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.LEFT_ELBOW),
    (PoseLandmarkName.LEFT_ELBOW, PoseLandmarkName.LEFT_WRIST),
    (PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_ELBOW),
    (PoseLandmarkName.RIGHT_ELBOW, PoseLandmarkName.RIGHT_WRIST),
    (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.LEFT_HIP),
    (PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_HIP),
    (PoseLandmarkName.LEFT_HIP, PoseLandmarkName.RIGHT_HIP),
    (PoseLandmarkName.LEFT_HIP, PoseLandmarkName.LEFT_KNEE),
    (PoseLandmarkName.LEFT_KNEE, PoseLandmarkName.LEFT_ANKLE),
    (PoseLandmarkName.LEFT_ANKLE, PoseLandmarkName.LEFT_FOOT_INDEX),
    (PoseLandmarkName.RIGHT_HIP, PoseLandmarkName.RIGHT_KNEE),
    (PoseLandmarkName.RIGHT_KNEE, PoseLandmarkName.RIGHT_ANKLE),
    (PoseLandmarkName.RIGHT_ANKLE, PoseLandmarkName.RIGHT_FOOT_INDEX),
)

_LEFT_COLOR_BGR = (255, 140, 0)
_RIGHT_COLOR_BGR = (0, 140, 255)
_MIDLINE_COLOR_BGR = (0, 220, 0)
_LOW_CONFIDENCE_COLOR_BGR = (0, 0, 255)

_LEFT_NAMES = {name for name in PoseLandmarkName if name.value.startswith("left_")}
_RIGHT_NAMES = {name for name in PoseLandmarkName if name.value.startswith("right_")}

_MIN_VISIBILITY_FOR_CONFIDENT_COLOR = 0.5
_MIN_PRESENCE_FOR_CONFIDENT_COLOR = 0.5


def _edge_color(a: PoseLandmarkName) -> tuple[int, int, int]:
    if a in _LEFT_NAMES:
        return _LEFT_COLOR_BGR
    if a in _RIGHT_NAMES:
        return _RIGHT_COLOR_BGR
    return _MIDLINE_COLOR_BGR


def _point_color(name: PoseLandmarkName, landmark) -> tuple[int, int, int]:
    if (
        landmark.visibility < _MIN_VISIBILITY_FOR_CONFIDENT_COLOR
        or landmark.presence < _MIN_PRESENCE_FOR_CONFIDENT_COLOR
    ):
        return _LOW_CONFIDENCE_COLOR_BGR
    return _edge_color(name)


def _draw_frame(image_bgr: np.ndarray, landmark_frame: LandmarkFrame, racket_side: str | None) -> None:
    landmarks = landmark_frame.pose_landmarks

    for a, b in _SKELETON_EDGES:
        la, lb = landmarks.get(a), landmarks.get(b)
        if la is None or lb is None:
            continue
        pt_a = (int(round(la.position.x)), int(round(la.position.y)))
        pt_b = (int(round(lb.position.x)), int(round(lb.position.y)))
        is_racket_edge = (
            racket_side is not None and a.value.startswith(racket_side) and b.value.startswith(racket_side)
        )
        thickness = 4 if is_racket_edge else 2
        cv2.line(image_bgr, pt_a, pt_b, _edge_color(a), thickness, lineType=cv2.LINE_AA)

    for name, landmark in landmarks.items():
        pt = (int(round(landmark.position.x)), int(round(landmark.position.y)))
        cv2.circle(image_bgr, pt, radius=5, color=_point_color(name, landmark), thickness=-1, lineType=cv2.LINE_AA)

    if racket_side is not None:
        cv2.putText(
            image_bgr, f"racket side: {racket_side}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
    cv2.putText(
        image_bgr, f"frame {landmark_frame.timing.frame_index}", (10, image_bgr.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )


def render_landmarks_video(
    video_path: str,
    landmark_frames: tuple[LandmarkFrame, ...],
    output_path: str,
    rotation_degrees: int = 0,
    racket_side: str | None = None,
    fps: float | None = None,
) -> str:
    """Re-decodes `video_path` and draws each frame's already-computed
    landmarks on top of it, writing an .mp4 to `output_path`. Only frames
    present in `landmark_frames` are rendered."""
    frames_by_index = {frame.timing.frame_index: frame for frame in landmark_frames}
    if not frames_by_index:
        raise ValueError("No landmark frames to render")

    video_loader = FFmpegVideoLoader()
    loader_config = VideoLoaderConfig(source_path=video_path, target_fps=None, max_resolution=None)
    metadata = video_loader.load_metadata(loader_config)

    frame_metas = SampledFrameExtractor(metadata).extract(
        FrameExtractionConfig(stride=1, start_frame_index=0, end_frame_index=max(frames_by_index) + 1)
    )
    frame_iterator = VideoFrameIterator(FFmpegFrameReader(), FrameIteratorConfig())

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    output_fps = fps or metadata.fps or 30.0
    frames_written = 0

    try:
        for frame_meta, image in frame_iterator.iter_frames(
            video_path, metadata, frame_metas, rotation_degrees=rotation_degrees
        ):
            landmark_frame = frames_by_index.get(frame_meta.index)
            if landmark_frame is None:
                continue

            array = np.frombuffer(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))
            image_bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            _draw_frame(image_bgr, landmark_frame, racket_side)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, output_fps, (image.width, image.height))
                if not writer.isOpened():
                    raise RuntimeError(f"cv2.VideoWriter failed to open {output_path}")
            writer.write(image_bgr)
            frames_written += 1
    finally:
        if writer is not None:
            writer.release()

    if frames_written == 0:
        raise ValueError("No frames were written -- landmark_frames indices never matched decoded frame indices")

    return output_path
