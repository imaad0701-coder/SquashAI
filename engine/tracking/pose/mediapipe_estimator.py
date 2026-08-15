"""MediaPipe-backed pose estimation.

Only `MediaPipePoseDetectorAdapter` and `create_mediapipe_pose_detector` touch
mediapipe/numpy, and both import those libraries lazily inside their own
bodies rather than at module scope. Everything else here depends only on the
small `PoseDetector` protocol below, so `MediaPipePoseEstimator` can be
exercised with a fake detector without mediapipe or numpy installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Protocol, Sequence

from engine.exceptions import PoseDetectionError
from engine.preprocessing.frame_iterator import FrameImage
from engine.tracking.pose.pose_estimator import PoseEstimator, PoseEstimatorConfig
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.video import FrameMeta, FrameTiming


class RawPoseLandmark(Protocol):
    x: float
    y: float
    z: float
    visibility: float
    presence: float


class RawPoseResult(Protocol):
    landmarks: Sequence[RawPoseLandmark] | None


class PoseDetector(Protocol):
    """The only seam between this package and the real MediaPipe library."""

    def process(self, image: FrameImage) -> RawPoseResult: ...


# Fixed indices from mediapipe.solutions.pose.PoseLandmark (33-point body model),
# mapped down to the 13-entry subset this engine models.
_MEDIAPIPE_LANDMARK_INDEX: dict[PoseLandmarkName, int] = {
    PoseLandmarkName.NOSE: 0,
    PoseLandmarkName.LEFT_SHOULDER: 11,
    PoseLandmarkName.RIGHT_SHOULDER: 12,
    PoseLandmarkName.LEFT_ELBOW: 13,
    PoseLandmarkName.RIGHT_ELBOW: 14,
    PoseLandmarkName.LEFT_WRIST: 15,
    PoseLandmarkName.RIGHT_WRIST: 16,
    PoseLandmarkName.LEFT_HIP: 23,
    PoseLandmarkName.RIGHT_HIP: 24,
    PoseLandmarkName.LEFT_KNEE: 25,
    PoseLandmarkName.RIGHT_KNEE: 26,
    PoseLandmarkName.LEFT_ANKLE: 27,
    PoseLandmarkName.RIGHT_ANKLE: 28,
    PoseLandmarkName.LEFT_FOOT_INDEX: 31,
    PoseLandmarkName.RIGHT_FOOT_INDEX: 32,
}


class MediaPipePoseEstimator(PoseEstimator):
    """Converts raw MediaPipe detections into LandmarkFrame instances.

    `estimate_sequence` is the primary path: MediaPipe's tracker is stateful
    across ordered `.process()` calls on one `Pose` instance, so frames must
    be fed through the same detector in ascending order for cross-frame
    tracking continuity to work. It takes real, synchronized `FrameTiming`
    (see engine.preprocessing.frame_sync.FrameSynchronizer) so every emitted
    LandmarkFrame carries an exact timestamp rather than an estimate.

    `estimate` exists only to satisfy the `PoseEstimator` ABC, which takes a
    `FrameMeta` (no image, no synchronized timing) — it fetches pixel data via
    the injected `frame_image_provider` and builds a degenerate FrameTiming
    from that FrameMeta's rough timestamp (delta_time_ms=0.0, since no prior
    frame is known in this single-frame path). Prefer `estimate_sequence`.
    """

    def __init__(
        self,
        detector: PoseDetector,
        config: PoseEstimatorConfig,
        frame_image_provider: Callable[[FrameMeta], FrameImage] | None = None,
    ) -> None:
        self._detector = detector
        self._config = config
        self._frame_image_provider = frame_image_provider

    def estimate_sequence(
        self, frames: Iterable[tuple[FrameTiming, FrameImage]]
    ) -> Iterator[LandmarkFrame]:
        for timing, image in frames:
            raw_result = self._detector.process(image)
            yield self._to_landmark_frame(timing, raw_result)

    def estimate(self, frame: FrameMeta, config: PoseEstimatorConfig) -> LandmarkFrame:
        if self._frame_image_provider is None:
            raise PoseDetectionError(
                "estimate() requires a frame_image_provider to fetch pixel data for a single frame"
            )
        image = self._frame_image_provider(frame)
        raw_result = self._detector.process(image)
        timing = FrameTiming(
            frame_index=frame.index,
            timestamp_ms=frame.timestamp_seconds * 1000.0,
            delta_time_ms=0.0,
        )
        return self._to_landmark_frame(timing, raw_result)

    def _to_landmark_frame(self, timing: FrameTiming, raw_result: RawPoseResult) -> LandmarkFrame:
        if raw_result.landmarks is None:
            return LandmarkFrame(timing=timing, pose_landmarks={}, racket_landmarks={})

        pose_landmarks: dict[PoseLandmarkName, Landmark] = {}
        for name, index in _MEDIAPIPE_LANDMARK_INDEX.items():
            if index >= len(raw_result.landmarks):
                continue
            raw = raw_result.landmarks[index]
            pose_landmarks[name] = Landmark(
                position=Point3D(x=raw.x, y=raw.y, z=raw.z),
                visibility=raw.visibility,
                presence=raw.presence,
            )
        return LandmarkFrame(timing=timing, pose_landmarks=pose_landmarks, racket_landmarks={})


@dataclass(frozen=True)
class _SimpleRawPoseLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass(frozen=True)
class _SimpleRawPoseResult:
    landmarks: Sequence[_SimpleRawPoseLandmark] | None


class MediaPipePoseDetectorAdapter:
    """Wraps a real mediapipe.solutions.pose.Pose instance as a PoseDetector.

    This is the actual mediapipe/numpy integration boundary; it cannot be
    exercised without both installed.
    """

    def __init__(self, mediapipe_pose: object) -> None:
        self._mediapipe_pose = mediapipe_pose

    def process(self, image: FrameImage) -> RawPoseResult:
        import numpy as np

        array = np.frombuffer(image.data, dtype=np.uint8).reshape(
            (image.height, image.width, image.channels)
        )
        results = self._mediapipe_pose.process(array)  # type: ignore[attr-defined]
        if results.pose_landmarks is None:
            return _SimpleRawPoseResult(landmarks=None)

        # MediaPipe reports x/y/z normalized to [0, 1] (z on roughly the same
        # scale as x), not pixels. Denormalize here so every LandmarkFrame
        # this engine produces is actually in pixel space, matching what
        # every kinematics/joint-angle docstring in this project claims.
        #
        # The legacy Solutions API (unlike the newer Tasks API) never
        # populates `presence` at all — the protobuf field exists but is
        # simply never set, so it silently reads back as 0.0 for every
        # landmark on every real detection. Confirmed empirically against
        # mediapipe==0.10.9. Using that verbatim would fail every downstream
        # presence threshold regardless of detection quality, so visibility
        # (the one real per-landmark confidence signal this API provides) is
        # used as its stand-in.
        landmarks = [
            _SimpleRawPoseLandmark(
                x=landmark.x * image.width,
                y=landmark.y * image.height,
                z=landmark.z * image.width,
                visibility=landmark.visibility,
                presence=landmark.visibility,
            )
            for landmark in results.pose_landmarks.landmark
        ]
        return _SimpleRawPoseResult(landmarks=landmarks)


def create_mediapipe_pose_detector(config: PoseEstimatorConfig) -> PoseDetector:
    """Lazily constructs a real MediaPipe Pose detector.

    mediapipe is a heavy optional runtime dependency, so it is imported here
    rather than at module scope; this raises PoseDetectionError with a clear
    message if it isn't installed, instead of failing to import this module.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise PoseDetectionError(
            "mediapipe is not installed; install it to use create_mediapipe_pose_detector()"
        ) from exc

    if not hasattr(mp, "solutions"):
        # mediapipe's Windows wheels dropped the legacy Solutions API from
        # 0.10.35 onward (and entirely in 1.0.0) in favor of the Tasks API,
        # which this adapter does not implement. Pin mediapipe==0.10.9 (see
        # requirements.txt) rather than "latest".
        raise PoseDetectionError(
            f"mediapipe {mp.__version__} has no `solutions` module (the legacy "
            "Solutions API was removed in newer releases); install mediapipe==0.10.9"
        )

    mediapipe_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=config.model_complexity,
        min_detection_confidence=config.tracker_config.min_detection_confidence,
        min_tracking_confidence=config.tracker_config.min_tracking_confidence,
    )
    return MediaPipePoseDetectorAdapter(mediapipe_pose)
