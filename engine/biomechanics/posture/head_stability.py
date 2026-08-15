"""Head stability: how much the head (nose landmark) moves within a short
window of frames, centered on the frame being scored — the pose analogue of
the coaching cue "keep your head still through the swing".

Unlike the single-frame AngleCalculator family, this genuinely needs a
trajectory: "stability" is meaningless for one frame in isolation. Displacement
is normalized by shoulder width at the center frame so the metric stays
comparable regardless of how large the player appears (distance from camera,
video resolution) — the same reasoning as CenterOfMassCalculator.height_ratio.
"""

from __future__ import annotations

from typing import Sequence

from engine.biomechanics.posture.angle_calculator import MIN_LANDMARK_PRESENCE, MIN_LANDMARK_VISIBILITY
from engine.biomechanics.scoring import score_within_range
from engine.types.biomechanics import HeadStabilityMeasurement
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.scoring import BenchmarkSpec, ScoreResult
from engine.utils.geometry import distance, is_finite_point, magnitude, vector_between
from engine.utils.math_utils import EPSILON

DEFAULT_WINDOW_SIZE = 5


def _usable(landmark: Landmark | None) -> bool:
    if landmark is None:
        return False
    if landmark.visibility < MIN_LANDMARK_VISIBILITY or landmark.presence < MIN_LANDMARK_PRESENCE:
        return False
    return is_finite_point(landmark.position)


class HeadStabilityCalculator:
    def calculate(
        self, frames: Sequence[LandmarkFrame], center_index: int, window_size: int = DEFAULT_WINDOW_SIZE
    ) -> HeadStabilityMeasurement:
        try:
            return self._calculate(frames, center_index, window_size)
        except Exception:
            return self._invalid(frames[center_index])

    def _calculate(
        self, frames: Sequence[LandmarkFrame], center_index: int, window_size: int
    ) -> HeadStabilityMeasurement:
        center_frame = frames[center_index]
        center_nose = center_frame.pose_landmarks.get(PoseLandmarkName.NOSE)
        if not _usable(center_nose):
            return self._invalid(center_frame)
        assert center_nose is not None

        half_window = max(window_size, 1) // 2
        start = max(0, center_index - half_window)
        end = min(len(frames), center_index + half_window + 1)

        nose_positions: list[Point3D] = []
        for frame in frames[start:end]:
            nose = frame.pose_landmarks.get(PoseLandmarkName.NOSE)
            if _usable(nose):
                assert nose is not None
                nose_positions.append(nose.position)

        if len(nose_positions) < 2:
            return self._invalid(center_frame)  # only the center frame itself (or nothing): no window to compare

        mean_position = Point3D(
            x=sum(p.x for p in nose_positions) / len(nose_positions),
            y=sum(p.y for p in nose_positions) / len(nose_positions),
            z=sum(p.z for p in nose_positions) / len(nose_positions),
        )
        displacement = distance(center_nose.position, mean_position)

        left_shoulder = center_frame.pose_landmarks.get(PoseLandmarkName.LEFT_SHOULDER)
        right_shoulder = center_frame.pose_landmarks.get(PoseLandmarkName.RIGHT_SHOULDER)
        if not (_usable(left_shoulder) and _usable(right_shoulder)):
            return self._invalid(center_frame)
        assert left_shoulder is not None and right_shoulder is not None
        shoulder_width = magnitude(vector_between(left_shoulder.position, right_shoulder.position))
        if shoulder_width <= EPSILON:
            return self._invalid(center_frame)

        return HeadStabilityMeasurement(
            frame_index=center_frame.timing.frame_index,
            timestamp_ms=center_frame.timing.timestamp_ms,
            normalized_displacement=displacement / shoulder_width,
            is_valid=True,
        )

    def validate(
        self, frames: Sequence[LandmarkFrame], center_index: int, window_size: int = DEFAULT_WINDOW_SIZE
    ) -> bool:
        return self.calculate(frames, center_index, window_size).is_valid

    def confidence(
        self, frames: Sequence[LandmarkFrame], center_index: int, window_size: int = DEFAULT_WINDOW_SIZE
    ) -> float:
        center_frame = frames[center_index]
        nose = center_frame.pose_landmarks.get(PoseLandmarkName.NOSE)
        left_shoulder = center_frame.pose_landmarks.get(PoseLandmarkName.LEFT_SHOULDER)
        right_shoulder = center_frame.pose_landmarks.get(PoseLandmarkName.RIGHT_SHOULDER)
        if not (_usable(nose) and _usable(left_shoulder) and _usable(right_shoulder)):
            return 0.0
        assert nose is not None and left_shoulder is not None and right_shoulder is not None
        return min(
            min(nose.visibility, nose.presence),
            min(left_shoulder.visibility, left_shoulder.presence),
            min(right_shoulder.visibility, right_shoulder.presence),
        )

    def benchmark(self, measurement: HeadStabilityMeasurement, spec: BenchmarkSpec) -> ScoreResult:
        if not measurement.is_valid or measurement.normalized_displacement is None:
            return ScoreResult(metric_name=spec.metric_name, raw_value=0.0, band=spec.band, score=0.0)
        raw = measurement.normalized_displacement
        return ScoreResult(
            metric_name=spec.metric_name, raw_value=raw, band=spec.band, score=score_within_range(raw, spec)
        )

    def _invalid(self, frame: LandmarkFrame) -> HeadStabilityMeasurement:
        return HeadStabilityMeasurement(
            frame_index=frame.timing.frame_index,
            timestamp_ms=frame.timing.timestamp_ms,
            normalized_displacement=None,
            is_valid=False,
        )
