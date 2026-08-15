"""Whole-body center of mass: a segmental estimate built from Winter's
anthropometric mass-fraction tables (Winter, "Biomechanics and Motor Control
of Human Movement"), applied to joint-center landmarks rather than true
segment endpoints (we don't track e.g. a distinct hand landmark, only the
wrist) — so this is a documented approximation, not a lab-grade measurement.

Segment mass fractions used (of total body mass), each segment's center
taken as the midpoint of its two bounding landmarks (or the landmark itself
for the single-point head segment):

    head (nose)                       0.081
    trunk (mid-shoulder..mid-hip)     0.497
    upper arm (shoulder..elbow) x2    0.028 each
    forearm+hand (elbow..wrist) x2    0.022 each  (0.016 + 0.006 combined,
                                                    since this engine has no
                                                    separate hand landmark)
    thigh (hip..knee) x2              0.100 each
    shank (knee..ankle) x2            0.0465 each
    foot (ankle..foot_index) x2       0.0145 each

Fractions sum to 1.0. Any segment whose landmarks aren't all resolvable is
dropped and the remaining weights renormalized, EXCEPT the trunk: at ~50%
of body mass it anchors the whole estimate, so a frame without a resolvable
trunk is reported invalid rather than silently renormalizing over the
lighter, more errorprone limb segments alone.
"""

from __future__ import annotations

from typing import Final

from engine.biomechanics.posture.angle_calculator import (
    MIN_LANDMARK_PRESENCE,
    MIN_LANDMARK_VISIBILITY,
    midpoint,
)
from engine.biomechanics.scoring import score_within_range
from engine.types.biomechanics import CenterOfMassMeasurement
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.scoring import BenchmarkSpec, ScoreResult
from engine.types.video import FrameTiming
from engine.utils.geometry import is_finite_point
from engine.utils.math_utils import EPSILON

_TRUNK_LANDMARKS: Final[tuple[PoseLandmarkName, ...]] = (
    PoseLandmarkName.LEFT_SHOULDER,
    PoseLandmarkName.RIGHT_SHOULDER,
    PoseLandmarkName.LEFT_HIP,
    PoseLandmarkName.RIGHT_HIP,
)

# (weight, endpoint landmarks). A single-element tuple means "use that
# landmark's position directly" (the head segment has no second endpoint).
_SEGMENTS: Final[tuple[tuple[float, tuple[PoseLandmarkName, ...]], ...]] = (
    (0.081, (PoseLandmarkName.NOSE,)),
    (0.028, (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.LEFT_ELBOW)),
    (0.028, (PoseLandmarkName.RIGHT_SHOULDER, PoseLandmarkName.RIGHT_ELBOW)),
    (0.022, (PoseLandmarkName.LEFT_ELBOW, PoseLandmarkName.LEFT_WRIST)),
    (0.022, (PoseLandmarkName.RIGHT_ELBOW, PoseLandmarkName.RIGHT_WRIST)),
    (0.100, (PoseLandmarkName.LEFT_HIP, PoseLandmarkName.LEFT_KNEE)),
    (0.100, (PoseLandmarkName.RIGHT_HIP, PoseLandmarkName.RIGHT_KNEE)),
    (0.0465, (PoseLandmarkName.LEFT_KNEE, PoseLandmarkName.LEFT_ANKLE)),
    (0.0465, (PoseLandmarkName.RIGHT_KNEE, PoseLandmarkName.RIGHT_ANKLE)),
    (0.0145, (PoseLandmarkName.LEFT_ANKLE, PoseLandmarkName.LEFT_FOOT_INDEX)),
    (0.0145, (PoseLandmarkName.RIGHT_ANKLE, PoseLandmarkName.RIGHT_FOOT_INDEX)),
)
_TRUNK_WEIGHT: Final[float] = 0.497


def _usable(landmark: Landmark | None) -> bool:
    if landmark is None:
        return False
    if landmark.visibility < MIN_LANDMARK_VISIBILITY or landmark.presence < MIN_LANDMARK_PRESENCE:
        return False
    return is_finite_point(landmark.position)


def _segment_center(frame: LandmarkFrame, endpoints: tuple[PoseLandmarkName, ...]) -> Point3D | None:
    landmarks = [frame.pose_landmarks.get(name) for name in endpoints]
    if not all(_usable(landmark) for landmark in landmarks):
        return None
    positions = [landmark.position for landmark in landmarks if landmark is not None]
    if len(positions) == 1:
        return positions[0]
    return midpoint(positions[0], positions[1])


class CenterOfMassCalculator:
    def calculate(self, frame: LandmarkFrame) -> CenterOfMassMeasurement:
        try:
            return self._calculate(frame)
        except Exception:
            return self._invalid(frame.timing)

    def _calculate(self, frame: LandmarkFrame) -> CenterOfMassMeasurement:
        mid_shoulder = _segment_center(frame, (PoseLandmarkName.LEFT_SHOULDER, PoseLandmarkName.RIGHT_SHOULDER))
        mid_hip = _segment_center(frame, (PoseLandmarkName.LEFT_HIP, PoseLandmarkName.RIGHT_HIP))
        if mid_shoulder is None or mid_hip is None:
            return self._invalid(frame.timing)  # trunk not resolvable: don't estimate at all

        weighted_sum = Point3D(x=0.0, y=0.0, z=0.0)
        total_weight = 0.0
        used_landmarks: list[Landmark] = []

        trunk_center = midpoint(mid_shoulder, mid_hip)
        weighted_sum = Point3D(
            x=weighted_sum.x + _TRUNK_WEIGHT * trunk_center.x,
            y=weighted_sum.y + _TRUNK_WEIGHT * trunk_center.y,
            z=weighted_sum.z + _TRUNK_WEIGHT * trunk_center.z,
        )
        total_weight += _TRUNK_WEIGHT
        for name in _TRUNK_LANDMARKS:
            landmark = frame.pose_landmarks[name]
            used_landmarks.append(landmark)

        for weight, endpoints in _SEGMENTS:
            center = _segment_center(frame, endpoints)
            if center is None:
                continue  # limb segment not resolvable: drop and renormalize
            weighted_sum = Point3D(
                x=weighted_sum.x + weight * center.x,
                y=weighted_sum.y + weight * center.y,
                z=weighted_sum.z + weight * center.z,
            )
            total_weight += weight
            for name in endpoints:
                used_landmarks.append(frame.pose_landmarks[name])

        if total_weight <= EPSILON:
            return self._invalid(frame.timing)

        position = Point3D(
            x=weighted_sum.x / total_weight, y=weighted_sum.y / total_weight, z=weighted_sum.z / total_weight
        )
        confidence = min(min(lm.visibility, lm.presence) for lm in used_landmarks)
        height_ratio = self._height_ratio(frame, position)

        return CenterOfMassMeasurement(
            frame_index=frame.timing.frame_index,
            timestamp_ms=frame.timing.timestamp_ms,
            position=position,
            height_ratio=height_ratio,
            confidence=confidence,
            is_valid=True,
        )

    def _height_ratio(self, frame: LandmarkFrame, com: Point3D) -> float | None:
        nose = frame.pose_landmarks.get(PoseLandmarkName.NOSE)
        left_ankle = frame.pose_landmarks.get(PoseLandmarkName.LEFT_ANKLE)
        right_ankle = frame.pose_landmarks.get(PoseLandmarkName.RIGHT_ANKLE)
        if not (_usable(nose) and _usable(left_ankle) and _usable(right_ankle)):
            return None
        assert nose is not None and left_ankle is not None and right_ankle is not None
        ankle_mid_y = (left_ankle.position.y + right_ankle.position.y) / 2.0
        stature = ankle_mid_y - nose.position.y  # positive: feet below head, y grows downward
        if stature <= EPSILON:
            return None
        return (ankle_mid_y - com.y) / stature

    def validate(self, frame: LandmarkFrame) -> bool:
        return self.calculate(frame).is_valid

    def confidence(self, frame: LandmarkFrame) -> float:
        return self.calculate(frame).confidence

    def benchmark(self, measurement: CenterOfMassMeasurement, spec: BenchmarkSpec) -> ScoreResult:
        if not measurement.is_valid or measurement.height_ratio is None:
            return ScoreResult(metric_name=spec.metric_name, raw_value=0.0, band=spec.band, score=0.0)
        raw = measurement.height_ratio
        return ScoreResult(
            metric_name=spec.metric_name, raw_value=raw, band=spec.band, score=score_within_range(raw, spec)
        )

    def _invalid(self, timing: FrameTiming) -> CenterOfMassMeasurement:
        return CenterOfMassMeasurement(
            frame_index=timing.frame_index,
            timestamp_ms=timing.timestamp_ms,
            position=None,
            height_ratio=None,
            confidence=0.0,
            is_valid=False,
        )
