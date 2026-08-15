"""Shared calculate/validate/confidence/benchmark orchestration for every
single-frame joint-angle metric (knee_angle.py, hip_angle.py, elbow_angle.py,
shoulder_angle.py, ankle_angle.py, trunk_inclination.py, pelvis_rotation.py,
shoulder_rotation.py).

Subclasses supply which landmarks are required for a given side
(`_required_landmarks`) and how to turn resolved positions into a degree
value, or None for unmeasurable/degenerate geometry (`_measure`). Everything
else — landmark presence/confidence/NaN validation, confidence scoring,
benchmark scoring, and the "never raise" guarantee — is shared here.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Final, Iterable

from engine.biomechanics.scoring import score_within_range
from engine.types.biomechanics import AngleMeasurement, JointAngleType, Side
from engine.types.geometry import Point3D
from engine.types.landmarks import Landmark, LandmarkFrame, PoseLandmarkName
from engine.types.scoring import BenchmarkSpec, ScoreResult
from engine.utils.geometry import angle_between, is_finite_point, magnitude, vector_between
from engine.utils.math_utils import EPSILON

MIN_LANDMARK_VISIBILITY: Final[float] = 0.5
MIN_LANDMARK_PRESENCE: Final[float] = 0.5


def midpoint(a: Point3D, b: Point3D) -> Point3D:
    return Point3D(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0, z=(a.z + b.z) / 2.0)


class AngleCalculator(ABC):
    joint_angle_type: JointAngleType

    @abstractmethod
    def _required_landmarks(self, side: Side | None) -> tuple[PoseLandmarkName, ...]: ...

    @abstractmethod
    def _measure(
        self, positions: dict[PoseLandmarkName, Point3D], required: tuple[PoseLandmarkName, ...]
    ) -> float | None: ...

    def calculate(self, frame: LandmarkFrame, side: Side | None = None) -> AngleMeasurement:
        try:
            required = self._required_landmarks(side)
            landmarks = self._resolve_landmarks(frame, required)
            if landmarks is None:
                return self._invalid(side)

            positions = {name: landmark.position for name, landmark in landmarks.items()}
            angle_degrees = self._measure(positions, required)
            if angle_degrees is None or not math.isfinite(angle_degrees):
                return self._invalid(side)

            return AngleMeasurement(
                joint_angle_type=self.joint_angle_type,
                side=side,
                angle_degrees=angle_degrees,
                confidence=self._confidence_from_landmarks(landmarks.values()),
                is_valid=True,
            )
        except Exception:
            # calculate() must never raise, no matter what the input holds.
            return self._invalid(side)

    def validate(self, frame: LandmarkFrame, side: Side | None = None) -> bool:
        try:
            required = self._required_landmarks(side)
            return self._resolve_landmarks(frame, required) is not None
        except Exception:
            return False

    def confidence(self, frame: LandmarkFrame, side: Side | None = None) -> float:
        try:
            required = self._required_landmarks(side)
            landmarks = self._resolve_landmarks(frame, required)
            if landmarks is None:
                return 0.0
            return self._confidence_from_landmarks(landmarks.values())
        except Exception:
            return 0.0

    def benchmark(self, measurement: AngleMeasurement, spec: BenchmarkSpec) -> ScoreResult:
        if not measurement.is_valid or measurement.angle_degrees is None:
            return ScoreResult(metric_name=spec.metric_name, raw_value=0.0, band=spec.band, score=0.0)

        raw = measurement.angle_degrees
        return ScoreResult(
            metric_name=spec.metric_name, raw_value=raw, band=spec.band, score=score_within_range(raw, spec)
        )

    def _invalid(self, side: Side | None) -> AngleMeasurement:
        return AngleMeasurement(
            joint_angle_type=self.joint_angle_type,
            side=side,
            angle_degrees=None,
            confidence=0.0,
            is_valid=False,
        )

    def _resolve_landmarks(
        self, frame: LandmarkFrame, required: tuple[PoseLandmarkName, ...]
    ) -> dict[PoseLandmarkName, Landmark] | None:
        resolved: dict[PoseLandmarkName, Landmark] = {}
        for name in required:
            landmark = frame.pose_landmarks.get(name)
            if landmark is None:
                return None  # missing / occluded joint
            if landmark.visibility < MIN_LANDMARK_VISIBILITY or landmark.presence < MIN_LANDMARK_PRESENCE:
                return None  # low-confidence landmark
            if not is_finite_point(landmark.position):
                return None  # NaN/inf coordinate
            resolved[name] = landmark
        return resolved

    @staticmethod
    def _confidence_from_landmarks(landmarks: Iterable[Landmark]) -> float:
        # Weakest link: the least-trustworthy required landmark caps how
        # much to trust the resulting angle.
        return min(min(landmark.visibility, landmark.presence) for landmark in landmarks)


class ThreePointAngleCalculator(AngleCalculator):
    """Shared vertex-angle math for (near, vertex, far) landmark triplets.

    The angle is measured at the vertex, between the vectors vertex->near
    and vertex->far — e.g. knee angle is measured at the knee, between the
    vectors to the hip and to the ankle.
    """

    @abstractmethod
    def _triplet(self, side: Side | None) -> tuple[PoseLandmarkName, PoseLandmarkName, PoseLandmarkName]:
        """Returns (near, vertex, far)."""

    def _required_landmarks(self, side: Side | None) -> tuple[PoseLandmarkName, ...]:
        return self._triplet(side)

    def _measure(
        self, positions: dict[PoseLandmarkName, Point3D], required: tuple[PoseLandmarkName, ...]
    ) -> float | None:
        near_name, vertex_name, far_name = required
        vertex_to_near = vector_between(positions[vertex_name], positions[near_name])
        vertex_to_far = vector_between(positions[vertex_name], positions[far_name])
        if magnitude(vertex_to_near) <= EPSILON or magnitude(vertex_to_far) <= EPSILON:
            return None  # coincident landmarks / zero-length vector
        return angle_between(vertex_to_near, vertex_to_far)
