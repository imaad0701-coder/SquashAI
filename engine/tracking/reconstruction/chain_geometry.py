"""Pure geometry helpers for arm-chain-aware reconstruction, shared by
chain_smoothing.py (Strategy B: joint-chain smoothing) and
angle_space_smoothing.py (Strategy C: angle-space smoothing).

Self-contained vector math -- no engine.biomechanics code is imported,
called, or reimplemented from here (per this iteration's "do not modify
biomechanics calculators" constraint; this also avoids depending on it at
all, even read-only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.tracking.reconstruction.confidence_state import LandmarkState
from engine.types.biomechanics import Side
from engine.types.geometry import Point3D, Vector3D
from engine.types.landmarks import PoseLandmarkName
from engine.utils.geometry import cross_product, dot_product, magnitude, vector_between
from engine.utils.math_utils import EPSILON

SHOULDER_BY_SIDE: dict[Side, PoseLandmarkName] = {
    Side.LEFT: PoseLandmarkName.LEFT_SHOULDER,
    Side.RIGHT: PoseLandmarkName.RIGHT_SHOULDER,
}
ELBOW_BY_SIDE: dict[Side, PoseLandmarkName] = {
    Side.LEFT: PoseLandmarkName.LEFT_ELBOW,
    Side.RIGHT: PoseLandmarkName.RIGHT_ELBOW,
}
WRIST_BY_SIDE: dict[Side, PoseLandmarkName] = {
    Side.LEFT: PoseLandmarkName.LEFT_WRIST,
    Side.RIGHT: PoseLandmarkName.RIGHT_WRIST,
}


def unit_vector(v: Vector3D) -> Vector3D | None:
    m = magnitude(v)
    if m <= EPSILON:
        return None
    return Vector3D(x=v.x / m, y=v.y / m, z=v.z / m)


def scale(v: Vector3D, factor: float) -> Vector3D:
    return Vector3D(x=v.x * factor, y=v.y * factor, z=v.z * factor)


def vector_add(a: Vector3D, b: Vector3D) -> Vector3D:
    return Vector3D(x=a.x + b.x, y=a.y + b.y, z=a.z + b.z)


def point_plus_vector(p: Point3D, v: Vector3D) -> Point3D:
    return Point3D(x=p.x + v.x, y=p.y + v.y, z=p.z + v.z)


def elbow_angle_degrees(shoulder: Point3D, elbow: Point3D, wrist: Point3D) -> float | None:
    """Same vertex-angle definition engine.biomechanics.posture.elbow_angle
    uses (angle at the elbow, between elbow->shoulder and elbow->wrist),
    independently computed here so this module has zero dependency on
    engine.biomechanics."""
    v1 = vector_between(elbow, shoulder)
    v2 = vector_between(elbow, wrist)
    m1, m2 = magnitude(v1), magnitude(v2)
    if m1 <= EPSILON or m2 <= EPSILON:
        return None
    return math.degrees(math.atan2(magnitude(cross_product(v1, v2)), dot_product(v1, v2)))


def rotate_vector_around_axis(v: Vector3D, axis: Vector3D, angle_degrees: float) -> Vector3D:
    """Rodrigues' rotation formula: rotates `v` by `angle_degrees` around
    `axis` (must already be a unit vector)."""
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    term1 = scale(v, cos_t)
    term2 = scale(cross_product(axis, v), sin_t)
    term3 = scale(axis, dot_product(axis, v) * (1 - cos_t))
    return vector_add(vector_add(term1, term2), term3)


@dataclass(frozen=True)
class ChainPlaneReference:
    """A fixed rotation axis (unit vector) capturing which rotational sense
    the forearm bends in relative to the upper arm -- derived once, from a
    reference (most recent FRESH) shoulder/elbow/wrist triple, and held
    constant for an entire predicted run.

    This is an explicit, documented approximation: the arm's swing plane is
    assumed not to twist meaningfully within a single short reconstruction
    gap. Reasonable for the gap lengths this module targets (a handful of
    frames -- see ReconstructionConfig.max_prediction_frames); not a claim
    that real arm motion is planar in general.
    """

    axis: Vector3D


def compute_chain_plane_reference(shoulder: Point3D, elbow: Point3D, wrist: Point3D) -> ChainPlaneReference | None:
    """None when the reference triple is degenerate (shoulder/elbow/wrist
    nearly colinear -- a fully extended arm has no well-defined bend plane)."""
    v1 = vector_between(elbow, shoulder)
    v2 = vector_between(elbow, wrist)
    axis = unit_vector(cross_product(v1, v2))
    return ChainPlaneReference(axis=axis) if axis is not None else None


def reconstruct_forearm_direction(
    shoulder: Point3D, elbow: Point3D, target_angle_degrees: float, plane: ChainPlaneReference
) -> Vector3D | None:
    """The unit elbow->wrist direction that makes
    elbow_angle_degrees(shoulder, elbow, wrist) equal `target_angle_degrees`,
    bending in the same rotational sense `plane` was derived from. None if
    `shoulder`/`elbow` are coincident (no upper-arm direction to rotate)."""
    v1 = unit_vector(vector_between(elbow, shoulder))
    if v1 is None:
        return None
    return rotate_vector_around_axis(v1, plane.axis, target_angle_degrees)


def find_contiguous_runs(states: list[LandmarkState], target: LandmarkState) -> list[tuple[int, int]]:
    """Maximal contiguous [start, end) index ranges where `states[i] ==
    target`."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(states)
    while i < n:
        if states[i] != target:
            i += 1
            continue
        start = i
        while i < n and states[i] == target:
            i += 1
        runs.append((start, i))
    return runs


def robust_reference_length(lengths: list[float], max_samples: int = 5) -> float | None:
    """Median of up to the last `max_samples` values -- a stable "expected
    limb length" reference resistant to any single noisy sample, same
    intent as (independently computed from) the running reference
    landmark_reconstructor.py's own plausibility guard keeps."""
    if not lengths:
        return None
    recent = lengths[-max_samples:]
    ordered = sorted(recent)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
