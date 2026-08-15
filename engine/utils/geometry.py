"""Geometry operations: structural contracts, plus their concrete vector math.

The concrete functions below are the shared, reusable implementation of
`DistanceFunc`/`AngleFunc` (and supporting vector primitives) that any
biomechanics calculator should call into rather than reimplementing.
"""

from __future__ import annotations

import math
from typing import Protocol

from engine.types.geometry import Point2D, Point3D, Vector3D
from engine.utils.math_utils import EPSILON


class DistanceFunc(Protocol):
    def __call__(self, a: Point3D, b: Point3D) -> float: ...


class AngleFunc(Protocol):
    def __call__(self, a: Vector3D, b: Vector3D) -> float: ...


class ProjectionFunc(Protocol):
    def __call__(self, point: Point3D) -> Point2D: ...


def is_finite_point(point: Point3D) -> bool:
    """False for any NaN/inf coordinate."""
    return math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)


def vector_between(start: Point3D, end: Point3D) -> Vector3D:
    """The vector pointing from start to end."""
    return Vector3D(x=end.x - start.x, y=end.y - start.y, z=end.z - start.z)


def magnitude(vector: Vector3D) -> float:
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def dot_product(a: Vector3D, b: Vector3D) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross_product(a: Vector3D, b: Vector3D) -> Vector3D:
    return Vector3D(
        x=a.y * b.z - a.z * b.y,
        y=a.z * b.x - a.x * b.z,
        z=a.x * b.y - a.y * b.x,
    )


def distance(a: Point3D, b: Point3D) -> float:
    return magnitude(vector_between(a, b))


def angle_between(a: Vector3D, b: Vector3D) -> float:
    """Unsigned angle in degrees between two vectors, in [0, 180].

    Uses atan2(|a x b|, a . b) rather than acos(dot / (|a||b|)): the atan2
    form stays numerically stable near 0 and 180 degrees, where acos's
    derivative blows up. Returns 0.0 for a zero-length input — callers must
    treat that as an invalid measurement themselves (via their own
    zero-length checks), since a zero vector has no direction to compare.
    """
    if magnitude(a) <= EPSILON or magnitude(b) <= EPSILON:
        return 0.0
    return math.degrees(math.atan2(magnitude(cross_product(a, b)), dot_product(a, b)))


def heading_angle_degrees(vector: Vector3D) -> float:
    """Signed angle in degrees, in (-180, 180], of a vector's projection onto
    the XZ plane, measured from the +X axis via atan2(z, x).

    Unlike angle_between (unsigned, for the angle *between* two vectors),
    this is a directed heading for a single vector — used for rotation-style
    measures (pelvis/shoulder turn) where the sign indicates direction.
    """
    return math.degrees(math.atan2(vector.z, vector.x))
