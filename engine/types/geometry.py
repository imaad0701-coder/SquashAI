"""Geometric value types shared across the engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BoundingBox:
    top_left: Point2D
    bottom_right: Point2D
