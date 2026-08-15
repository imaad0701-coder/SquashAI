"""Camera calibration configuration and provider contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engine.types.geometry import Point2D


@dataclass(frozen=True)
class CameraIntrinsics:
    focal_length_px: tuple[float, float]
    principal_point_px: Point2D
    distortion_coefficients: tuple[float, ...]


@dataclass(frozen=True)
class CalibrationConfig:
    reference_points_world: tuple[Point2D, ...]
    reference_points_image: tuple[Point2D, ...]
    court_length_m: float
    court_width_m: float


class CalibrationProvider(Protocol):
    def load(self, config: CalibrationConfig) -> CameraIntrinsics: ...

    def pixel_to_world(self, point: Point2D, intrinsics: CameraIntrinsics) -> Point2D: ...
