"""Low-level drawing primitives contracts (moved from utils/drawing.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engine.types.geometry import Point2D


@dataclass(frozen=True)
class DrawingStyle:
    color_rgb: tuple[int, int, int]
    thickness: int
    opacity: float


class Drawer(Protocol):
    def draw_point(self, point: Point2D, style: DrawingStyle) -> None: ...

    def draw_line(self, start: Point2D, end: Point2D, style: DrawingStyle) -> None: ...
