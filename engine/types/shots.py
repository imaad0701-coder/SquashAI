"""Shot classification contracts."""

from __future__ import annotations

from enum import Enum


class ShotType(Enum):
    FOREHAND = "forehand"
    BACKHAND = "backhand"


class Handedness(Enum):
    LEFT = "left"
    RIGHT = "right"
