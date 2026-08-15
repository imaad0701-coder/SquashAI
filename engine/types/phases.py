"""Swing phase contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PhaseLabel(Enum):
    READY = "ready"
    BACKSWING = "backswing"
    FORWARD_SWING = "forward_swing"
    CONTACT = "contact"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class PhaseSegment:
    label: PhaseLabel
    start_frame_index: int
    end_frame_index: int
