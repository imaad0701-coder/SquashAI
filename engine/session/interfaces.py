"""Player and session contracts for longitudinal tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from engine.types.shots import Handedness


@dataclass(frozen=True)
class Player:
    player_id: str
    display_name: str
    handedness: Handedness


@dataclass(frozen=True)
class SessionConfig:
    max_clips_per_session: int
    min_clip_duration_seconds: float


@dataclass(frozen=True)
class Session:
    session_id: str
    player: Player
    created_at: datetime
    pipeline_result_ids: tuple[str, ...] = field(default_factory=tuple)
