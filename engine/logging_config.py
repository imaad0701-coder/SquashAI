"""Typed logging configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class LoggingConfig:
    level: LogLevel
    log_dir: str
    json_format: bool
    max_bytes: int
    backup_count: int
