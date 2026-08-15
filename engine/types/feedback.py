"""Feedback and coaching-recommendation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FeedbackSeverity(Enum):
    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass(frozen=True)
class FeedbackItem:
    message: str
    severity: FeedbackSeverity
    related_metric: str


@dataclass(frozen=True)
class Recommendation:
    title: str
    description: str
    related_feedback: tuple[FeedbackItem, ...]
