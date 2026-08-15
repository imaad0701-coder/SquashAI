"""Feedback-generation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.types.feedback import FeedbackItem
from engine.types.scoring import ScoreResult


class Coach(ABC):
    @abstractmethod
    def generate_feedback(self, scores: tuple[ScoreResult, ...]) -> tuple[FeedbackItem, ...]: ...
