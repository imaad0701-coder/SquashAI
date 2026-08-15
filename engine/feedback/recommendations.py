"""Recommendation-generation contracts (renamed from reccomendations.py)."""

from __future__ import annotations

from typing import Protocol

from engine.types.feedback import FeedbackItem, Recommendation


class RecommendationRule(Protocol):
    def apply(self, feedback: tuple[FeedbackItem, ...]) -> Recommendation | None: ...
