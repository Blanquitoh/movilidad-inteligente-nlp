"""Heuristic severity scoring for traffic incidents."""
from __future__ import annotations

from loguru import logger


class KeywordSeverityScorer:
    """Assign severity labels based on keyword presence."""

    def __init__(self) -> None:
        self._severe_keywords = {"grave", "fatal", "heridos", "colapso"}
        self._moderate_keywords = {"demora", "retraso", "lento"}

    def score(self, text: str) -> str | None:
        lowered = text.lower()
        logger.debug("Scoring severity for text: {}", text)
        if any(keyword in lowered for keyword in self._severe_keywords):
            return "alta"
        if any(keyword in lowered for keyword in self._moderate_keywords):
            return "media"
        return None


__all__ = ["KeywordSeverityScorer"]
