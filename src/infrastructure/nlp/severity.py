"""Heuristic severity scoring for traffic incidents."""
from __future__ import annotations

from src.utils.logger import logger


class KeywordSeverityScorer:
    """Assign severity labels based on keyword presence."""

    def __init__(self) -> None:
        self._severe_keywords = {
            "grave",
            "fatal",
            "herid",
            "colapso",
            "incendio",
            "atrapad",
            "fallecid",
            "volcad",
            "choque multiple",
            "derrumbe",
            "explosi",
        }
        self._moderate_keywords = {
            "demora",
            "retras",
            "lento",
            "congestion",
            "congest",
            "tapon",
            "obstaculo",
            "vehiculo varado",
            "averia",
            "bloqueo",
        }

    def score(self, text: str) -> str | None:
        lowered = self._normalize(text)
        logger.debug("Scoring severity for text: {}", text)
        if any(keyword in lowered for keyword in self._severe_keywords):
            return "alta"
        if any(keyword in lowered for keyword in self._moderate_keywords):
            return "media"
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = text.lower()
        replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
        for accented, plain in replacements.items():
            normalized = normalized.replace(accented, plain)
        return normalized


__all__ = ["KeywordSeverityScorer"]
