"""Keyword-based category refinements for traffic classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.utils.logger import logger


def _normalize(text: str) -> str:
    normalized = text.lower()
    replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    for accented, plain in replacements.items():
        normalized = normalized.replace(accented, plain)
    return normalized


@dataclass(frozen=True)
class KeywordCategoryResolver:
    """Override model predictions when strong keyword evidence exists."""

    keyword_map: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized_map = {_normalize(key): value for key, value in self.keyword_map.items()}
        object.__setattr__(self, "_normalized_map", normalized_map)

    def refine(self, text: str, predicted_category: str) -> str:
        logger.debug("Refining category '{}' for text: {}", predicted_category, text)
        normalized_text = _normalize(text)
        for keyword, category in self._normalized_map.items():
            if keyword in normalized_text:
                logger.debug(
                    "Keyword '{}' matched text. Overriding '{}' with '{}'",
                    keyword,
                    predicted_category,
                    category,
                )
                return category
        return predicted_category

    @classmethod
    def for_obstacles(cls) -> "KeywordCategoryResolver":
        keywords: dict[str, str] = {
            "escombro": "obstaculo",
            "escombros": "obstaculo",
            "poste caido": "obstaculo",
            "poste caído": "obstaculo",
            "arbol caido": "obstaculo",
            "arbol caído": "obstaculo",
            "vehiculo varado": "obstaculo",
            "vehiculo averiado": "obstaculo",
            "camion varado": "obstaculo",
            "derrumb": "obstaculo",
            "bloqueo": "obstaculo",
            "obstaculo": "obstaculo",
            "obstaculo en via": "obstaculo",
            "material en la via": "obstaculo",
        }
        return cls(keyword_map=keywords)


__all__ = ["KeywordCategoryResolver"]
