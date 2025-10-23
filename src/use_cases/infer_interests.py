"""Use case for inferring user interests from raw texts."""
from __future__ import annotations

from typing import Protocol, Sequence

from src.core.entities import DetectedInterest


class InterestModel(Protocol):
    """Protocol describing an interest inference service."""

    def detect_interests(self, texts: Sequence[str], max_keywords: int = 5) -> list[DetectedInterest]:
        ...


class InferInterestsUseCase:
    """Infer the most relevant interests for a set of user generated texts."""

    def __init__(self, model: InterestModel) -> None:
        self._model = model

    def execute(self, texts: Sequence[str], max_keywords: int = 5) -> list[DetectedInterest]:
        return self._model.detect_interests(texts, max_keywords=max_keywords)


__all__ = ["InferInterestsUseCase"]
