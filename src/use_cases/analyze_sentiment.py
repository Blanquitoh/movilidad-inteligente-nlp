"""Use case for running sentiment predictions on user texts."""
from __future__ import annotations

from typing import Protocol, Sequence

from src.core.entities import SentimentPrediction


class SentimentService(Protocol):
    def predict(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        ...


class AnalyzeSentimentUseCase:
    """Thin wrapper that delegates to an underlying sentiment service."""

    def __init__(self, service: SentimentService) -> None:
        self._service = service

    def execute(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        return self._service.predict(texts)


__all__ = ["AnalyzeSentimentUseCase"]
