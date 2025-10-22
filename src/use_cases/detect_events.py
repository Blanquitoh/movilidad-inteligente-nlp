"""Use case for detecting traffic events from tweets."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Protocol

import pandas as pd
from loguru import logger

from src.core.entities import TrafficEvent


class TextClassifier(Protocol):
    def predict(self, texts: Iterable[str]) -> list[str]:
        ...

    def predict_proba(self, texts: Iterable[str]) -> Iterable[Iterable[float]]:
        ...


class GeoResolver(Protocol):
    def extract_location(self, text: str) -> tuple[str | None, float | None, float | None]:
        ...


class SeverityScorer(Protocol):
    def score(self, text: str) -> str | None:
        ...


class DetectEventsUseCase:
    """Detect traffic events and enrich them with location and severity."""

    def __init__(
        self,
        classifier: TextClassifier,
        geo_resolver: GeoResolver,
        severity_scorer: SeverityScorer,
    ) -> None:
        self._classifier = classifier
        self._geo_resolver = geo_resolver
        self._severity_scorer = severity_scorer

    def execute(self, texts: list[str]) -> list[TrafficEvent]:
        logger.info("Running detection on {} texts", len(texts))
        predictions = self._classifier.predict(texts)
        events: list[TrafficEvent] = []

        for text, predicted in zip(texts, predictions):
            location, lat, lon = self._geo_resolver.extract_location(text)
            severity = self._severity_scorer.score(text)
            created_at = self._infer_created_at(text)

            event = TrafficEvent(
                text=text,
                created_at=created_at,
                predicted_category=predicted,
                location_name=location,
                latitude=lat,
                longitude=lon,
                severity=severity,
            )
            logger.debug("Detected event: {}", event)
            events.append(event)

        return events

    @staticmethod
    def _infer_created_at(text: str) -> datetime | None:
        try:
            df = pd.to_datetime([text], errors="coerce")
            return df.iloc[0].to_pydatetime() if pd.notna(df.iloc[0]) else None
        except Exception:  # noqa: BLE001
            return None


__all__ = ["DetectEventsUseCase"]
