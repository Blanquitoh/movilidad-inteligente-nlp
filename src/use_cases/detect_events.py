"""Use case for detecting traffic events from tweets."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Protocol, Sequence

import pandas as pd
from src.utils.logger import logger

from src.core.entities import TrafficEvent
from src.infrastructure.events.priority import PriorityAssessment


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


class CategoryResolver(Protocol):
    def refine(self, text: str, predicted_category: str) -> str:
        ...


class PriorityAssessor(Protocol):
    def assess(
        self,
        created_at: Optional[datetime],
        severity: Optional[str],
        reference_time: Optional[datetime] = None,
    ) -> PriorityAssessment:
        ...


class DetectEventsUseCase:
    """Detect traffic events and enrich them with location and severity."""

    def __init__(
        self,
        classifier: TextClassifier,
        geo_resolver: GeoResolver,
        severity_scorer: SeverityScorer,
        category_resolver: CategoryResolver | None = None,
        priority_assessor: PriorityAssessor | None = None,
    ) -> None:
        self._classifier = classifier
        self._geo_resolver = geo_resolver
        self._severity_scorer = severity_scorer
        self._category_resolver = category_resolver
        self._priority_assessor = priority_assessor

    def execute(
        self,
        texts: list[str],
        created_at_values: Sequence[datetime | None] | None = None,
        reference_time: datetime | None = None,
    ) -> list[TrafficEvent]:
        logger.info("Running detection on {} texts", len(texts))
        predictions = list(self._classifier.predict(texts))
        events: list[TrafficEvent] = []
        provided_dates: Sequence[datetime | None]
        if created_at_values is not None:
            if len(created_at_values) != len(texts):
                msg = "Length of created_at_values must match number of texts"
                raise ValueError(msg)
            provided_dates = created_at_values
        else:
            provided_dates = [None] * len(texts)

        for index, (text, predicted) in enumerate(zip(texts, predictions)):
            location, lat, lon = self._geo_resolver.extract_location(text)
            severity = self._severity_scorer.score(text)
            provided_created_at = provided_dates[index]
            created_at = provided_created_at or self._infer_created_at(text)

            if self._category_resolver is not None:
                predicted = self._category_resolver.refine(text, predicted)

            priority_label: str | None = None
            priority_score: float | None = None
            if self._priority_assessor is not None:
                assessment = self._priority_assessor.assess(
                    created_at,
                    severity,
                    reference_time=reference_time,
                )
                priority_label = assessment.label
                priority_score = assessment.score

            event = TrafficEvent(
                text=text,
                created_at=created_at,
                predicted_category=predicted,
                location_name=location,
                latitude=lat,
                longitude=lon,
                severity=severity,
                priority=priority_label,
                priority_score=priority_score,
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
