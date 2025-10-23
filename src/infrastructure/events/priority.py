"""Priority assessment utilities for traffic events."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional


@dataclass(frozen=True)
class PriorityAssessment:
    """Result of computing a priority label and score."""

    label: str
    score: float


class TimeSeverityPriorityAssessor:
    """Combine severity and recency to derive a priority suggestion."""

    def __init__(
        self,
        high_threshold: float = 3.0,
        medium_threshold: float = 1.5,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._high_threshold = high_threshold
        self._medium_threshold = medium_threshold
        self._now_provider = now_provider or datetime.utcnow
        self._severity_weights = {"alta": 2.0, "media": 1.0}
        self._recency_windows: tuple[tuple[timedelta, float], ...] = (
            (timedelta(minutes=30), 2.0),
            (timedelta(hours=2), 1.0),
        )

    def assess(
        self,
        created_at: Optional[datetime],
        severity: Optional[str],
        reference_time: Optional[datetime] = None,
    ) -> PriorityAssessment:
        """Return a priority suggestion using the configured thresholds."""

        severity_score = self._severity_weights.get(severity or "", 0.0)
        recency_score = self._compute_recency_score(created_at, reference_time)
        total_score = severity_score + recency_score

        if total_score >= self._high_threshold:
            label = "alta"
        elif total_score >= self._medium_threshold:
            label = "media"
        else:
            label = "baja"

        return PriorityAssessment(label=label, score=total_score)

    def _compute_recency_score(
        self, created_at: Optional[datetime], reference_time: Optional[datetime]
    ) -> float:
        if created_at is None:
            return 0.0

        now = reference_time or self._now_provider()
        if created_at > now:
            created_at = now

        age = now - created_at
        for window, score in self._recency_windows:
            if age <= window:
                return score
        return 0.0


__all__ = ["PriorityAssessment", "TimeSeverityPriorityAssessor"]
