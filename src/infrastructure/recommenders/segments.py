"""Utilities for converting raw ages into configurable demographic segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class AgeSegment:
    """A single inclusive age bucket."""

    label: str
    min_age: Optional[int] = None
    max_age: Optional[int] = None

    def contains(self, age: int) -> bool:
        if self.min_age is not None and age < self.min_age:
            return False
        if self.max_age is not None and age > self.max_age:
            return False
        return True


class AgeSegmenter:
    """Classify ages into configured buckets."""

    def __init__(self, segments: Sequence[AgeSegment]) -> None:
        if not segments:
            raise ValueError("At least one age segment must be configured.")

        self._segments: tuple[AgeSegment, ...] = tuple(self._ordered(segments))

    @staticmethod
    def _ordered(segments: Sequence[AgeSegment]) -> Iterable[AgeSegment]:
        return sorted(
            segments,
            key=lambda segment: (
                -1 if segment.min_age is None else segment.min_age,
                float("inf") if segment.max_age is None else segment.max_age,
                segment.label,
            ),
        )

    @classmethod
    def default(cls) -> "AgeSegmenter":
        return cls(
            segments=(
                AgeSegment(label="under-18", min_age=None, max_age=17),
                AgeSegment(label="18-24", min_age=18, max_age=24),
                AgeSegment(label="25-34", min_age=25, max_age=34),
                AgeSegment(label="35-44", min_age=35, max_age=44),
                AgeSegment(label="45-54", min_age=45, max_age=54),
                AgeSegment(label="55+", min_age=55, max_age=None),
            )
        )

    @classmethod
    def from_config(cls, config: Sequence[dict[str, object]]) -> "AgeSegmenter":
        segments: list[AgeSegment] = []
        for entry in config:
            label = str(entry.get("label")) if entry.get("label") is not None else None
            if not label:
                raise ValueError("Each age segment must define a non-empty 'label'.")

            min_age = entry.get("min")
            max_age = entry.get("max")
            min_age_int = int(min_age) if min_age is not None else None
            max_age_int = int(max_age) if max_age is not None else None

            if min_age_int is not None and max_age_int is not None and min_age_int > max_age_int:
                raise ValueError(
                    f"Invalid age segment '{label}': 'min' ({min_age_int}) cannot be greater than 'max' ({max_age_int})."
                )

            segments.append(AgeSegment(label=label, min_age=min_age_int, max_age=max_age_int))

        return cls(segments)

    def segment(self, age: Optional[int]) -> Optional[str]:
        if age is None or age < 0:
            return None

        for segment in self._segments:
            if segment.contains(age):
                return segment.label
        return None


__all__ = ["AgeSegment", "AgeSegmenter"]
