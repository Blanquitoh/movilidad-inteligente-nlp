"""Simple content-based recommender using keyword overlap."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Optional, Sequence

import pandas as pd

from src.core.entities import DetectedInterest, UserProfile
from src.infrastructure.recommenders.segments import AgeSegmenter
from src.utils.logger import logger


def _coerce_age(value: Any) -> Optional[int]:
    """Convert raw age values into integers when possible."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        if value != value:  # NaN check
            return None
        return int(value)

    text = str(value).strip()
    if not text or not any(char.isdigit() for char in text):
        return None

    try:
        return int(float(text))
    except ValueError:
        return None


@dataclass(frozen=True)
class _InterestLexicon:
    """Lexical rules that map free text into high level interests."""

    vocabulary: dict[str, set[str]]

    @classmethod
    def default(cls) -> "_InterestLexicon":
        return cls(
            vocabulary={
                "deporte": {"deporte", "sport", "sports", "fútbol", "football", "fitness", "gym", "running", "cycling"},
                "salud": {"salud", "wellness", "health", "bienestar", "yoga", "mindfulness", "meditation"},
                "cine": {"cine", "movie", "film", "cinema", "película", "hollywood"},
                "dieta": {"dieta", "diet", "nutrition", "nutricion", "food", "receta", "recipe", "vegan"},
                "hobby": {"hobby", "travel", "viaje", "photography", "photo", "pets", "reading", "libro", "adventure"},
            }
        )

    def detect(self, text: str) -> set[str]:
        lowered = text.lower()
        tokens = set(re.findall(r"\b\w+\b", lowered, flags=re.UNICODE))
        matches: set[str] = set()
        for category, keywords in self.vocabulary.items():
            for keyword in keywords:
                if " " in keyword:
                    if re.search(rf"\b{re.escape(keyword)}\b", lowered):
                        matches.add(category)
                        break
                elif keyword in tokens:
                    matches.add(category)
                    break
        return matches


@dataclass
class KeywordInterestDetector:
    """Detect high-level interests based on keyword occurrences."""

    lexicon: _InterestLexicon

    def detect(self, texts: Sequence[str], max_keywords: int = 5) -> list[DetectedInterest]:
        aggregated = Counter()
        keyword_counts: dict[str, Counter[str]] = {}

        for raw_text in texts:
            if not raw_text:
                continue

            lowered = raw_text.lower()
            tokens = set(re.findall(r"\b\w+\b", lowered, flags=re.UNICODE))
            matched_keywords: dict[str, set[str]] = {}

            for category, keywords in self.lexicon.vocabulary.items():
                for keyword in keywords:
                    if " " in keyword:
                        pattern = rf"\b{re.escape(keyword)}\b"
                        if re.search(pattern, lowered):
                            matched_keywords.setdefault(category, set()).add(keyword)
                    elif keyword in tokens:
                        matched_keywords.setdefault(category, set()).add(keyword)

            for category, keywords in matched_keywords.items():
                aggregated[category] += 1
                counter = keyword_counts.setdefault(category, Counter())
                for keyword in keywords:
                    counter[keyword] += 1

        detections: list[DetectedInterest] = []
        for category, count in aggregated.most_common():
            keywords_counter = keyword_counts.get(category, Counter())
            top_keywords = tuple(
                keyword for keyword, _ in keywords_counter.most_common(max(0, int(max_keywords)))
            )
            detections.append(DetectedInterest(name=category, score=count, keywords=top_keywords))

        return detections


class ContentBasedRecommender:
    """Recommend topics that overlap with the user's interests."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        lexicon: _InterestLexicon | None = None,
        age_segmenter: AgeSegmenter | None = None,
    ) -> None:
        self._lexicon = lexicon or _InterestLexicon.default()
        self._age_segmenter = age_segmenter or AgeSegmenter.default()
        self._interest_detector = KeywordInterestDetector(self._lexicon)
        self.catalog = self._normalise_catalog(
            catalog,
            self._lexicon,
            age_segmenter=self._age_segmenter,
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        lexicon: _InterestLexicon | None = None,
        age_segmenter: AgeSegmenter | None = None,
    ) -> "ContentBasedRecommender":
        logger.info("Loading recommendation data from {}", path)
        catalog = pd.read_csv(path)
        return cls(catalog, lexicon=lexicon, age_segmenter=age_segmenter)

    @staticmethod
    def _normalise_catalog(
        catalog: pd.DataFrame,
        lexicon: _InterestLexicon,
        age_segmenter: AgeSegmenter | None = None,
    ) -> pd.DataFrame:
        if catalog.empty:
            return pd.DataFrame(columns=["category", "source_text", "age_segment", "gender"])

        segmenter = age_segmenter or AgeSegmenter.default()

        text_columns = [
            column
            for column in (
                "category",
                "categories",
                "Hashtags",
                "Post Text",
                "User Description 1",
                "User Description 2",
                "User Bio",
            )
            if column in catalog.columns
        ]

        records: list[dict[str, Optional[str]]] = []
        for _, row in catalog.iterrows():
            values = [
                str(row.get(column, ""))
                for column in text_columns
                if pd.notna(row.get(column, "")) and str(row.get(column, "")).strip()
            ]
            text = " ".join(values)
            detected = lexicon.detect(text)
            age_value = row.get("User Age") or row.get("user_age")
            age_segment = segmenter.segment(_coerce_age(age_value))
            gender_value = row.get("User Gender") or row.get("user_gender")
            gender = str(gender_value).strip().lower() if pd.notna(gender_value) and str(gender_value).strip() else None
            for category in detected:
                records.append(
                    {
                        "category": category,
                        "source_text": text,
                        "age_segment": age_segment,
                        "gender": gender,
                    }
                )

        if not records:
            logger.warning("No interest categories detected in recommendation dataset")
            return pd.DataFrame(columns=["category", "source_text", "age_segment", "gender"])

        normalised = pd.DataFrame(records).drop_duplicates()
        logger.debug("Prepared {} recommendation records", len(normalised))
        return normalised

    def recommend(self, user: UserProfile, top_k: int = 5) -> list[str]:
        if self.catalog.empty:
            return []

        logger.debug(
            "Recommending topics for user interests: {} with age {}",
            user.interests,
            user.age,
        )

        working_catalog = self.catalog
        user_segment = self._age_segmenter.segment(user.age)
        if user_segment:
            segmented = working_catalog[working_catalog["age_segment"] == user_segment]
            if not segmented.empty:
                working_catalog = segmented

        preference_counts = Counter(interest.lower() for interest in user.interests)

        def score_row(row) -> float:
            category = str(row.get("category", "")).strip().lower()
            return preference_counts.get(category, 0)

        scored = working_catalog.assign(score=working_catalog.apply(score_row, axis=1))
        scored = scored[scored["score"] > 0]
        if scored.empty:
            return sorted({category for category in working_catalog["category"]})[:top_k]

        top = scored.sort_values(["score", "category"], ascending=[False, True]).drop_duplicates(
            subset=["category"]
        )
        return top.head(top_k)["category"].astype(str).tolist()

    def available_topics(self) -> list[str]:
        if self.catalog.empty:
            return []

        categories = (
            self.catalog["category"].dropna().astype(str).str.strip()
        )
        return sorted({category for category in categories if category})

    def detect_interests(self, texts: Sequence[str], max_keywords: int = 5) -> list[DetectedInterest]:
        return self._interest_detector.detect(texts, max_keywords=max_keywords)


__all__ = ["ContentBasedRecommender", "KeywordInterestDetector"]
