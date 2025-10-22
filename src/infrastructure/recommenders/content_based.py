"""Simple content-based recommender using keyword overlap."""
from __future__ import annotations

from collections import Counter

import pandas as pd
from loguru import logger

from src.core.entities import UserProfile


class ContentBasedRecommender:
    """Recommend topics that overlap with the user's interests."""

    def __init__(self, catalog: pd.DataFrame) -> None:
        self.catalog = catalog.copy()

    @classmethod
    def from_csv(cls, path: str) -> "ContentBasedRecommender":
        logger.info("Loading recommendation data from {}", path)
        catalog = pd.read_csv(path)
        return cls(catalog)

    def recommend(self, user: UserProfile, top_k: int = 5) -> list[str]:
        if self.catalog.empty:
            return []

        logger.debug("Recommending topics for user interests: {}", user.interests)
        preference_counts = Counter(interest.lower() for interest in user.interests)

        def score_row(row) -> float:
            tags = {tag.strip().lower() for tag in str(row.get("category", "")).split(",") if tag}
            return sum(preference_counts.get(tag, 0) for tag in tags)

        scored = self.catalog.assign(score=self.catalog.apply(score_row, axis=1))
        top = scored.sort_values("score", ascending=False).head(top_k)
        return top["category"].astype(str).tolist()


__all__ = ["ContentBasedRecommender"]
