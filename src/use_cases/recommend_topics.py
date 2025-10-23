"""Use case for recommending topics to users."""
from __future__ import annotations

from typing import Protocol

from src.utils.logger import logger

from src.core.entities import UserProfile


class Recommender(Protocol):
    def recommend(self, user: UserProfile, top_k: int = 5) -> list[str]:
        ...

    def available_topics(self) -> list[str]:
        ...


class RecommendTopicsUseCase:
    """Recommend topics based on user profile information."""

    def __init__(self, recommender: Recommender) -> None:
        self._recommender = recommender

    def execute(self, user: UserProfile, top_k: int = 5) -> list[str]:
        logger.info("Generating recommendations for user {}", user.user_id)
        return self._recommender.recommend(user, top_k=top_k)

    def available_topics(self) -> list[str]:
        return self._recommender.available_topics()


__all__ = ["RecommendTopicsUseCase"]
