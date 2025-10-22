"""Unit tests for the content-based recommender."""
from __future__ import annotations

import pandas as pd

from src.core.entities import UserProfile
from src.infrastructure.recommenders.content_based import ContentBasedRecommender


def test_recommender_normalises_catalog_and_scores_interests() -> None:
    catalog = pd.DataFrame(
        {
            "Hashtags": ["#FitnessGoals", "#MovieNight"],
            "Post Text": ["Entrenamiento en el gym", "Estreno de película"],
            "User Description 1": ["Fanático del deporte", "Amante del cine"],
        }
    )

    recommender = ContentBasedRecommender(catalog)
    user = UserProfile(user_id="demo", interests=["deporte", "cine"])

    recommendations = recommender.recommend(user, top_k=5)

    assert "deporte" in recommendations
    assert "cine" in recommendations
