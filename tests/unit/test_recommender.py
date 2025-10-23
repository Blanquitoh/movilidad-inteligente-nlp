"""Unit tests for the content-based recommender."""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from src.core.entities import UserProfile
from src.infrastructure.recommenders.content_based import ContentBasedRecommender


def test_recommender_normalises_catalog_and_scores_interests() -> None:
    catalog = pd.DataFrame(
        {
            "Hashtags": ["#FitnessGoals", "#MovieNight"],
            "Post Text": ["Entrenamiento en el gym", "Estreno de película"],
            "User Description 1": ["Fanático del deporte", "Amante del cine"],
            "User Age": [27, 42],
            "User Gender": ["male", "female"],
        }
    )

    recommender = ContentBasedRecommender(catalog)
    user = UserProfile(user_id="demo", interests=["deporte", "cine"], age=29, gender="male")

    recommendations = recommender.recommend(user, top_k=5)

    assert "deporte" in recommendations
    assert "cine" in recommendations


def test_recommender_prioritises_matching_age_segment() -> None:
    catalog = pd.DataFrame(
        {
            "Post Text": ["Sesión intensa en el gimnasio", "Clásicos del cine"],
            "User Description 1": ["Deporte y salud", "Cine de autor"],
            "User Age": [24, 48],
        }
    )

    recommender = ContentBasedRecommender(catalog)
    young_user = UserProfile(user_id="u1", interests=["deporte", "cine"], age=23)

    recommendations = recommender.recommend(young_user, top_k=5)

    assert "deporte" in recommendations
    assert "cine" not in recommendations
