"""Integration tests for the content-based recommender."""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from src.core.entities import UserProfile
from src.infrastructure.recommenders.content_based import ContentBasedRecommender
from src.infrastructure.recommenders.segments import AgeSegmenter


def test_recommender_from_csv_respects_age_segments(tmp_path) -> None:
    catalog = pd.DataFrame(
        {
            "Post Text": [
                "Plan de entrenamiento y deporte en equipo",
                "Cartelera de cine independiente para este fin de semana",
            ],
            "User Age": [28, 62],
        }
    )
    csv_path = tmp_path / "catalog.csv"
    catalog.to_csv(csv_path, index=False)

    segmenter = AgeSegmenter.from_config(
        [
            {"label": "adulto", "min": 18, "max": 59},
            {"label": "senior", "min": 60},
        ]
    )

    recommender = ContentBasedRecommender.from_csv(str(csv_path), age_segmenter=segmenter)

    segments = set(recommender.catalog["age_segment"].dropna())
    assert segments == {"adulto", "senior"}

    adult_user = UserProfile(user_id="adulto", interests=["deporte", "cine"], age=30)
    senior_user = UserProfile(user_id="senior", interests=["deporte", "cine"], age=65)

    adult_recommendations = recommender.recommend(adult_user, top_k=5)
    senior_recommendations = recommender.recommend(senior_user, top_k=5)

    assert adult_recommendations == ["deporte"]
    assert senior_recommendations == ["cine"]
