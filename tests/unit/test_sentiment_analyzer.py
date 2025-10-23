"""Tests for the Spanish sentiment analyzer."""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from src.infrastructure.nlp.sentiment_analysis import SentimentAnalyzer


def test_sentiment_analyzer_predicts_labels_from_csv(tmp_path) -> None:
    dataset = pd.DataFrame(
        {
            "text": [
                "me siento muy feliz hoy",
                "este dia es una alegria total",
                "estoy realmente molesto por el trafico",
                "que enojo con este tranque infinito",
            ],
            "sentiment": ["joyful", "joyful", "mad", "mad"],
            "emotion": ["thankful", "playful", "irritated", "frustrated"],
        }
    )
    csv_path = tmp_path / "sentiments.csv"
    dataset.to_csv(csv_path, index=False)

    analyzer = SentimentAnalyzer.from_csv(str(csv_path))
    predictions = analyzer.predict(
        [
            "amaneci feliz despues del paseo",
            "estoy molesto con el atasco",
        ]
    )

    assert len(predictions) == 2
    labels = {prediction.sentiment for prediction in predictions}
    assert labels.issubset({"joyful", "mad"})
    for prediction in predictions:
        assert 0.0 <= prediction.probability <= 1.0
