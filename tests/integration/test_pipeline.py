"""Integration test for ETL and training pipeline."""
from __future__ import annotations

import pandas as pd

from scripts.etl_build_dataset import build_datasets
from scripts.train_text_classifier import train_model
from src.infrastructure.nlp.model_builder import TextClassifierPipeline


def test_full_pipeline(tmp_path):
    raw_path = tmp_path / "raw.csv"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"

    base_records = [
        ("Accidente en la avenida central", "accidente"),
        ("Protesta en Via Espana", "protesta"),
        ("Lluvia intensa en casco viejo", "lluvia"),
        ("Obstaculo en puente de las americas", "obstaculo"),
    ]
    data = pd.DataFrame(
        {
            "text": [text for text, _ in base_records for _ in range(2)],
            "category": [label for _, label in base_records for _ in range(2)],
            "created_at": ["2022-01-01"] * 8,
            "location": ["Panamá"] * 8,
        }
    )
    data.to_csv(raw_path, index=False)

    config = {
        "paths": {
            "raw_data": str(raw_path),
            "recommendation_data": str(tmp_path / "reco.csv"),
            "sentiment_data": str(tmp_path / "sent.csv"),
            "processed_train": str(train_path),
            "processed_test": str(test_path),
            "model_artifact": str(model_path),
        },
        "split": {"test_size": 0.25, "random_state": 42},
        "model": {"max_features": 100, "C": 1.0, "n_neurons": 0},
        "logging": {"level": "INFO"},
    }

    build_datasets(config)
    assert train_path.exists()
    assert test_path.exists()

    estimator = train_model(config)
    assert isinstance(estimator, TextClassifierPipeline)

    test_df = pd.read_csv(test_path)
    predictions = estimator.predict(test_df["clean_text"].tolist())
    assert len(predictions) == len(test_df)
