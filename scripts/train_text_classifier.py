"""Train the text classifier based on configuration."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report

if __package__ is None or __package__ == "":
    _SCRIPT_PARENT = Path(__file__).resolve().parents[1]
    _SCRIPT_PARENT_STR = str(_SCRIPT_PARENT)
    if _SCRIPT_PARENT_STR not in sys.path:
        sys.path.insert(0, _SCRIPT_PARENT_STR)

from scripts.bootstrap import bootstrap_project

_PROJECT_ROOT = bootstrap_project()

from src.utils.logger import logger
from src.infrastructure.nlp.model_builder import (
    TextClassifierPipeline,
    build_logistic_pipeline,
    build_neural_components,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def train_model(config: dict) -> TextClassifierPipeline:
    train_path = Path(config["paths"]["processed_train"])
    model_path = Path(config["paths"]["model_artifact"])

    logger.info("Loading training data from {}", train_path)
    df = pd.read_csv(train_path)
    texts = df["clean_text"].astype(str).tolist()
    labels = df["category"].astype(str).tolist()

    model_cfg = config.get("model", {})
    max_features = model_cfg.get("max_features", 5000)
    C = model_cfg.get("C", 1.0)
    n_neurons = model_cfg.get("n_neurons", 0)
    ngram_range = model_cfg.get("ngram_range")
    stopword_exclusions = model_cfg.get("stopword_exclusions")

    if n_neurons and n_neurons > 0:
        logger.info("Training neural network with {} neurons", n_neurons)
        vectorizer, model, label_encoder = build_neural_components(
            max_features,
            n_neurons,
            len(set(labels)),
            ngram_range=ngram_range,
            stopword_exclusions=stopword_exclusions,
        )
        label_encoder.fit(labels)
        y_encoded = label_encoder.transform(labels)
        y_one_hot = np.eye(len(label_encoder.classes_))[y_encoded]
        vectorizer.fit(texts)
        features = vectorizer.transform(texts).toarray()
        model.fit(features, y_one_hot, epochs=5, batch_size=32, verbose=0)
        predictions = label_encoder.inverse_transform(np.argmax(model.predict(features, verbose=0), axis=1))
        artifact_base = model_path.with_suffix("")
        estimator = TextClassifierPipeline((vectorizer, model, label_encoder), use_neural=True)
        estimator.save(artifact_base)
    else:
        logger.info("Training logistic regression classifier")
        pipeline = build_logistic_pipeline(
            max_features,
            C,
            ngram_range=ngram_range,
            stopword_exclusions=stopword_exclusions,
        )
        pipeline.fit(texts, labels)
        predictions = pipeline.predict(texts)
        estimator = TextClassifierPipeline(pipeline, use_neural=False)
        estimator.save(model_path)

    logger.info("Training report:\n{}", classification_report(labels, predictions))
    return estimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train traffic event classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_model(config)


if __name__ == "__main__":
    main()
