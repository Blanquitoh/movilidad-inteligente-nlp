"""Factories for NLP vectorizers and classifiers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

try:  # Optional dependency for neural network experiments
    from tensorflow import keras
except Exception:  # noqa: BLE001
    keras = None

from src.utils.text_cleaning import clean_text, ensure_stopwords


@dataclass
class ModelConfig:
    max_features: int
    C: float
    n_neurons: int


class TextClassifierPipeline:
    """Wrapper around scikit-learn or Keras models with a consistent API."""

    def __init__(self, estimator, use_neural: bool = False):
        self._estimator = estimator
        self._use_neural = use_neural

    def predict(self, texts: Iterable[str]):
        if self._use_neural:
            vectorizer, model, label_encoder = self._estimator
            features = vectorizer.transform(texts)
            probs = model.predict(features.toarray())
            indices = np.argmax(probs, axis=1)
            return label_encoder.inverse_transform(indices)
        return self._estimator.predict(texts)

    def predict_proba(self, texts: Iterable[str]):
        if self._use_neural:
            vectorizer, model, _ = self._estimator
            features = vectorizer.transform(texts)
            return model.predict(features.toarray())
        return self._estimator.predict_proba(texts)

    def save(self, path: Path) -> None:
        logger.info("Saving model to {}", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._use_neural:
            vectorizer, model, label_encoder = self._estimator
            joblib.dump(vectorizer, path.with_suffix(".vectorizer.joblib"))
            joblib.dump(label_encoder, path.with_suffix(".label_encoder.joblib"))
            model.save(path.with_suffix(".h5"))
        else:
            joblib.dump(self._estimator, path)

    @classmethod
    def load(cls, path: Path):
        if path.suffix == ".joblib":
            estimator = joblib.load(path)
            return cls(estimator, use_neural=False)
        if keras is None:
            raise RuntimeError("TensorFlow is required to load neural models")
        vectorizer = joblib.load(path.with_suffix(".vectorizer.joblib"))
        label_encoder = joblib.load(path.with_suffix(".label_encoder.joblib"))
        model = keras.models.load_model(path.with_suffix(".h5"))
        return cls((vectorizer, model, label_encoder), use_neural=True)


def _clean_batch(texts, stopwords):
    return [clean_text(text, stopwords) for text in texts]


def _build_cleaner():
    stopwords = ensure_stopwords()
    return FunctionTransformer(
        func=partial(_clean_batch, stopwords=stopwords),
        validate=False,
    )


def build_logistic_pipeline(max_features: int, C: float) -> Pipeline:
    logger.info("Building logistic regression pipeline")
    cleaner = _build_cleaner()
    vectorizer = TfidfVectorizer(max_features=max_features)
    classifier = LogisticRegression(C=C, max_iter=1000)
    pipeline = Pipeline(
        steps=[
            ("cleaner", cleaner),
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )
    return pipeline


def build_neural_components(max_features: int, n_neurons: int, num_classes: int):
    if keras is None:
        raise RuntimeError("TensorFlow is required for neural pipeline")

    cleaner = _build_cleaner()
    vectorizer = TfidfVectorizer(max_features=max_features)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(max_features,)),
            keras.layers.Dense(n_neurons, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    label_encoder = LabelEncoder()

    class CleanerVectorizer:
        def __init__(self, cleaner, vectorizer):
            self.cleaner = cleaner
            self.vectorizer = vectorizer

        def fit(self, texts):
            cleaned = self.cleaner.transform(texts)
            self.vectorizer.fit(cleaned)
            return self

        def transform(self, texts):
            cleaned = self.cleaner.transform(texts)
            return self.vectorizer.transform(cleaned)

    pipeline = CleanerVectorizer(cleaner, vectorizer)
    return pipeline, model, label_encoder


__all__ = [
    "ModelConfig",
    "TextClassifierPipeline",
    "build_logistic_pipeline",
    "build_neural_components",
]
