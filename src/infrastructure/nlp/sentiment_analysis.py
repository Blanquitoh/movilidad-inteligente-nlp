"""Sentiment analysis utilities trained from Spanish tweet data."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.core.entities import SentimentPrediction
from src.infrastructure.nlp.model_builder import build_logistic_pipeline
from src.utils.logger import logger


@dataclass
class _SentimentModel:
    pipeline: Pipeline
    labels: np.ndarray
    sentiment_emotions: dict[str, str]


class SentimentAnalyzer:
    """Train and run sentiment predictions using the project CSV dataset."""

    def __init__(
        self,
        model: _SentimentModel | None = None,
        *,
        label_aliases: Mapping[str, str] | None = None,
        emotion_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self._model = model
        self._label_aliases = {key.lower(): value for key, value in (label_aliases or {}).items()}
        self._emotion_aliases = {key.lower(): value for key, value in (emotion_aliases or {}).items()}

    @classmethod
    def empty(cls) -> "SentimentAnalyzer":
        """Return an analyzer that always yields empty predictions."""

        return cls(model=None)

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        max_features: int = 5000,
        C: float = 1.0,
        ngram_range: Sequence[int] | None = None,
        stopword_exclusions: Sequence[str] | None = None,
        label_aliases: Mapping[str, str] | None = None,
        emotion_aliases: Mapping[str, str] | None = None,
    ) -> "SentimentAnalyzer":
        csv_path = Path(path)
        logger.info("Loading sentiment dataset from {}", csv_path)
        dataframe = pd.read_csv(csv_path)
        dataframe = dataframe.dropna(subset=["text", "sentiment"])

        if dataframe.empty:
            logger.warning("Sentiment dataset is empty; returning no-op analyzer.")
            return cls.empty()

        texts = dataframe["text"].astype(str).tolist()
        sentiments = dataframe["sentiment"].astype(str).tolist()

        pipeline = build_logistic_pipeline(
            max_features=max_features,
            C=C,
            ngram_range=ngram_range,
            stopword_exclusions=stopword_exclusions,
        )
        pipeline.fit(texts, sentiments)

        classifier = pipeline.named_steps["classifier"]
        labels = np.array(classifier.classes_)

        sentiment_emotions: dict[str, str] = {}
        emotion_counter = defaultdict(Counter)
        for sentiment, emotion in zip(dataframe["sentiment"], dataframe.get("emotion", [])):
            if not isinstance(sentiment, str) or not isinstance(emotion, str):
                continue
            cleaned_sentiment = sentiment.strip().lower()
            cleaned_emotion = emotion.strip().lower()
            if cleaned_sentiment and cleaned_emotion:
                emotion_counter[cleaned_sentiment][cleaned_emotion] += 1

        for sentiment, counts in emotion_counter.items():
            if counts:
                sentiment_emotions[sentiment] = counts.most_common(1)[0][0]

        model = _SentimentModel(
            pipeline=pipeline,
            labels=labels,
            sentiment_emotions=sentiment_emotions,
        )
        return cls(
            model=model,
            label_aliases=label_aliases,
            emotion_aliases=emotion_aliases,
        )

    def predict(self, texts: Sequence[str]) -> list[SentimentPrediction]:
        if self._model is None:
            logger.debug("Sentiment analyzer not initialised; returning empty predictions.")
            return []

        cleaned_texts = [text if isinstance(text, str) else "" for text in texts]
        if not cleaned_texts:
            return []

        probabilities = self._model.pipeline.predict_proba(cleaned_texts)
        predictions: list[SentimentPrediction] = []

        for raw_text, proba in zip(cleaned_texts, probabilities):
            if proba.size == 0:
                continue
            top_index = int(np.argmax(proba))
            sentiment = str(self._model.labels[top_index])
            probability = float(proba[top_index])
            emotion = self._model.sentiment_emotions.get(sentiment.lower())
            sentiment_display = self._label_aliases.get(sentiment.lower(), sentiment)
            emotion_display = None
            if emotion:
                emotion_display = self._emotion_aliases.get(emotion.lower(), emotion)
            predictions.append(
                SentimentPrediction(
                    text=raw_text,
                    sentiment=sentiment_display,
                    probability=probability,
                    emotion=emotion_display,
                )
            )

        return predictions


__all__ = ["SentimentAnalyzer"]
