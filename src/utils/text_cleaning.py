"""Utility functions for text preprocessing."""
from __future__ import annotations

import re
from typing import Iterable

import nltk
from loguru import logger

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_MENTION_PATTERN = re.compile(r"@[\w_]+")
_HASHTAG_PATTERN = re.compile(r"#[\w_]+")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\sáéíóúñüÁÉÍÓÚÑÜ]")
_MULTISPACE_PATTERN = re.compile(r"\s+")


def ensure_stopwords(language: str = "spanish") -> set[str]:
    """Ensure stopwords are available and return the set."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        logger.info("Downloading NLTK stopwords corpus…")
        try:
            nltk.download("stopwords", quiet=True)
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to download stopwords: {}", error)
            return {"de", "la", "en", "el"}
    from nltk.corpus import stopwords

    return set(stopwords.words(language))


def clean_text(text: str, stopwords: Iterable[str] | None = None) -> str:
    """Normalize text by removing urls, mentions, hashtags, and punctuation."""
    logger.debug("Cleaning text: {}", text)
    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _MENTION_PATTERN.sub(" ", text)
    text = _HASHTAG_PATTERN.sub(" ", text)
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    text = _MULTISPACE_PATTERN.sub(" ", text).strip()

    if stopwords:
        tokens = [token for token in text.split() if token not in stopwords]
        text = " ".join(tokens)

    logger.debug("Cleaned text: {}", text)
    return text


__all__ = ["clean_text", "ensure_stopwords"]
