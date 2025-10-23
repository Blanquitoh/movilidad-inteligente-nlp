"""Utility functions for text preprocessing."""
from __future__ import annotations

import re
from typing import Iterable

from src.utils.logger import logger

try:  # NLTK is optional at runtime
    import nltk
except ImportError:  # pragma: no cover - exercised when nltk is absent
    nltk = None  # type: ignore[assignment]

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_MENTION_PATTERN = re.compile(r"@[\w_]+")
_HASHTAG_PATTERN = re.compile(r"#[\w_]+")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\sáéíóúñüÁÉÍÓÚÑÜ]")
_MULTISPACE_PATTERN = re.compile(r"\s+")


def ensure_stopwords(language: str = "spanish") -> set[str]:
    """Ensure stopwords are available and return the set."""
    fallback = {
        "a",
        "acaso",
        "aqui",
        "alli",
        "allí",
        "ante",
        "antes",
        "como",
        "con",
        "contra",
        "cuando",
        "cual",
        "cuales",
        "de",
        "del",
        "desde",
        "donde",
        "dónde",
        "el",
        "ella",
        "ellas",
        "ellos",
        "en",
        "entre",
        "era",
        "eres",
        "es",
        "esa",
        "ese",
        "eso",
        "esta",
        "este",
        "estos",
        "estas",
        "estoy",
        "fin",
        "fue",
        "ha",
        "han",
        "hasta",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "mas",
        "más",
        "me",
        "mi",
        "mis",
        "mucho",
        "muy",
        "ni",
        "no",
        "nos",
        "nosotros",
        "nuestra",
        "nuestro",
        "otra",
        "otro",
        "para",
        "pero",
        "poco",
        "por",
        "porque",
        "que",
        "qué",
        "quien",
        "quienes",
        "quién",
        "se",
        "ser",
        "si",
        "sí",
        "sin",
        "sobre",
        "soy",
        "su",
        "sus",
        "te",
        "ti",
        "tiene",
        "tienen",
        "todo",
        "toda",
        "todos",
        "todas",
        "tu",
        "tus",
        "un",
        "una",
        "uno",
        "unos",
        "vez",
        "ya",
        "yo",
    }

    if nltk is None:
        logger.warning(
            "NLTK is not installed; using fallback stopwords for language '%s'", language
        )
        return fallback

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        logger.info("Downloading NLTK stopwords corpus…")
        try:
            nltk.download("stopwords", quiet=True)
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to download stopwords: {}", error)
            return fallback

    from nltk.corpus import stopwords

    try:
        return set(stopwords.words(language))
    except LookupError as error:
        logger.warning("Stopwords corpus unavailable: {}", error)
        return fallback


def clean_text(text: str, stopwords: Iterable[str] | None = None) -> str:
    """Normalize text by removing urls, mentions, hashtags, and punctuation."""
    logger.debug("Cleaning text: {}", text)
    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _MENTION_PATTERN.sub(" ", text)
    text = _HASHTAG_PATTERN.sub(" ", text)
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    text = _MULTISPACE_PATTERN.sub(" ", text).strip()

    if stopwords is not None:
        stopword_set = set(stopwords)
        if stopword_set:
            tokens = [token for token in text.split() if token not in stopword_set]
            text = " ".join(tokens)

    logger.debug("Cleaned text: {}", text)
    return text


__all__ = ["clean_text", "ensure_stopwords"]
