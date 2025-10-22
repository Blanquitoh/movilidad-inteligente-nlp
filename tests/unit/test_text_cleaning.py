"""Unit tests for text cleaning utilities."""
from __future__ import annotations

from src.utils.text_cleaning import clean_text


def test_clean_text_removes_urls_and_mentions():
    text = "Accidente en Vía España! https://example.com @transito"
    cleaned = clean_text(text, stopwords={"en"})
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "en" not in cleaned


def test_clean_text_handles_hashtags_and_punctuation():
    text = "#Urgente accidente!!! en la avenida Central."
    cleaned = clean_text(text)
    assert "#" not in cleaned
    assert "!" not in cleaned
    assert "accidente" in cleaned
