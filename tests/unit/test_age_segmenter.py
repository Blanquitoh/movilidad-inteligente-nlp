"""Unit tests for the age segmenter configuration."""
from __future__ import annotations

import pytest

from src.infrastructure.recommenders.segments import AgeSegmenter


def test_age_segmenter_from_config_handles_open_bounds() -> None:
    segmenter = AgeSegmenter.from_config(
        [
            {"label": "joven", "max": 17},
            {"label": "adulto", "min": 18},
        ]
    )

    assert segmenter.segment(12) == "joven"
    assert segmenter.segment(25) == "adulto"
    assert segmenter.segment(None) is None


def test_age_segmenter_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        AgeSegmenter.from_config(
            [
                {"label": "invalido", "min": 40, "max": 30},
            ]
        )
