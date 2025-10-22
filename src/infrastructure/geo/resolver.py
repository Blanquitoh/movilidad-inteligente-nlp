"""Location extraction and geocoding utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Pattern

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
from loguru import logger


TERMINATOR_PATTERN = re.compile(r"[,;:](?:\s|$)")
CONNECTOR_PATTERN = re.compile(
    r"\b(?:cuando|donde|porque|por|debido a|tras|después de|luego de|ya que|mientras(?: que)?)\b",
    re.IGNORECASE,
)


@dataclass
class GeoResolver:
    """Extract location strings from text and resolve to coordinates."""

    pattern: Pattern[str] = re.compile(
        r"en\s+([A-Za-zÁÉÍÓÚáéíóúñÑüÜ0-9°º.,-]+(?:\s+[A-Za-zÁÉÍÓÚáéíóúñÑüÜ0-9°º.,-]+)*)"
    )
    user_agent: str = "movilidad-inteligente-nlp"

    _ARTICLES: tuple[str, ...] = ("la", "el", "los", "las")
    _COMMON_LOCATION_PREFIXES: frozenset[str] = frozenset(
        {
            "avenida",
            "autopista",
            "bulevar",
            "carretera",
            "calle",
            "circunvalacion",
            "circunvalación",
            "expreso",
            "marginal",
            "puente",
            "tunel",
            "túnel",
        }
    )
    _FALLBACK_LOCALITIES: tuple[str, ...] = (
        "Santo Domingo",
        "Distrito Nacional",
        "Santiago de los Caballeros",
        "Santo Domingo Este",
        "Santo Domingo Norte",
        "Santo Domingo Oeste",
    )

    def __post_init__(self) -> None:
        self._geolocator = Nominatim(user_agent=self.user_agent)

    def extract_location(self, text: str) -> tuple[Optional[str], Optional[float], Optional[float]]:
        logger.debug("Extracting location from text: {}", text)
        match = self.pattern.search(text)
        if not match:
            return None, None, None

        location_name = self._normalize_location_name(match.group(1))
        logger.debug("Detected location string: {}", location_name)

        queries = self._build_candidate_queries(location_name)
        location = None
        for candidate in queries:
            try:
                location = self._geolocator.geocode(
                    candidate,
                    language="es",
                    country_codes="do",
                )
            except (GeocoderServiceError, ValueError) as error:
                logger.warning("Geocoding failed for {}: {}", candidate, error)
                continue

            if location:
                break

        if location is None:
            logger.info("No coordinates found for {}", location_name)
            return location_name, None, None

        address = getattr(location, "raw", {}).get("address", {})
        if address.get("country_code") != "do":
            logger.info("Discarded location {} outside Dominican Republic", location_name)
            return location_name, None, None

        logger.debug("Resolved {} to ({}, {})", location_name, location.latitude, location.longitude)
        return location_name, location.latitude, location.longitude

    def _normalize_location_name(self, raw_location: str) -> str:
        """Reduce raw regex matches to the most relevant location span."""

        normalized = re.sub(r"\s+", " ", raw_location).strip(" ,.-")
        normalized = TERMINATOR_PATTERN.split(normalized, maxsplit=1)[0].strip(" ,.-")

        connector_match = CONNECTOR_PATTERN.search(normalized)
        if connector_match:
            normalized = normalized[: connector_match.start()].rstrip(" ,.-")

        if not normalized:
            return re.sub(r"\s+", " ", raw_location).strip(" ,.-")

        expanded = self._expand_location_tokens(normalized)
        without_article = self._strip_leading_article(expanded)

        return without_article or expanded

    def _strip_leading_article(self, location: str) -> str:
        lowered = location.lower()
        for article in self._ARTICLES:
            prefix = f"{article} "
            if lowered.startswith(prefix):
                remainder = location[len(prefix) :].lstrip()
                if not remainder:
                    return location

                first_token = re.split(r"\s+", remainder, maxsplit=1)[0]
                normalized_token = re.sub(r"[^\wÁÉÍÓÚáéíóúñÑüÜ]", "", first_token).lower()
                if normalized_token in self._COMMON_LOCATION_PREFIXES:
                    return remainder

        return location

    def _expand_location_tokens(self, location: str) -> str:
        replacements: tuple[tuple[str, str], ...] = (
            (r"\bav(?:\.|da\.?|e\.?)?(?=\s|[,;]|$)", "Avenida"),
            (r"\baut(?:\.|opista)?(?=\s|[,;]|$)", "Autopista"),
            (r"\bc(?:\.|alle)?(?=\s|[,;]|$)", "Calle"),
            (r"\btunel(?=\s|[,;]|$)", "Túnel"),
            (r"\bpuent(?:e)?(?=\s|[,;]|$)", "Puente"),
            (r"\bcirc(?:\.|unv\.?|unvalaci[oó]n)(?=\s|[,;]|$)", "Circunvalación"),
            (r"\bexp(?:\.|reso)?(?=\s|[,;]|$)", "Expreso"),
        )

        expanded = location
        for pattern, replacement in replacements:
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)

        return expanded

    def _build_candidate_queries(self, normalized: str) -> list[str]:
        expanded = self._expand_location_tokens(normalized)
        variants: list[str] = self._dedupe_variants(
            (
                normalized,
                self._strip_leading_article(normalized),
                expanded,
                self._strip_leading_article(expanded),
            )
        )

        queries: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            cleaned = candidate.strip()
            if not cleaned:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            queries.append(cleaned)

        for variant in variants:
            add(variant)

        queries_snapshot: list[str] = list(queries)
        for variant in queries_snapshot:
            add(f"{variant}, República Dominicana")
            for locality in self._FALLBACK_LOCALITIES:
                add(f"{variant}, {locality}, República Dominicana")

        return queries

    @staticmethod
    def _dedupe_variants(variants: Iterable[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            cleaned = variant.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(cleaned)
        return unique


__all__ = ["GeoResolver"]
