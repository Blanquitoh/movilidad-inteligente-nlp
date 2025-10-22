"""Location extraction and geocoding utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Pattern

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

    def __post_init__(self) -> None:
        self._geolocator = Nominatim(user_agent=self.user_agent)

    def extract_location(self, text: str) -> tuple[Optional[str], Optional[float], Optional[float]]:
        logger.debug("Extracting location from text: {}", text)
        match = self.pattern.search(text)
        if not match:
            return None, None, None

        location_name = self._normalize_location_name(match.group(1))
        logger.debug("Detected location string: {}", location_name)

        try:
            location = self._geolocator.geocode(
                location_name, language="es", country_codes="do"
            )
        except (GeocoderServiceError, ValueError) as error:
            logger.warning("Geocoding failed for {}: {}", location_name, error)
            return location_name, None, None

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

        return normalized


__all__ = ["GeoResolver"]
