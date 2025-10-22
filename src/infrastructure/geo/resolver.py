"""Location extraction and geocoding utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Pattern

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
from loguru import logger


@dataclass
class GeoResolver:
    """Extract location strings from text and resolve to coordinates."""

    pattern: Pattern[str] = re.compile(r"en\s+([A-Za-zÁÉÍÓÚáéíóúñÑüÜ\s]+)")
    user_agent: str = "movilidad-inteligente-nlp"

    def __post_init__(self) -> None:
        self._geolocator = Nominatim(user_agent=self.user_agent)

    def extract_location(self, text: str) -> tuple[Optional[str], Optional[float], Optional[float]]:
        logger.debug("Extracting location from text: {}", text)
        match = self.pattern.search(text)
        if not match:
            return None, None, None

        location_name = match.group(1).strip()
        logger.debug("Detected location string: {}", location_name)

        try:
            location = self._geolocator.geocode(location_name, language="es")
        except (GeocoderServiceError, ValueError) as error:
            logger.warning("Geocoding failed for {}: {}", location_name, error)
            return location_name, None, None

        if location is None:
            logger.info("No coordinates found for {}", location_name)
            return location_name, None, None

        logger.debug("Resolved {} to ({}, {})", location_name, location.latitude, location.longitude)
        return location_name, location.latitude, location.longitude


__all__ = ["GeoResolver"]
