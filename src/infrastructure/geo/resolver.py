"""Location extraction and geocoding utilities."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Pattern

try:  # geopy is optional during unit tests
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderServiceError
    from geopy.point import Point
except ImportError:  # pragma: no cover - exercised when geopy is absent
    Nominatim = None  # type: ignore[assignment]

    class GeocoderServiceError(Exception):
        """Fallback geocoder error when geopy is unavailable."""

    @dataclass(frozen=True)
    class Point:  # type: ignore[no-redef]
        """Lightweight substitute for geopy.point.Point."""

        latitude: float
        longitude: float
from src.utils.logger import logger


TERMINATOR_PATTERN = re.compile(r"[,;:](?:\s|$)")
_CONNECTORS = (
    "cuando",
    "donde",
    "porque",
    "por",
    "debido a",
    "tras",
    "después de",
    "despues de",
    "luego de",
    "ya que",
    "mientras",
    "mientras que",
    "próximo a",
    "proximo a",
    "cerca de",
    "a la altura",
    "a la altura de",
    "a la altura del",
    "a la altura de la",
    "frente a",
    "frente al",
    "antes de",
    "desvío",
    "desvio",
)
CONNECTOR_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(connector) for connector in _CONNECTORS) + r")\b",
    re.IGNORECASE,
)

_ARTICLES_PATTERN = r"(?:la|el|los|las)"
_LOCATION_HEAD_PATTERN = (
    r"(?:avenida|av\.?|autopista|aut\.?|calle|c\.?|carretera|circunvalaci[oó]n|circ\.?|marginal|"
    r"expreso|bulevar|boulevard|puente|t[úu]nel|elevado|malec[oó]n|viaducto|paso\s+a\s+desnivel|"
    r"kil[oó]metro|km|glorieta|parque|plaza|sector|barrio|urbanizaci[oó]n|urb\.?|residencial|"
    r"peaje|hospital|centro comercial|estaci[oó]n|terminal|kil\.?|km\.)"
)
_LOCATION_BODY_PATTERN = (
    rf"(?:(?:{_ARTICLES_PATTERN})\s+)?{_LOCATION_HEAD_PATTERN}(?=(?:\s|[,;:.]|$))"
    rf"(?:\s+[A-Za-zÁÉÍÓÚáéíóúñÑüÜ0-9°º.,#-]+)*"
)


@dataclass
class GeoResolver:
    """Extract location strings from text and resolve to coordinates."""

    pattern: Pattern[str] = re.compile(
        rf"\b(?:en|sobre|entre|cerca de|frente a|frente al|hacia|rumbo a|a la altura(?: del?| de la)?|"
        rf"a la salida de|a la entrada de|pasando|antes de|despu[eé]s de|"
        rf"por(?:\s+la|\s+el|\s+los|\s+las)?|del|de la|de los|de las)\s+"
        rf"(?P<location>{_LOCATION_BODY_PATTERN})",
        re.IGNORECASE,
    )
    additional_patterns: tuple[Pattern[str], ...] = (
        re.compile(rf"\b(?P<location>{_LOCATION_BODY_PATTERN})", re.IGNORECASE),
        re.compile(
            rf"\bkm\s*(?P<location>\d+\s*(?:{_LOCATION_HEAD_PATTERN})?)",
            re.IGNORECASE,
        ),
    )
    user_agent: str = "movilidad-inteligente-nlp"
    timeout: int = 5
    viewbox: tuple[tuple[float, float], tuple[float, float]] = (
        (19.95, -72.05),  # North-West corner of the Dominican Republic
        (17.35, -68.25),  # South-East corner of the Dominican Republic
    )

    _ARTICLES: tuple[str, ...] = ("la", "el", "los", "las")
    _COMMON_LOCATION_PREFIXES: frozenset[str] = frozenset(
        {
            "avenida",
            "autopista",
            "bulevar",
            "boulevard",
            "carretera",
            "calle",
            "circunvalacion",
            "circunvalación",
            "expreso",
            "elevado",
            "kilometro",
            "kilómetro",
            "km",
            "malecón",
            "malecon",
            "parque",
            "paso",
            "marginal",
            "puente",
            "tunel",
            "túnel",
            "viaducto",
            "plaza",
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
    gazetteer: Mapping[str, str] = field(
        default_factory=lambda: {
            "puente juan bosch": "Puente Juan Bosch",
            "puente juan pablo": "Puente Juan Pablo Duarte",
            "kilometro 9": "Kilómetro 9 Autopista Duarte",
            "km 9": "Kilómetro 9 Autopista Duarte",
            "km 13": "Kilómetro 13 Autopista Duarte",
            "plaza de la bandera": "Plaza de la Bandera",
            "estacion mam": "Estación Mamá Tingó",
        }
    )

    def __post_init__(self) -> None:
        if Nominatim is None:
            msg = "geopy must be installed to use GeoResolver"
            raise RuntimeError(msg)

        self._geolocator = Nominatim(user_agent=self.user_agent, timeout=self.timeout)

    def extract_location(self, text: str) -> tuple[Optional[str], Optional[float], Optional[float]]:
        logger.debug("Extracting location from text: {}", text)
        match = self._match_location(text)
        if not match:
            logger.debug("No regex match found; attempting gazetteer lookup")
            match = self._lookup_gazetteer(text)
            if not match:
                return None, None, None

        location_name = self._normalize_location_name(match)
        logger.debug("Detected location string: {}", location_name)

        queries = self._build_candidate_queries(location_name)
        location = None
        viewbox = self._format_viewbox()

        for candidate in queries:
            try:
                location = self._geolocator.geocode(
                    candidate,
                    language="es",
                    country_codes="do",
                    viewbox=viewbox,
                    bounded=True,
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
        country_code = self._infer_country_code(address)
        if country_code and country_code != "do":
            logger.info("Discarded location {} outside Dominican Republic", location_name)
            return location_name, None, None

        logger.debug("Resolved {} to ({}, {})", location_name, location.latitude, location.longitude)
        return location_name, location.latitude, location.longitude

    def _match_location(self, text: str) -> Optional[str]:
        for pattern in (self.pattern, *self.additional_patterns):
            match = pattern.search(text)
            if not match:
                continue

            location = match.groupdict().get("location") or match.group(1)
            if location:
                return location

        return None

    def _lookup_gazetteer(self, text: str) -> Optional[str]:
        normalized = self._strip_accents(text.lower())
        for needle, location in self.gazetteer.items():
            if needle in normalized:
                return location
        return None

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
            (r"\burb(?:\.|anizaci[oó]n)?(?=\s|[,;]|$)", "Urbanización"),
            (r"\bresid\.(?=\s|[,;]|$)", "Residencial"),
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

    def _format_viewbox(self) -> tuple[Point, Point]:
        """Return a geopy-compatible bounding box ordered south-west to north-east."""
        (lat1, lon1), (lat2, lon2) = self.viewbox

        south = min(lat1, lat2)
        north = max(lat1, lat2)
        west = min(lon1, lon2)
        east = max(lon1, lon2)

        return (Point(latitude=south, longitude=west), Point(latitude=north, longitude=east))

    @staticmethod
    def _infer_country_code(address: Mapping[str, object]) -> Optional[str]:
        """Detect the country code from a geopy address payload."""
        if not isinstance(address, Mapping):
            return None

        raw_code = address.get("country_code")
        if isinstance(raw_code, str) and raw_code.strip():
            return raw_code.strip().lower()

        country = address.get("country")
        if isinstance(country, str):
            normalized = country.casefold()
            if "dominican republic" in normalized or "república dominicana" in normalized:
                return "do"
            normalized_no_accents = GeoResolver._strip_accents(normalized)
            if "republica dominicana" in normalized_no_accents:
                return "do"

        return None

    @staticmethod
    def _strip_accents(value: str) -> str:
        """Remove diacritics to ease string comparisons."""
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(char for char in normalized if not unicodedata.combining(char))


__all__ = ["GeoResolver"]
