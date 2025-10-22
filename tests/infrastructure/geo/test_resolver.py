"""Tests for the geo resolver utilities."""
from __future__ import annotations

from unittest.mock import Mock

from src.infrastructure.geo.resolver import GeoResolver


def build_resolver(monkeypatch):
    """Create a resolver with a mocked geolocator to avoid network access."""

    geocode = Mock()
    geolocator = Mock(geocode=geocode)
    monkeypatch.setattr("src.infrastructure.geo.resolver.Nominatim", lambda user_agent: geolocator)
    resolver = GeoResolver()
    return resolver, geocode


def test_extract_location_trims_trailing_description(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    geocode.return_value = None

    text = (
        "Accidente grave en la Av. 27 de Febrero, dos vehículos colisionaron y el tránsito está detenido."
    )

    location, lat, lon = resolver.extract_location(text)

    assert location == "la Av. 27 de Febrero"
    assert lat is None and lon is None
    geocode.assert_called_once_with("la Av. 27 de Febrero", language="es", country_codes="do")


def test_extract_location_stops_at_connectors(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    geocode.return_value = None

    text = "Tráfico pesado en la avenida Luperón por accidente múltiple"

    location, lat, lon = resolver.extract_location(text)

    assert location == "la avenida Luperón"
    assert lat is None and lon is None
    geocode.assert_called_once_with("la avenida Luperón", language="es", country_codes="do")
