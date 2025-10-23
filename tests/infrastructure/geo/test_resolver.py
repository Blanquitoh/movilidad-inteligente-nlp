"""Tests for the geo resolver utilities."""
from __future__ import annotations

from unittest.mock import Mock

from src.infrastructure.geo.resolver import GeoResolver


def build_resolver(monkeypatch):
    """Create a resolver with a mocked geolocator to avoid network access."""

    geocode = Mock()
    geolocator = Mock(geocode=geocode)
    monkeypatch.setattr(
        "src.infrastructure.geo.resolver.Nominatim",
        lambda user_agent, **kwargs: geolocator,
    )
    resolver = GeoResolver()
    return resolver, geocode


def test_extract_location_trims_trailing_description(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    geocode.return_value = None

    text = (
        "Accidente grave en la Av. 27 de Febrero, dos vehículos colisionaron y el tránsito está detenido."
    )

    location, lat, lon = resolver.extract_location(text)

    assert location == "Avenida 27 de Febrero"
    assert lat is None and lon is None

    first_call = geocode.call_args_list[0]
    assert first_call.args[0] == "Avenida 27 de Febrero"
    assert first_call.kwargs == {
        "language": "es",
        "country_codes": "do",
        "viewbox": resolver.viewbox,
        "bounded": True,
    }


def test_extract_location_stops_at_connectors(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    geocode.return_value = None

    text = "Tráfico pesado en la avenida Luperón por accidente múltiple"

    location, lat, lon = resolver.extract_location(text)

    assert location == "avenida Luperón"
    assert lat is None and lon is None

    first_call = geocode.call_args_list[0]
    assert first_call.args[0] == "avenida Luperón"
    assert first_call.kwargs == {
        "language": "es",
        "country_codes": "do",
        "viewbox": resolver.viewbox,
        "bounded": True,
    }


def test_extract_location_handles_del_pattern(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    geocode.return_value = None

    text = "Colapso del túnel de la Avenida 27 de Febrero con Ortega y Gasset."

    location, lat, lon = resolver.extract_location(text)

    assert location == "túnel de la Avenida 27 de Febrero con Ortega y Gasset"
    assert lat is None and lon is None

    first_call = geocode.call_args_list[0]
    assert first_call.args[0] == "túnel de la Avenida 27 de Febrero con Ortega y Gasset"


def test_extract_location_handles_entre_pattern(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    geocode.return_value = None

    text = (
        "Tránsito lento reportado entre la Calle Ocho y la Avenida España, "
        "un camión bloquea parcialmente el carril."
    )

    location, lat, lon = resolver.extract_location(text)

    assert location == "Calle Ocho y la Avenida España"
    assert lat is None and lon is None

    first_call = geocode.call_args_list[0]
    assert first_call.args[0] == "Calle Ocho y la Avenida España"


def test_extract_location_handles_elevado_without_preposition(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    geocode.return_value = None

    text = "Colapsó el elevado de la 27 con Churchill."

    location, lat, lon = resolver.extract_location(text)

    assert location == "elevado de la 27 con Churchill"
    assert lat is None and lon is None

    first_call = geocode.call_args_list[0]
    assert first_call.args[0] == "elevado de la 27 con Churchill"
