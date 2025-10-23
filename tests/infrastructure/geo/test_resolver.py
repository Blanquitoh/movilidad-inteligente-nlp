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


def make_location(lat: float, lon: float, *, address: dict | None = None) -> Mock:
    """Return a mock geopy location with the given coordinates."""
    payload = {"country_code": "do"}
    if address:
        payload.update(address)
    return Mock(latitude=lat, longitude=lon, raw={"address": payload})


def assert_default_geocode_call(geocode: Mock, expected_query: str, resolver: GeoResolver) -> None:
    """Validate the geocode invocation uses the resolver defaults."""
    assert geocode.call_args_list, "geocode was not called"
    first_call = geocode.call_args_list[0]
    assert first_call.args[0] == expected_query

    kwargs = first_call.kwargs
    assert kwargs["language"] == "es"
    assert kwargs["country_codes"] == "do"
    assert kwargs["bounded"] is True
    expected_viewbox = resolver._format_viewbox()
    assert kwargs["viewbox"] == expected_viewbox


def test_extract_location_trims_trailing_description(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    responses = {
        "Avenida 27 de Febrero": make_location(18.472215, -69.931568),
    }
    geocode.side_effect = lambda query, **kwargs: responses.get(query)

    text = (
        "Accidente grave en la Av. 27 de Febrero, dos vehículos colisionaron y el tránsito está detenido."
    )

    location, lat, lon = resolver.extract_location(text)

    assert location == "Avenida 27 de Febrero"
    assert lat == 18.472215
    assert lon == -69.931568

    assert_default_geocode_call(geocode, "Avenida 27 de Febrero", resolver)


def test_extract_location_stops_at_connectors(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    responses = {
        "avenida Luperón": make_location(18.448692, -69.977123),
    }
    geocode.side_effect = lambda query, **kwargs: responses.get(query)

    text = "Tráfico pesado en la avenida Luperón por accidente múltiple"

    location, lat, lon = resolver.extract_location(text)

    assert location == "avenida Luperón"
    assert lat == 18.448692
    assert lon == -69.977123

    assert_default_geocode_call(geocode, "avenida Luperón", resolver)


def test_extract_location_handles_del_pattern(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    responses = {
        "túnel de la Avenida 27 de Febrero con Ortega y Gasset": make_location(
            18.475361, -69.933992
        ),
    }
    geocode.side_effect = lambda query, **kwargs: responses.get(query)

    text = "Colapso del túnel de la Avenida 27 de Febrero con Ortega y Gasset."

    location, lat, lon = resolver.extract_location(text)

    assert location == "túnel de la Avenida 27 de Febrero con Ortega y Gasset"
    assert lat == 18.475361
    assert lon == -69.933992

    assert_default_geocode_call(
        geocode, "túnel de la Avenida 27 de Febrero con Ortega y Gasset", resolver
    )


def test_extract_location_handles_entre_pattern(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    responses = {
        "Calle Ocho y la Avenida España": make_location(18.466273, -69.862914),
    }
    geocode.side_effect = lambda query, **kwargs: responses.get(query)

    text = (
        "Tránsito lento reportado entre la Calle Ocho y la Avenida España, "
        "un camión bloquea parcialmente el carril."
    )

    location, lat, lon = resolver.extract_location(text)

    assert location == "Calle Ocho y la Avenida España"
    assert lat == 18.466273
    assert lon == -69.862914

    assert_default_geocode_call(geocode, "Calle Ocho y la Avenida España", resolver)


def test_extract_location_handles_elevado_without_preposition(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    responses = {
        "elevado de la 27 con Churchill": make_location(18.472951, -69.939215),
    }
    geocode.side_effect = lambda query, **kwargs: responses.get(query)

    text = "Colapsó el elevado de la 27 con Churchill."

    location, lat, lon = resolver.extract_location(text)

    assert location == "elevado de la 27 con Churchill"
    assert lat == 18.472951
    assert lon == -69.939215

    assert_default_geocode_call(geocode, "elevado de la 27 con Churchill", resolver)


def test_extract_location_accepts_uppercase_country_code(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    responses = {
        "Avenida Sabana Larga": make_location(
            18.482, -69.85, address={"country_code": "DO"}
        ),
    }
    geocode.side_effect = lambda query, **kwargs: responses.get(query)

    text = "Choque en Avenida Sabana Larga próximo a la charles."

    location, lat, lon = resolver.extract_location(text)

    assert location == "Avenida Sabana Larga"
    assert lat == 18.482
    assert lon == -69.85
    assert_default_geocode_call(geocode, "Avenida Sabana Larga", resolver)


def test_extract_location_accepts_country_name_without_code(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    responses = {
        "Avenida Sabana Larga": make_location(
            18.482, -69.85, address={"country_code": "", "country": "República Dominicana"}
        ),
    }
    geocode.side_effect = lambda query, **kwargs: responses.get(query)

    text = "Tránsito pesado sobre Avenida Sabana Larga."

    location, lat, lon = resolver.extract_location(text)

    assert location == "Avenida Sabana Larga"
    assert lat == 18.482
    assert lon == -69.85
    assert_default_geocode_call(geocode, "Avenida Sabana Larga", resolver)


def test_extract_location_discards_non_dominican_addresses(monkeypatch):
    resolver, geocode = build_resolver(monkeypatch)
    responses = {
        "Avenida Sabana Larga": make_location(
            25.774, -80.19, address={"country_code": "us", "country": "United States"}
        ),
    }
    geocode.side_effect = lambda query, **kwargs: responses.get(query)

    text = "Accidente en Avenida Sabana Larga."

    location, lat, lon = resolver.extract_location(text)

    assert location == "Avenida Sabana Larga"
    assert lat is None
    assert lon is None
    assert_default_geocode_call(geocode, "Avenida Sabana Larga", resolver)
