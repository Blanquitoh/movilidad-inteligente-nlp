from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..location import Location


class Nominatim:
    def __init__(self, *, user_agent: str, **kwargs: Any) -> None: ...

    def geocode(
        self,
        query: str,
        *,
        exactly_one: bool = ...,
        timeout: float | None = ...,
        limit: int | None = ...,
        addressdetails: bool = ...,
        language: str | None = ...,
        geometry: str | None = ...,
        extratags: bool = ...,
        country_codes: str | Sequence[str] | None = ...,
        viewbox: Sequence[float] | None = ...,
        bounded: bool = ...,
        featuretype: str | Sequence[str] | None = ...,
        namedetails: bool = ...,
        **kwargs: Any,
    ) -> Location | None: ...

    def reverse(self, query: Mapping[str, float], **kwargs: Any) -> Location | None: ...
