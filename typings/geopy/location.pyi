from __future__ import annotations

from typing import Any, Mapping


class Location:
    address: str
    latitude: float
    longitude: float
    raw: Mapping[str, Any]
