"""Core entities for the mobility intelligent NLP domain."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class TrafficEvent:
    """Domain entity representing a traffic-related tweet."""

    text: str
    created_at: Optional[datetime]
    predicted_category: str
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    severity: Optional[str] = None
    priority: Optional[str] = None
    priority_score: Optional[float] = None


@dataclass(frozen=True)
class UserProfile:
    """Domain entity describing a user's interest profile."""

    user_id: str
    interests: list[str]
    recent_interactions: Optional[list[str]] = None
    age: Optional[int] = None
    gender: Optional[str] = None


__all__ = ["TrafficEvent", "UserProfile"]
