"""Bootstrap helpers shared across command-line entry points."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


def _resolve_project_root() -> Path:
    """Return the repository root containing the ``src`` package."""

    current = Path(__file__).resolve().parents[1]
    marker = current / "src"
    if not marker.exists():
        raise RuntimeError(
            "Could not locate the project root. Expected a 'src' directory next to scripts/."
        )
    return current


@lru_cache(maxsize=1)
def bootstrap_project() -> Path:
    """Ensure the repository root is present on ``sys.path``.

    The function is cached so multiple invocations are effectively no-ops after the
    first call, keeping imports idempotent and inexpensive.
    """

    project_root = _resolve_project_root()
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root

