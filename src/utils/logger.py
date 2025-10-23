"""Centralised logger factory with graceful fallback when Loguru is absent."""
from __future__ import annotations

import importlib
import logging
from typing import Any


def _build_logger() -> Any:
    """Return a project-wide logger instance.

    If Loguru is available it is preferred for its rich formatting. Otherwise the
    standard library ``logging`` module is used to keep the application working in
    minimal environments such as the execution sandbox used for automated tests.
    """

    spec = importlib.util.find_spec("loguru")
    if spec is not None:
        module = importlib.import_module("loguru")
        return module.logger

    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("movilidad_inteligente_nlp")


logger = _build_logger()

__all__ = ["logger"]
