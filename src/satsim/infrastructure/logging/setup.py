"""Structured logging configuration for SatSim.

Uses the standard library by default so the scaffold has zero hard dependency
on structlog at import time. When ``structlog`` is installed, callers may
extend this module to bind rich processors.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    fmt: str | None = None,
) -> None:
    """Configure root logging for CLI and scripts.

    Args:
        level: Logging level name or numeric level.
        stream: Output stream (defaults to stderr).
        fmt: Optional format string.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt
            or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third parties by default when they appear later.
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
