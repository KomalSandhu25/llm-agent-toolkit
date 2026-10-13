"""Logging helpers using Rich for readable console output."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the ``agent_toolkit`` logger.

    Idempotent: repeated calls replace the handler instead of stacking them.

    Args:
        level: Logging level name (e.g. ``"DEBUG"``).

    Returns:
        The configured package logger.

    Example:
        >>> configure_logging("DEBUG").level == logging.DEBUG
        True
    """
    logger = logging.getLogger("agent_toolkit")
    logger.handlers.clear()
    handler = RichHandler(rich_tracebacks=True, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger
