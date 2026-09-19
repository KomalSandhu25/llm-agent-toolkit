"""Rich-based structured logging setup for llm_agent_toolkit.

The agent loop is inherently narrative -- a sequence of THOUGHT / ACTION /
OBSERVATION steps -- so logging is configured to render nicely in a
terminal via :mod:`rich`, while still being plain :mod:`logging` underneath
so it composes with whatever logging setup a host application already has.

Example:
    >>> logger = configure_logging(level="DEBUG")
    >>> logger.info("agent started")  # doctest: +SKIP
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_LOGGER_NAME = "llm_agent_toolkit"
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the package-wide logger.

    Idempotent: calling this multiple times will not attach duplicate
    handlers. Subsequent calls only adjust the log level.

    Args:
        level: A standard :mod:`logging` level name, e.g. ``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``.

    Returns:
        The configured ``llm_agent_toolkit`` logger.

    Example:
        >>> logger = configure_logging("WARNING")
        >>> logger.level
        30
    """

    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)

    if not _CONFIGURED:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True

    logger.setLevel(level.upper())
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``llm_agent_toolkit`` namespace.

    Ensures :func:`configure_logging` has been called at least once with
    default settings, so any module can call this without needing to worry
    about initialization order.

    Args:
        name: Optional dotted suffix, e.g. ``"tools.calculator"`` yields a
            logger named ``"llm_agent_toolkit.tools.calculator"``. If
            ``None``, the root package logger is returned.

    Returns:
        A :class:`logging.Logger` instance.

    Example:
        >>> logger = get_logger("tools.calculator")
        >>> logger.name
        'llm_agent_toolkit.tools.calculator'
    """

    if not logging.getLogger(_LOGGER_NAME).handlers:
        configure_logging()
    return logging.getLogger(_LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}")
