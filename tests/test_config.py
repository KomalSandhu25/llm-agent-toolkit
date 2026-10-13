"""Tests for agent_toolkit.config and logging."""

import logging

import pytest

from agent_toolkit.config import load_settings
from agent_toolkit.exceptions import ConfigurationError
from agent_toolkit.logging_utils import configure_logging


def test_defaults() -> None:
    s = load_settings({})
    assert s.model == "gpt-4o-mini"
    assert s.max_steps == 8
    assert s.openai_api_key is None


def test_env_overrides_and_coercion() -> None:
    s = load_settings(
        {"AGENT_MAX_STEPS": "5", "AGENT_TEMPERATURE": "0.7", "AGENT_LOG_LEVEL": "debug"}
    )
    assert s.max_steps == 5
    assert s.temperature == pytest.approx(0.7)
    assert s.log_level == "DEBUG"


def test_empty_values_ignored() -> None:
    assert load_settings({"AGENT_MAX_STEPS": ""}).max_steps == 8


@pytest.mark.parametrize(
    "env",
    [
        {"AGENT_MAX_STEPS": "0"},
        {"AGENT_TEMPERATURE": "9"},
        {"AGENT_LOG_LEVEL": "loud"},
        {"AGENT_MAX_STEPS": "x"},
    ],
)
def test_invalid_raises(env: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        load_settings(env)


def test_settings_frozen() -> None:
    s = load_settings({})
    with pytest.raises(Exception):
        s.max_steps = 3  # type: ignore[misc]


def test_configure_logging_idempotent() -> None:
    configure_logging("DEBUG")
    logger = configure_logging("WARNING")
    assert len(logger.handlers) == 1
    assert logger.level == logging.WARNING
