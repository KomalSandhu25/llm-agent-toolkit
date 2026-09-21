"""Unit tests for src.config.Settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings


def test_defaults_are_valid() -> None:
    """A Settings instance built with no environment overrides should be valid."""
    settings = Settings(_env_file=None)
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.max_iterations > 0
    assert settings.log_level == "INFO"


def test_get_settings_is_cached() -> None:
    """get_settings() must return the same instance on repeated calls."""
    assert get_settings() is get_settings()


@pytest.mark.parametrize("bad_level", ["chatty", "verbose", ""])
def test_invalid_log_level_rejected(bad_level: str) -> None:
    """Log levels outside the standard logging set should fail validation."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level=bad_level)


def test_log_level_is_case_insensitive() -> None:
    """Lower-case log levels should be normalized to upper-case."""
    settings = Settings(_env_file=None, log_level="debug")
    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize("temp", [-0.1, 2.1])
def test_temperature_out_of_range_rejected(temp: float) -> None:
    """Temperature must stay within OpenAI's accepted [0, 2] range."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_temperature=temp)
