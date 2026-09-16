"""Tests for llm_agent_toolkit.config.Settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_agent_toolkit.config import Settings


def test_defaults_require_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should construct with sensible defaults and no API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.openai_api_key is None
    assert settings.agent_model == "gpt-4o-mini"
    assert settings.max_iterations == 10
    assert settings.log_level == "INFO"


def test_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should pick up values from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("AGENT_MODEL", "gpt-4o")
    monkeypatch.setenv("MAX_ITERATIONS", "25")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.openai_api_key == "sk-test-123"
    assert settings.agent_model == "gpt-4o"
    assert settings.max_iterations == 25


def test_log_level_is_normalized_to_uppercase() -> None:
    """Lower-case log levels should be normalized on construction."""
    settings = Settings(log_level="debug", _env_file=None)  # type: ignore[call-arg]
    assert settings.log_level == "DEBUG"


def test_invalid_log_level_raises() -> None:
    """An unrecognized log level should fail validation."""
    with pytest.raises(ValidationError):
        Settings(log_level="not-a-level", _env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_out_of_range_raises(temperature: float) -> None:
    """Temperature outside [0, 2] should fail validation."""
    with pytest.raises(ValidationError):
        Settings(agent_temperature=temperature, _env_file=None)  # type: ignore[call-arg]


def test_max_iterations_must_be_positive() -> None:
    """A non-positive iteration budget should fail validation."""
    with pytest.raises(ValidationError):
        Settings(max_iterations=0, _env_file=None)  # type: ignore[call-arg]
