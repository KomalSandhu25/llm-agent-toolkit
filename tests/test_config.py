"""Unit tests for :mod:`src.config`."""

from __future__ import annotations

import pytest

from src.config import Settings, get_settings


def test_settings_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should fall back to documented defaults when unset."""
    monkeypatch.delenv("AGENT_MAX_ITERATIONS", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.agent_max_iterations == 8
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.redis_url == "redis://localhost:6379/0"


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should be overridable via environment variables."""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "3")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.openai_model == "gpt-4o"
    assert settings.agent_max_iterations == 3


def test_invalid_temperature_raises() -> None:
    """Temperature outside [0, 2] must fail validation."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, openai_temperature=5.0)  # type: ignore[call-arg]


def test_invalid_log_level_raises() -> None:
    """An unrecognized log level must fail validation."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, log_level="NOISY")  # type: ignore[call-arg]


def test_get_settings_is_memoized() -> None:
    """get_settings() should return the same cached instance every call."""
    assert get_settings() is get_settings()
