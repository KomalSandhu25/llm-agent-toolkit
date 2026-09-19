"""Tests for llm_agent_toolkit.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_agent_toolkit.config import Settings, get_settings


def test_defaults_are_sane() -> None:
    settings = Settings(_env_file=None)
    assert settings.model_name == "gpt-4o-mini"
    assert settings.max_iterations == 10
    assert 0.0 <= settings.temperature <= 2.0
    assert settings.openai_api_key is None


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_AGENT_MAX_ITERATIONS", "3")
    monkeypatch.setenv("LLM_AGENT_MODEL_NAME", "gpt-4o")
    settings = Settings(_env_file=None)
    assert settings.max_iterations == 3
    assert settings.model_name == "gpt-4o"


def test_temperature_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, temperature=3.5)


def test_blank_model_name_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, model_name="   ")


def test_get_settings_returns_settings_instance() -> None:
    assert isinstance(get_settings(), Settings)
