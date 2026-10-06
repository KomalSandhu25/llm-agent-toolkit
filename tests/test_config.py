"""Tests for agent_toolkit.config."""

import pytest

from agent_toolkit.config import AgentConfig
from agent_toolkit.exceptions import ConfigurationError


def test_defaults_are_valid() -> None:
    cfg = AgentConfig()
    assert cfg.max_steps == 10
    assert cfg.temperature == 0.0
    assert cfg.redis_url is None


def test_config_is_frozen() -> None:
    cfg = AgentConfig()
    with pytest.raises(Exception):
        cfg.max_steps = 3  # type: ignore[misc]


def test_from_env_parses_values() -> None:
    cfg = AgentConfig.from_env(
        {
            "AGENT_MODEL": "gpt-4o",
            "AGENT_TEMPERATURE": "0.7",
            "AGENT_MAX_STEPS": "4",
            "AGENT_VERBOSE": "yes",
            "AGENT_REDIS_URL": "redis://localhost:6379/0",
        }
    )
    assert cfg.model == "gpt-4o"
    assert cfg.temperature == pytest.approx(0.7)
    assert cfg.max_steps == 4
    assert cfg.verbose is True
    assert cfg.redis_url == "redis://localhost:6379/0"


def test_from_env_ignores_unrelated_variables() -> None:
    assert AgentConfig.from_env({"PATH": "/usr/bin"}) == AgentConfig()


def test_from_env_rejects_bad_values() -> None:
    with pytest.raises(ConfigurationError) as info:
        AgentConfig.from_env({"AGENT_MAX_STEPS": "zero"})
    assert "AGENT_MAX_STEPS" in str(info.value)


def test_validation_bounds() -> None:
    with pytest.raises(ConfigurationError):
        AgentConfig.from_env({"AGENT_TEMPERATURE": "5"})
