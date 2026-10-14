"""Centralised, environment-driven configuration for the agent toolkit.

All runtime knobs (model choice, step budget, Redis location) are declared once
here as a validated ``pydantic-settings`` model and read from environment
variables or a local ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings.

    Args:
        openai_api_key: API key for the OpenAI client.
        agent_model: Chat model used for reasoning steps.
        agent_temperature: Sampling temperature (0 for deterministic traces).
        agent_max_steps: Hard cap on Thought/Action/Observation iterations.
        redis_url: Connection URL for the persistent memory backend.
        memory_ttl_seconds: Expiry applied to stored conversation memory.
        log_level: Python logging level name.

    Example:
        >>> Settings(openai_api_key="sk-test").agent_max_steps
        8
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    openai_api_key: SecretStr = Field(default=SecretStr(""))
    agent_model: str = "gpt-4o-mini"
    agent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    agent_max_steps: int = Field(default=8, ge=1, le=50)
    redis_url: str = "redis://localhost:6379/0"
    memory_ttl_seconds: int = Field(default=86_400, ge=1)
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Returns:
        The process-wide settings object.

    Example:
        >>> get_settings() is get_settings()
        True
    """
    return Settings()
