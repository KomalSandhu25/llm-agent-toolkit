"""Centralised, environment-driven configuration.

Settings are read from environment variables (and an optional ``.env`` file)
and validated by Pydantic. Use :func:`get_settings` for a cached instance.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the agent framework.

    Example:
        >>> settings = Settings(openai_api_key="sk-test", agent_max_steps=5)
        >>> settings.agent_max_steps
        5
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    openai_api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key.")
    agent_model: str = Field(default="gpt-4o-mini", description="Chat model identifier.")
    agent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    agent_max_steps: int = Field(default=10, ge=1, le=100, description="Max ReAct iterations.")
    agent_tool_timeout_seconds: float = Field(default=30.0, gt=0.0)

    redis_url: str = Field(default="redis://localhost:6379/0")
    agent_memory_ttl_seconds: int = Field(default=86_400, ge=0)

    agent_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Returns:
        The validated settings object.

    Example:
        >>> get_settings() is get_settings()
        True
    """
    return Settings()
