"""Application configuration.

Centralizes every tunable used across the agent, memory, and serving
layers behind a single Pydantic ``Settings`` object so that configuration
is validated once, at process startup, instead of being re-parsed from
``os.environ`` throughout the codebase.

Example:
    >>> from src.config import get_settings
    >>> settings = get_settings()
    >>> settings.openai_model
    'gpt-4o-mini'
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings, sourced from the environment.

    Values are read from process environment variables and, if present,
    a local ``.env`` file (see ``.env.example`` for the full set of keys).
    All fields have safe defaults except ``openai_api_key``, which must be
    supplied for the agent to actually call an LLM.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM provider ---
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_model: str = Field(default="gpt-4o-mini", description="Chat completion model name.")
    openai_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    openai_request_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Agent behavior ---
    max_iterations: int = Field(
        default=10, gt=0, description="Max thought/action/observation cycles per task."
    )
    max_tool_retries: int = Field(default=2, ge=0)
    agent_verbose: bool = Field(default=True)

    # --- Memory backend ---
    redis_url: str = Field(default="redis://localhost:6379/0")
    memory_ttl_seconds: int = Field(default=86_400, gt=0)
    max_conversation_turns: int = Field(default=50, gt=0)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    # --- API server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, gt=0, lt=65536)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}, got {value!r}"
            raise ValueError(msg)
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached ``Settings`` instance.

    Using ``lru_cache`` means the environment is parsed exactly once per
    process, and every caller shares the same validated configuration
    object.

    Returns:
        The cached ``Settings`` instance.

    Example:
        >>> s1 = get_settings()
        >>> s2 = get_settings()
        >>> s1 is s2
        True
    """
    return Settings()
