"""Centralized, typed application configuration.

All runtime configuration for the agent toolkit is defined here as a single
Pydantic ``BaseSettings`` subclass. Values are sourced from environment
variables (or a local ``.env`` file, see ``.env.example``) so that the same
code runs unmodified across local development, tests, and containerized
deployments.

Downstream modules should never read ``os.environ`` directly; instead they
should depend on :func:`get_settings`, which is cached so the environment is
parsed and validated exactly once per process.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Attributes:
        openai_api_key: Secret key used to authenticate with the OpenAI API.
        openai_model: Chat-completion model identifier the agent calls.
        openai_temperature: Sampling temperature passed to the model,
            constrained to ``[0.0, 2.0]``.
        openai_max_tokens: Maximum number of tokens requested per completion.
        openai_request_timeout_seconds: Per-request timeout for OpenAI calls.
        agent_max_iterations: Hard cap on thought/action/observation cycles
            before the agent aborts with ``MaxIterationsExceededError``.
        agent_max_tool_retries: Number of times a failed tool call is
            retried before the failure is surfaced to the agent as an
            observation.
        agent_verbose: When ``True``, the agent logs each reasoning step at
            INFO level (and the CLI renders it live).
        redis_url: Connection string for the Redis-backed memory store.
        memory_ttl_seconds: Time-to-live applied to persisted conversation
            state in Redis.
        memory_max_turns: Maximum number of conversation turns retained in
            memory before older turns are evicted.
        api_host: Bind host for the FastAPI server.
        api_port: Bind port for the FastAPI server.
        log_level: Root logging level, e.g. ``"INFO"`` or ``"DEBUG"``.

    Example:
        >>> import os
        >>> os.environ["OPENAI_API_KEY"] = "sk-test"
        >>> settings = Settings()
        >>> settings.agent_max_iterations
        8
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = Field(default="sk-placeholder")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=1024, gt=0)
    openai_request_timeout_seconds: int = Field(default=30, gt=0)

    # Agent behavior
    agent_max_iterations: int = Field(default=8, gt=0, le=50)
    agent_max_tool_retries: int = Field(default=2, ge=0, le=10)
    agent_verbose: bool = Field(default=True)

    # Memory
    redis_url: str = Field(default="redis://localhost:6379/0")
    memory_ttl_seconds: int = Field(default=86_400, gt=0)
    memory_max_turns: int = Field(default=50, gt=0)

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, gt=0, lt=65_536)

    # Logging
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalize and validate the configured log level.

        Args:
            value: Raw log level string from the environment.

        Returns:
            The upper-cased log level.

        Raises:
            ValueError: If ``value`` is not a recognized logging level name.
        """
        normalized = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, memoized :class:`Settings` instance.

    Using ``lru_cache`` guarantees the environment is parsed once, so
    settings stay consistent for the lifetime of the process even if
    ``os.environ`` is mutated later (e.g. by a test fixture running before
    this is first called).

    Returns:
        The cached :class:`Settings` instance.

    Example:
        >>> settings_a = get_settings()
        >>> settings_b = get_settings()
        >>> settings_a is settings_b
        True
    """
    return Settings()
