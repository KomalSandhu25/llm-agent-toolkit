"""Centralized application configuration.

All runtime configuration is defined here as a single typed ``Settings``
object, loaded from environment variables (and an optional ``.env`` file via
``python-dotenv`` / ``pydantic-settings``). Every other module should import
its configuration from here rather than reading ``os.environ`` directly, so
that behavior stays predictable and testable.

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
    """Strongly typed, environment-driven configuration for the agent toolkit.

    Fields are grouped by subsystem: LLM provider, agent execution behavior,
    memory backend, API server, and logging. Values are read from environment
    variables (case-insensitive) or a ``.env`` file in the project root.

    Attributes:
        openai_api_key: API key used to authenticate with the OpenAI API.
        openai_model: Model identifier used for both reasoning and tool-call
            generation steps.
        openai_temperature: Sampling temperature for the LLM calls.
        openai_max_tokens: Maximum number of tokens to generate per LLM call.
        agent_max_iterations: Hard cap on ReAct loop iterations per task, used
            to guarantee termination even if the agent fails to converge.
        agent_max_tool_retries: Number of times a single tool call may be
            retried after a transient failure before the step is marked failed.
        agent_verbose: Whether the agent should emit step-by-step trace output.
        redis_url: Connection URL for the Redis-backed memory store.
        memory_ttl_seconds: Expiry, in seconds, for stored conversation state.
        memory_max_turns: Maximum number of turns retained per conversation
            before older turns are evicted (sliding window).
        api_host: Bind host for the FastAPI server.
        api_port: Bind port for the FastAPI server.
        log_level: Root logging level, e.g. ``"INFO"`` or ``"DEBUG"``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM provider ---
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=1024, gt=0)

    # --- Agent behavior ---
    agent_max_iterations: int = Field(default=8, gt=0, le=50)
    agent_max_tool_retries: int = Field(default=2, ge=0, le=10)
    agent_verbose: bool = Field(default=True)

    # --- Memory backend ---
    redis_url: str = Field(default="redis://localhost:6379/0")
    memory_ttl_seconds: int = Field(default=86_400, gt=0)
    memory_max_turns: int = Field(default=50, gt=0)

    # --- API server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, gt=0, lt=65_536)

    # --- Logging ---
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalize and validate the configured logging level.

        Args:
            value: Raw log level string from the environment.

        Returns:
            The upper-cased log level string.

        Raises:
            ValueError: If the level is not a recognized ``logging`` level name.
        """
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide ``Settings`` instance.

    Using ``lru_cache`` ensures the environment is parsed once and the same
    validated settings object is reused everywhere, which keeps behavior
    consistent within a single process and avoids repeated I/O.

    Returns:
        The singleton ``Settings`` instance for this process.

    Example:
        >>> from src.config import get_settings
        >>> get_settings() is get_settings()
        True
    """
    return Settings()
