"""Environment-driven configuration for the agent toolkit.

Every setting the toolkit needs — API credentials, model selection, backend
URLs, and runtime limits — is declared on a single :class:`Settings` object.
No other module in this codebase should call ``os.environ`` directly; if a
new component needs a knob, add it here so there is one authoritative list
of everything the toolkit is configurable by.

Example:
    >>> import os
    >>> os.environ["OPENAI_API_KEY"] = "sk-test"
    >>> settings = Settings()
    >>> settings.agent_model
    'gpt-4o-mini'
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the agent toolkit.

    Values are read from environment variables (case-insensitive) or a
    ``.env`` file in the current working directory. Field names map to
    environment variables of the same name in upper case, e.g.
    ``agent_model`` reads from ``AGENT_MODEL``.

    Attributes:
        openai_api_key: API key used by the OpenAI-backed LLM client.
            Required in any environment that actually calls the LLM;
            optional here so the package can be imported and tested without
            credentials present.
        agent_model: Name of the chat-completion model the agent uses by
            default.
        agent_temperature: Sampling temperature passed to the LLM client.
            Must be between 0.0 and 2.0 inclusive.
        max_iterations: Upper bound on ReAct loop iterations for a single
            agent run, used to guarantee termination.
        redis_url: Connection URL for the Redis-backed episodic memory
            store introduced in a later milestone.
        log_level: Root log level for the toolkit's logger.

    Example:
        >>> settings = Settings(openai_api_key="sk-test", max_iterations=5)
        >>> settings.max_iterations
        5
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(
        default=None,
        description="API key for the OpenAI-backed LLM client.",
    )
    agent_model: str = Field(
        default="gpt-4o-mini",
        description="Default chat-completion model used by the agent.",
    )
    agent_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the LLM client.",
    )
    max_iterations: int = Field(
        default=10,
        gt=0,
        description="Maximum ReAct loop iterations before aborting a run.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Connection URL for the episodic memory backend.",
    )
    log_level: str = Field(
        default="INFO",
        description="Root log level for the toolkit's logger.",
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalize and validate the configured log level.

        Args:
            value: Raw log level string from the environment.

        Returns:
            The upper-cased log level.

        Raises:
            ValueError: If the value is not a recognized logging level.

        Example:
            >>> Settings._validate_log_level("debug")
            'DEBUG'
        """
        normalized = value.upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid_levels:
            raise ValueError(
                f"log_level must be one of {sorted(valid_levels)}, got {value!r}"
            )
        return normalized
