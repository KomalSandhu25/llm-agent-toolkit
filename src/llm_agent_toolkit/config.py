"""Environment-driven configuration for llm_agent_toolkit.

All settings are defined on a single :class:`Settings` model so the whole
application has one source of truth for configuration, validated once at
startup rather than scattered across ``os.environ.get(...)`` calls.

Example:
    >>> import os
    >>> os.environ["LLM_AGENT_MAX_ITERATIONS"] = "5"
    >>> settings = Settings()
    >>> settings.max_iterations
    5
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the agent toolkit.

    Every field can be overridden by an environment variable named
    ``LLM_AGENT_<FIELD_NAME_UPPERCASE>``, e.g. ``LLM_AGENT_MODEL_NAME``.

    Attributes:
        openai_api_key: API key used by the OpenAI-backed LLM client.
            Optional at import time so the test suite can run without one;
            required once a real :class:`BaseLLMClient` implementation is
            constructed.
        model_name: Chat-completion model identifier passed to the LLM
            client.
        temperature: Sampling temperature for LLM calls. Must be within
            ``[0.0, 2.0]``, matching the OpenAI API's accepted range.
        max_iterations: Upper bound on ReAct loop iterations (THOUGHT ->
            ACTION -> OBSERVATION) before :class:`MaxIterationsExceededError`
            is raised.
        redis_url: Connection string for the episodic memory backend
            (planned for Day 3).
        conversation_memory_max_turns: Number of most-recent turns the
            in-process conversation memory buffer retains.
        request_timeout_seconds: Timeout applied to outbound LLM and tool
            calls.

    Example:
        >>> settings = Settings(model_name="gpt-4o-mini", max_iterations=8)
        >>> settings.max_iterations
        8
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, repr=False)
    model_name: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_iterations: int = Field(default=10, ge=1, le=100)
    redis_url: str = Field(default="redis://localhost:6379/0")
    conversation_memory_max_turns: int = Field(default=20, ge=1)
    request_timeout_seconds: float = Field(default=30.0, gt=0.0)

    @field_validator("model_name")
    @classmethod
    def _model_name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_name must not be blank")
        return value


def get_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment.

    This is a thin factory (rather than a module-level singleton) so tests
    can freely construct settings with different environment variables set
    without needing to reset global state between cases.

    Returns:
        A validated :class:`Settings` instance.

    Example:
        >>> settings = get_settings()
        >>> isinstance(settings.max_iterations, int)
        True
    """

    return Settings()
