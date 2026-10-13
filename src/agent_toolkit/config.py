"""Runtime configuration for llm-agent-toolkit.

Settings are read from environment variables prefixed with ``AGENT_`` (plus
``OPENAI_API_KEY``) and validated with Pydantic. Pass a :class:`Settings`
instance explicitly to components rather than reading global state.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from agent_toolkit.exceptions import ConfigurationError

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseModel):
    """Validated, immutable application settings.

    Attributes:
        openai_api_key: API key for the LLM provider (may be unset for offline tests).
        model: Chat model identifier.
        temperature: Sampling temperature in ``[0, 2]``.
        max_steps: Maximum ReAct iterations per request.
        redis_url: Connection URL for the Redis memory backend.
        log_level: Standard logging level name.

    Example:
        >>> Settings(max_steps=3).max_steps
        3
    """

    model_config = {"frozen": True}

    openai_api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_steps: int = Field(default=8, ge=1, le=64)
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in _LOG_LEVELS:
            raise ValueError(f"log_level must be one of {sorted(_LOG_LEVELS)}")
        return upper


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Build :class:`Settings` from environment variables.

    Args:
        env: Mapping to read from. Defaults to ``os.environ``.

    Returns:
        A validated ``Settings`` instance.

    Raises:
        ConfigurationError: If any value fails validation.

    Example:
        >>> load_settings({"AGENT_MAX_STEPS": "5"}).max_steps
        5
    """
    source = os.environ if env is None else env
    raw: dict[str, object] = {}
    mapping = {
        "OPENAI_API_KEY": "openai_api_key",
        "AGENT_MODEL": "model",
        "AGENT_TEMPERATURE": "temperature",
        "AGENT_MAX_STEPS": "max_steps",
        "AGENT_REDIS_URL": "redis_url",
        "AGENT_LOG_LEVEL": "log_level",
    }
    for env_name, field in mapping.items():
        value = source.get(env_name)
        if value not in (None, ""):
            raw[field] = value
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
