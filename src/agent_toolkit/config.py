"""Runtime configuration for the agent toolkit, loaded from env vars / .env."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Typed settings for the agent runtime.

    Args:
        openai_api_key: API key for the LLM provider.
        model: Chat model identifier.
        max_steps: Maximum ReAct iterations before the agent gives up.
        temperature: Sampling temperature.
        redis_url: Connection URL for the persistent memory backend.

    Example:
        >>> AgentSettings(max_steps=3).max_steps
        3
    """

    openai_api_key: str = ""
    model: str = Field(default="gpt-4o-mini", alias="AGENT_MODEL")
    max_steps: int = Field(default=8, ge=1, le=64, alias="AGENT_MAX_STEPS")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="AGENT_TEMPERATURE")
    redis_url: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )
