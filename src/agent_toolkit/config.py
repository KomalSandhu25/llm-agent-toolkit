"""Configuration for the agent toolkit.

:class:`AgentConfig` is a validated, immutable Pydantic model. It can be built directly,
or loaded from environment variables with :meth:`AgentConfig.from_env`, which keeps
deployment-specific values (model name, Redis URL, ...) out of the code.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_toolkit.exceptions import ConfigurationError

ENV_PREFIX = "AGENT_"


class AgentConfig(BaseModel):
    """Runtime settings for an agent.

    Attributes:
        model: Name of the chat model to use.
        temperature: Sampling temperature in ``[0, 2]``.
        max_steps: Maximum thought/action/observation iterations per task.
        system_prompt: Instructions prepended to every conversation.
        memory_window: Number of most recent messages kept in short-term memory.
        redis_url: Optional Redis URL for persistent memory backends.
        verbose: Whether components should emit detailed progress output.

    Example:
        >>> cfg = AgentConfig(model="gpt-4o-mini", max_steps=5)
        >>> cfg.max_steps
        5
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model: str = Field(default="gpt-4o-mini", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_steps: int = Field(default=10, ge=1, le=100)
    system_prompt: str = Field(
        default=(
            "You are a helpful assistant that solves tasks step by step. "
            "Think before acting, use tools when they help, and give a concise final answer."
        ),
        min_length=1,
    )
    memory_window: int = Field(default=20, ge=1)
    redis_url: Optional[str] = None
    verbose: bool = False

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "AgentConfig":
        """Build a config from ``AGENT_*`` environment variables.

        Recognised variables: ``AGENT_MODEL``, ``AGENT_TEMPERATURE``,
        ``AGENT_MAX_STEPS``, ``AGENT_SYSTEM_PROMPT``, ``AGENT_MEMORY_WINDOW``,
        ``AGENT_REDIS_URL`` and ``AGENT_VERBOSE`` (``1/true/yes/on`` enable it).

        Args:
            environ: Mapping to read from. Defaults to ``os.environ``.

        Returns:
            A validated configuration; unset variables fall back to defaults.

        Raises:
            ConfigurationError: If a value cannot be parsed or fails validation.

        Example:
            >>> AgentConfig.from_env({"AGENT_MAX_STEPS": "3"}).max_steps
            3
        """
        env = os.environ if environ is None else environ
        raw: dict[str, object] = {}
        for field_name in cls.model_fields:
            key = f"{ENV_PREFIX}{field_name.upper()}"
            if key in env:
                raw[field_name] = env[key]

        if "verbose" in raw:
            raw["verbose"] = str(raw["verbose"]).strip().lower() in {"1", "true", "yes", "on"}

        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(
                f"{ENV_PREFIX}{'.'.join(str(p) for p in err['loc']).upper()}: {err['msg']}"
                for err in exc.errors()
            )
            raise ConfigurationError(f"Invalid agent configuration: {problems}") from exc
