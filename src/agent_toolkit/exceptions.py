"""Exception hierarchy for llm-agent-toolkit.

All errors raised intentionally by the toolkit derive from
:class:`AgentToolkitError`, so applications can catch the whole family with a
single ``except`` clause or handle specific failures individually.
"""

from __future__ import annotations


class AgentToolkitError(Exception):
    """Base class for all toolkit errors."""


class ConfigurationError(AgentToolkitError):
    """Raised when settings are missing or invalid.

    Example:
        >>> raise ConfigurationError("AGENT_MAX_STEPS must be positive")
        Traceback (most recent call last):
        ...
        agent_toolkit.exceptions.ConfigurationError: AGENT_MAX_STEPS must be positive
    """


class ToolNotFoundError(AgentToolkitError):
    """Raised when a requested tool is not registered.

    Args:
        name: The tool name that could not be resolved.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"Tool not found: {name!r}")
        self.name = name


class ToolExecutionError(AgentToolkitError):
    """Raised when a tool fails while executing.

    Args:
        name: Name of the failing tool.
        reason: Human-readable failure description.
    """

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"Tool {name!r} failed: {reason}")
        self.name = name
        self.reason = reason


class MaxStepsExceededError(AgentToolkitError):
    """Raised when the agent loop hits its step limit without a final answer.

    Args:
        max_steps: The step budget that was exhausted.
    """

    def __init__(self, max_steps: int) -> None:
        super().__init__(f"Agent did not finish within {max_steps} steps")
        self.max_steps = max_steps
