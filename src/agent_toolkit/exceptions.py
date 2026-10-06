"""Exception hierarchy for the agent toolkit.

Every error raised deliberately by the library derives from
:class:`AgentToolkitError`, so applications can handle all toolkit failures with a
single ``except`` clause while still being able to discriminate on subclasses.
"""

from __future__ import annotations


class AgentToolkitError(Exception):
    """Base class for all toolkit errors."""


class ConfigurationError(AgentToolkitError):
    """Raised when configuration is missing or invalid.

    Example:
        >>> raise ConfigurationError("AGENT_MAX_STEPS must be an integer")
        Traceback (most recent call last):
        ...
        agent_toolkit.exceptions.ConfigurationError: AGENT_MAX_STEPS must be an integer
    """


class ToolNotFoundError(AgentToolkitError):
    """Raised when the model requests a tool that is not registered.

    Args:
        tool_name: The name that could not be resolved.
        available: Names of the tools that are registered.
    """

    def __init__(self, tool_name: str, available: list[str] | None = None) -> None:
        self.tool_name = tool_name
        self.available = sorted(available or [])
        hint = f" Available tools: {', '.join(self.available)}." if self.available else ""
        super().__init__(f"Unknown tool {tool_name!r}.{hint}")


class ToolExecutionError(AgentToolkitError):
    """Raised when a tool fails while executing.

    Args:
        tool_name: Name of the failing tool.
        reason: Human-readable explanation of the failure.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool {tool_name!r} failed: {reason}")


class MemoryBackendError(AgentToolkitError):
    """Raised when a memory backend cannot read or write state."""


class MaxStepsExceededError(AgentToolkitError):
    """Raised when the agent loop hits its step limit without a final answer.

    Args:
        max_steps: The configured limit that was exceeded.
    """

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        super().__init__(f"Agent did not produce a final answer within {max_steps} steps")
