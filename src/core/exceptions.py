"""Exception hierarchy shared across the agent toolkit.

Every module raises a subclass of :class:`AgentToolkitError` rather than a
bare ``Exception`` or ``ValueError``, so callers (the API layer, the CLI,
and tests) can catch toolkit-specific failures precisely and map them to
the right HTTP status code or user-facing message.
"""

from __future__ import annotations


class AgentToolkitError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(AgentToolkitError):
    """Raised when application settings are missing or invalid."""


class ToolNotFoundError(AgentToolkitError):
    """Raised when the agent requests a tool that isn't registered.

    Attributes:
        tool_name: The name that was looked up and not found.
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"No tool registered under the name {tool_name!r}.")


class ToolExecutionError(AgentToolkitError):
    """Raised when a tool raises while executing.

    Attributes:
        tool_name: The name of the tool that failed.
        original_error: The underlying exception that caused the failure.
    """

    def __init__(self, tool_name: str, original_error: Exception) -> None:
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool {tool_name!r} raised {original_error!r}.")


class MaxIterationsExceededError(AgentToolkitError):
    """Raised when the agent exceeds ``Settings.agent_max_iterations``.

    Attributes:
        max_iterations: The configured limit that was exceeded.
        task: The task the agent was working on when it aborted.
    """

    def __init__(self, max_iterations: int, task: str) -> None:
        self.max_iterations = max_iterations
        self.task = task
        super().__init__(
            f"Agent exceeded the maximum of {max_iterations} iterations "
            f"while working on task: {task!r}."
        )


class MemoryBackendError(AgentToolkitError):
    """Raised when a memory backend (e.g. Redis) fails to read or write."""
