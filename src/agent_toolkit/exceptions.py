"""Exception hierarchy for llm-agent-toolkit.

All library errors derive from :class:`AgentToolkitError` so callers can
catch one base type at the application boundary.
"""

from __future__ import annotations


class AgentToolkitError(Exception):
    """Base class for all errors raised by this package."""


class ToolRegistrationError(AgentToolkitError):
    """Raised when a tool cannot be registered (e.g. duplicate name)."""


class ToolNotFoundError(AgentToolkitError):
    """Raised when the model requests a tool that is not registered.

    Args:
        name: The requested tool name.
        available: Names of tools that are registered.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        super().__init__(f"Unknown tool {name!r}. Available tools: {sorted(available)}")


class ToolExecutionError(AgentToolkitError):
    """Raised when a tool fails while executing.

    Args:
        tool_name: Name of the failing tool.
        cause: The underlying exception.
    """

    def __init__(self, tool_name: str, cause: BaseException) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(f"Tool {tool_name!r} failed: {cause}")
