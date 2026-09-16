"""Exception hierarchy for the agent toolkit.

All exceptions raised by this package inherit from
:class:`AgentToolkitError`, so calling code can catch broadly
(``except AgentToolkitError``) or narrowly (``except ToolExecutionError``)
depending on how it needs to respond.

Example:
    >>> try:
    ...     raise ToolNotFoundError("calculator")
    ... except AgentToolkitError as exc:
    ...     print(str(exc))
    Tool not found: 'calculator'
"""

from __future__ import annotations


class AgentToolkitError(Exception):
    """Base class for every exception raised by llm-agent-toolkit."""


class ConfigurationError(AgentToolkitError):
    """Raised when required configuration is missing or invalid."""


class LLMClientError(AgentToolkitError):
    """Raised when an LLM client fails to produce a response.

    Example:
        >>> raise LLMClientError("OpenAI request timed out after 30s")
        Traceback (most recent call last):
            ...
        llm_agent_toolkit.exceptions.LLMClientError: OpenAI request timed out after 30s
    """


class ToolNotFoundError(AgentToolkitError):
    """Raised when the agent requests a tool that isn't registered.

    Args:
        tool_name: The name of the tool that could not be found.

    Example:
        >>> raise ToolNotFoundError("calculator")
        Traceback (most recent call last):
            ...
        llm_agent_toolkit.exceptions.ToolNotFoundError: Tool not found: 'calculator'
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool not found: {tool_name!r}")


class ToolExecutionError(AgentToolkitError):
    """Raised when a registered tool raises while executing.

    Args:
        tool_name: The name of the tool that failed.
        original_error: The underlying exception that triggered this error.

    Example:
        >>> raise ToolExecutionError("search", ValueError("bad query"))
        Traceback (most recent call last):
            ...
        llm_agent_toolkit.exceptions.ToolExecutionError: Tool 'search' failed: bad query
    """

    def __init__(self, tool_name: str, original_error: Exception) -> None:
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool {tool_name!r} failed: {original_error}")


class MemoryBackendError(AgentToolkitError):
    """Raised when a memory backend fails to read or write state."""


class PlanningError(AgentToolkitError):
    """Raised when the planner cannot produce a valid plan for a goal."""


class MaxIterationsExceededError(AgentToolkitError):
    """Raised when an agent run exceeds its configured iteration budget.

    Args:
        max_iterations: The iteration budget that was exceeded.

    Example:
        >>> raise MaxIterationsExceededError(10)
        Traceback (most recent call last):
            ...
        llm_agent_toolkit.exceptions.MaxIterationsExceededError: Agent run exceeded 10 iterations without finishing
    """

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent run exceeded {max_iterations} iterations without finishing"
        )
