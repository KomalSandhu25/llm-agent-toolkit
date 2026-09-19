"""Exception hierarchy for llm_agent_toolkit.

Every error raised anywhere in this package inherits from
:class:`LLMAgentToolkitError`, so callers can catch broadly with a single
except clause, or narrowly by catching one of the specific subtypes below.

Example:
    >>> try:
    ...     raise ToolExecutionError("calculator", "division by zero")
    ... except LLMAgentToolkitError as exc:
    ...     print(str(exc))
    Tool 'calculator' failed: division by zero
"""

from __future__ import annotations


class LLMAgentToolkitError(Exception):
    """Base class for every exception raised by llm_agent_toolkit."""


class ToolExecutionError(LLMAgentToolkitError):
    """Raised when a :class:`~llm_agent_toolkit.core.interfaces.BaseTool`
    fails during execution.

    Args:
        tool_name: The registered name of the tool that failed.
        reason: A human-readable description of what went wrong.

    Example:
        >>> raise ToolExecutionError("web_search", "timed out after 30s")
        Traceback (most recent call last):
            ...
        llm_agent_toolkit.exceptions.ToolExecutionError: Tool 'web_search' failed: timed out after 30s
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' failed: {reason}")


class MemoryBackendError(LLMAgentToolkitError):
    """Raised when a :class:`~llm_agent_toolkit.core.interfaces.BaseMemory`
    backend cannot read or write agent history (e.g. a Redis connection
    failure in the episodic memory backend planned for Day 3).
    """

    def __init__(self, backend_name: str, reason: str) -> None:
        self.backend_name = backend_name
        self.reason = reason
        super().__init__(f"Memory backend '{backend_name}' failed: {reason}")


class AgentExecutionError(LLMAgentToolkitError):
    """Raised when the agent core (planned for Day 4) cannot complete a run
    for a reason that is not covered by a more specific exception below.
    """


class MaxIterationsExceededError(AgentExecutionError):
    """Raised when a ReAct loop reaches ``max_iterations`` without producing
    a ``FINAL_ANSWER`` step.

    Args:
        max_iterations: The iteration cap that was configured.

    Example:
        >>> raise MaxIterationsExceededError(10)
        Traceback (most recent call last):
            ...
        llm_agent_toolkit.exceptions.MaxIterationsExceededError: Agent did not reach a final answer within 10 iterations
    """

    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent did not reach a final answer within {max_iterations} iterations"
        )


class PlanningError(LLMAgentToolkitError):
    """Raised by the task planner (planned for Day 5) when a goal cannot be
    decomposed into an executable plan.
    """
