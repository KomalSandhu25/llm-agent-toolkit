"""Abstract interfaces for every swappable component in the toolkit.

Each class here defines a contract, not a behavior: concrete
implementations (an OpenAI-backed LLM client, a calculator tool, a
Redis-backed memory store, ...) are added in later milestones. The agent
core (also a later milestone) depends only on these interfaces, never on a
concrete implementation, so any of them can be swapped without touching the
core loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from llm_agent_toolkit.core.types import AgentStatus, Message, ToolCall, ToolResult


class BaseLLMClient(ABC):
    """Contract for a client that turns conversation history into a reply.

    Implementations translate the toolkit's provider-agnostic
    :class:`~llm_agent_toolkit.core.types.Message` objects into a specific
    provider's request format and translate the response back, so no
    provider-specific concept (e.g. a particular function-calling wire
    format) leaks into the agent core.
    """

    @abstractmethod
    async def generate(self, messages: list[Message], **kwargs: Any) -> Message:
        """Generate the next assistant message given conversation history.

        Args:
            messages: Full conversation history, oldest first.
            **kwargs: Provider-specific overrides (e.g. ``temperature``).

        Returns:
            The assistant's reply as a single :class:`Message`.

        Raises:
            llm_agent_toolkit.exceptions.LLMClientError: If the provider
                request fails or returns an unusable response.
        """
        raise NotImplementedError

    @abstractmethod
    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        """Stream the next assistant message incrementally.

        Args:
            messages: Full conversation history, oldest first.
            **kwargs: Provider-specific overrides (e.g. ``temperature``).

        Yields:
            Successive text chunks of the assistant's reply.

        Raises:
            llm_agent_toolkit.exceptions.LLMClientError: If the provider
                request fails or returns an unusable response.
        """
        raise NotImplementedError
        yield ""  # pragma: no cover - makes this an async generator for type checkers


class BaseTool(ABC):
    """Contract for a single tool the agent can invoke.

    A tool advertises its own name, description, and JSON-schema parameter
    definition so that an :class:`BaseLLMClient` implementation can present
    it to the underlying model for tool selection, and so arguments can be
    validated before ``execute`` is called.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique, stable identifier for this tool (e.g. ``"calculator"``)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural-language description shown to the LLM for tool selection."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON schema describing the keyword arguments ``execute`` accepts."""
        raise NotImplementedError

    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Run this tool for a specific invocation.

        Args:
            tool_call: The call to execute, including its arguments.

        Returns:
            The result of running the tool, with ``is_error`` set if the
            tool could not complete successfully.

        Raises:
            llm_agent_toolkit.exceptions.ToolExecutionError: If execution
                fails in a way the caller should treat as exceptional
                rather than as an error-carrying :class:`ToolResult`.
        """
        raise NotImplementedError


class BaseMemory(ABC):
    """Contract for a conversation memory backend.

    Implementations may be as simple as an in-process list (a conversation
    buffer) or as involved as a Redis-backed episodic store with
    summarization. The agent core only ever calls the three methods below.
    """

    @abstractmethod
    async def add(self, message: Message) -> None:
        """Persist a message to memory.

        Args:
            message: The message to store.

        Raises:
            llm_agent_toolkit.exceptions.MemoryBackendError: If the message
                cannot be persisted.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_history(self, limit: int | None = None) -> list[Message]:
        """Retrieve stored messages, oldest first.

        Args:
            limit: Maximum number of most-recent messages to return. When
                ``None``, the full available history is returned.

        Returns:
            A list of messages, oldest first, with at most ``limit``
            entries when ``limit`` is given.

        Raises:
            llm_agent_toolkit.exceptions.MemoryBackendError: If history
                cannot be retrieved.
        """
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        """Delete all stored messages for this memory instance.

        Raises:
            llm_agent_toolkit.exceptions.MemoryBackendError: If the backend
                cannot be cleared.
        """
        raise NotImplementedError


class BasePlanner(ABC):
    """Contract for decomposing a high-level goal into ordered sub-goals."""

    @abstractmethod
    async def create_plan(self, goal: str, context: list[Message] | None = None) -> list[str]:
        """Break a goal down into an ordered list of sub-goals.

        Args:
            goal: The high-level objective to decompose.
            context: Optional conversation history to inform planning.

        Returns:
            An ordered list of sub-goal descriptions, to be tackled one at
            a time by the agent.

        Raises:
            llm_agent_toolkit.exceptions.PlanningError: If a usable plan
                cannot be produced.
        """
        raise NotImplementedError


class BaseAgent(ABC):
    """Contract for an agent that pursues a goal to completion.

    A concrete implementation composes a :class:`BaseLLMClient`, a
    collection of :class:`BaseTool` instances, a :class:`BaseMemory`, and
    optionally a :class:`BasePlanner` behind this single ``run`` entry
    point.
    """

    @abstractmethod
    async def run(self, goal: str) -> AgentStatus:
        """Pursue a goal until it finishes, errors, or exhausts its budget.

        Args:
            goal: The natural-language objective for the agent to pursue.

        Returns:
            The terminal :class:`~llm_agent_toolkit.core.types.AgentStatus`
            of the run.

        Raises:
            llm_agent_toolkit.exceptions.MaxIterationsExceededError: If the
                run exceeds its configured iteration budget without
                finishing.
        """
        raise NotImplementedError
