"""Abstract base classes implemented by every concrete component.

Nothing in this module talks to OpenAI, Redis, or any other external
system -- these are pure contracts. Concrete implementations (an
OpenAI-backed :class:`BaseLLMClient`, a calculator :class:`BaseTool`, a
Redis-backed :class:`BaseMemory`, a :class:`BaseAgent` that runs the ReAct
loop) land in later revisions of this project; see docs/architecture.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_agent_toolkit.core.types import AgentResponse, ToolResult, ToolSpec


class BaseLLMClient(ABC):
    """Wraps a single chat-completion call behind a stable interface.

    Concrete subclasses adapt a specific provider (OpenAI, a
    LangChain-wrapped model, a local model server) to this shape so the
    agent core never depends on a specific SDK.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> str:
        """Request a single chat completion.

        Args:
            messages: Chat history in ``{"role": ..., "content": ...}``
                form, oldest first.
            tools: Tool specs to expose to the model for function-calling,
                or ``None`` if the call should not offer any tools.

        Returns:
            The model's raw text response.

        Raises:
            llm_agent_toolkit.exceptions.LLMAgentToolkitError: If the
                underlying provider call fails.
        """

        raise NotImplementedError


class BaseTool(ABC):
    """A named, described, schema-validated capability the agent can invoke.

    Every concrete tool (calculator, web search, file reader, ...) planned
    for Day 2 implements this interface, which is what lets the tool
    registry treat them uniformly.
    """

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return this tool's calling contract.

        Returns:
            A :class:`~llm_agent_toolkit.core.types.ToolSpec` describing
            the tool's name, description, and expected arguments.
        """

        raise NotImplementedError

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given keyword arguments.

        Args:
            **kwargs: Arguments matching the JSON schema declared in
                ``self.spec.parameters``.

        Returns:
            A :class:`~llm_agent_toolkit.core.types.ToolResult` describing
            the outcome. Implementations should catch their own exceptions
            and return ``ToolResult(success=False, error=...)`` rather than
            letting exceptions propagate, so a single failing tool call
            cannot crash the agent loop.
        """

        raise NotImplementedError


class BaseMemory(ABC):
    """Append/read/search interface over an agent's interaction history.

    Two concrete backends are planned for Day 3: an in-process
    ``ConversationMemory`` and a persistent, Redis-backed
    ``EpisodicMemory``. Both will implement this same interface so the
    agent core can be pointed at either without any code change.
    """

    @abstractmethod
    async def add(self, role: str, content: str) -> None:
        """Append one turn to the history.

        Args:
            role: The speaker role, e.g. ``"user"``, ``"assistant"``,
                ``"tool"``.
            content: The turn's text content.
        """

        raise NotImplementedError

    @abstractmethod
    async def get_history(self, *, limit: int | None = None) -> list[dict[str, str]]:
        """Return recent history, oldest first.

        Args:
            limit: Maximum number of most-recent turns to return, or
                ``None`` for the backend's full retained history.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts.
        """

        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        """Discard all stored history."""

        raise NotImplementedError


class BaseAgent(ABC):
    """Runs a ReAct loop (or delegates to a planner) to answer a query.

    The concrete ``ReActAgent`` (Day 4) and any planner-driven agent
    (Day 5) both implement this interface, so callers -- the API layer
    (Day 6) and the CLI (Day 7) -- depend only on this contract.
    """

    @abstractmethod
    async def run(self, query: str) -> AgentResponse:
        """Answer a single query, running the agent loop to completion.

        Args:
            query: The user's natural-language request.

        Returns:
            An :class:`~llm_agent_toolkit.core.types.AgentResponse`
            containing the final answer and the full step history that
            produced it.

        Raises:
            llm_agent_toolkit.exceptions.MaxIterationsExceededError: If the
                loop does not reach a final answer within the configured
                iteration budget.
        """

        raise NotImplementedError
