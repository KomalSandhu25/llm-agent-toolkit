"""Structural interfaces (Protocols) defining the toolkit's architecture.

Concrete implementations arrive in later milestones: an OpenAI-backed
:class:`LLMClient`, a Redis-backed :class:`Memory`, and an LLM-driven
:class:`Planner`. Defining the seams first keeps the agent loop testable
with simple fakes.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from agent_toolkit.schemas import Message


@runtime_checkable
class LLMClient(Protocol):
    """Chat-completion backend."""

    def complete(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> Message:
        """Return the assistant message for ``messages``.

        Args:
            messages: Conversation so far.
            tools: Optional OpenAI-style tool schemas the model may call.

        Returns:
            The assistant reply, possibly containing ``tool_calls``.
        """
        ...


@runtime_checkable
class Memory(Protocol):
    """Persistent conversation memory keyed by session."""

    def load(self, session_id: str) -> list[Message]:
        """Load all stored messages for ``session_id``."""
        ...

    def append(self, session_id: str, message: Message) -> None:
        """Append ``message`` to the session history."""
        ...

    def clear(self, session_id: str) -> None:
        """Delete the session history."""
        ...


@runtime_checkable
class Planner(Protocol):
    """Decomposes a goal into ordered sub-tasks."""

    def plan(self, goal: str) -> list[str]:
        """Return an ordered list of sub-task descriptions for ``goal``."""
        ...


__all__ = ["LLMClient", "Memory", "Planner"]
