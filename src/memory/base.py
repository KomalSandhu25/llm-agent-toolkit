"""Abstract memory interface for storing and retrieving conversation history.

The agent needs two things from memory: the ability to append new turns as a
conversation progresses, and the ability to retrieve recent (or relevant)
history to include in the next LLM call. This module defines that contract;
concrete backends (in-memory, Redis) are added on day 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


Role = Literal["user", "assistant", "tool", "system"]


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A single stored turn in a conversation's history.

    Attributes:
        role: Who produced this turn.
        content: The turn's text content.
        session_id: Identifier grouping entries into a single conversation.
        timestamp: UTC time the entry was recorded.
    """

    role: Role
    content: str
    session_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Memory(ABC):
    """Abstract interface for a conversation memory backend.

    Implementations decide how entries are persisted (in-process list, Redis
    list, a database, ...) and how retrieval windowing works, but must all
    honor this contract so the agent loop can be backend-agnostic.

    Example:
        >>> class ListMemory(Memory):
        ...     def __init__(self) -> None:
        ...         self._entries: list[MemoryEntry] = []
        ...     def add(self, entry: MemoryEntry) -> None:
        ...         self._entries.append(entry)
        ...     def get_history(self, session_id: str, limit: int = 50) -> list[MemoryEntry]:
        ...         matches = [e for e in self._entries if e.session_id == session_id]
        ...         return matches[-limit:]
        ...     def clear(self, session_id: str) -> None:
        ...         self._entries = [e for e in self._entries if e.session_id != session_id]
        >>> mem = ListMemory()
        >>> mem.add(MemoryEntry(role="user", content="hi", session_id="s1"))
        >>> len(mem.get_history("s1"))
        1
    """

    @abstractmethod
    def add(self, entry: MemoryEntry) -> None:
        """Persist a new memory entry.

        Args:
            entry: The conversation turn to store.
        """
        raise NotImplementedError

    @abstractmethod
    def get_history(self, session_id: str, limit: int = 50) -> list[MemoryEntry]:
        """Retrieve the most recent entries for a session, oldest first.

        Args:
            session_id: The conversation to fetch history for.
            limit: Maximum number of entries to return.

        Returns:
            Up to ``limit`` most recent ``MemoryEntry`` objects for the given
            session, ordered chronologically.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all stored entries for a session.

        Args:
            session_id: The conversation to clear.
        """
        raise NotImplementedError
