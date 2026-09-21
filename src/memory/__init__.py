"""Agent memory subsystem.

Defines the abstract ``Memory`` interface (``base.py``). Concrete backends --
an in-process buffer for tests/local runs and a Redis-backed store for
persistent, multi-session conversations -- are added on day 3.
"""

from src.memory.base import Memory, MemoryEntry

__all__ = ["Memory", "MemoryEntry"]
