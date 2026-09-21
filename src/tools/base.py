"""Abstract base interface that every agent tool must implement.

Concrete tools (calculator, web search, file I/O, ...) are added on day 2
along with a ``ToolRegistry`` for name-based dispatch. This module defines
only the contract: a name, a natural-language description the LLM uses to
decide when to call the tool, a JSON-schema-like argument spec, and a
``run`` method that performs the action.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of running a tool, before it is wrapped into an ``Observation``.

    Attributes:
        output: Human-readable result text to feed back to the LLM.
        success: Whether the tool executed without error.
        error: Error message when ``success`` is ``False``.
        metadata: Optional structured data about the call (e.g. raw API
            response), not shown to the LLM but useful for logging/debugging.
    """

    output: str
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] | None = None


class Tool(ABC):
    """Abstract interface all agent tools must implement.

    Subclasses declare a stable ``name`` (used by the LLM to select the tool
    and by the ``ToolRegistry`` for dispatch), a ``description`` (shown to the
    LLM verbatim, so it should clearly state what the tool does and when to
    use it), and a JSON-schema ``parameters`` dict describing expected
    arguments -- following the same shape OpenAI function-calling expects.

    Example:
        >>> class Echo(Tool):
        ...     name = "echo"
        ...     description = "Echo back the given text."
        ...     parameters = {
        ...         "type": "object",
        ...         "properties": {"text": {"type": "string"}},
        ...         "required": ["text"],
        ...     }
        ...     def _run(self, text: str) -> str:
        ...         return text
        >>> Echo().run(text="hi").output
        'hi'
    """

    name: str
    description: str
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Guard against tool subclasses forgetting required class attributes."""
        super().__init_subclass__(**kwargs)
        for attr in ("name", "description"):
            if not getattr(cls, attr, None):
                raise TypeError(f"{cls.__name__} must define a non-empty '{attr}' attribute")

    @abstractmethod
    def _run(self, **kwargs: Any) -> str:
        """Execute the tool's core logic.

        Subclasses implement this rather than ``run`` directly; ``run`` wraps
        it with timing and error handling shared by every tool.

        Args:
            **kwargs: Arguments matching this tool's ``parameters`` schema.

        Returns:
            Human-readable result text.
        """
        raise NotImplementedError

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, capturing errors and timing uniformly.

        Args:
            **kwargs: Arguments matching this tool's ``parameters`` schema.

        Returns:
            A ``ToolResult`` describing the outcome. Exceptions raised by
            ``_run`` are caught and converted into a failed ``ToolResult``
            rather than propagating, so a single bad tool call cannot crash
            the agent loop.

        Example:
            >>> class Boom(Tool):
            ...     name = "boom"
            ...     description = "Always fails."
            ...     def _run(self, **kwargs: object) -> str:
            ...         raise RuntimeError("nope")
            >>> Boom().run().success
            False
        """
        start = time.perf_counter()
        try:
            output = self._run(**kwargs)
            return ToolResult(output=output, success=True)
        except Exception as exc:  # noqa: BLE001 - intentionally broad, isolates tool failures
            return ToolResult(output="", success=False, error=str(exc))
        finally:
            self._last_latency_ms = (time.perf_counter() - start) * 1000

    def to_openai_schema(self) -> dict[str, Any]:
        """Render this tool as an OpenAI-style function-calling schema.

        Returns:
            A dict in the ``{"type": "function", "function": {...}}`` shape
            expected by the OpenAI chat completions tool-calling API.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
