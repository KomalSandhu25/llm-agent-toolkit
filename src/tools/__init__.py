"""The pluggable tool interface.

Concrete tools (calculator, search, file I/O) and the tool registry are
built out on Day 2. This module defines the contract every tool must
satisfy so the agent core (Day 4) can be implemented against a stable
interface regardless of which concrete tools exist yet.

A "tool" is anything the agent can invoke by name with keyword arguments
and that returns a text observation. Tools describe themselves via
:meth:`BaseTool.to_openai_schema`, which produces the JSON-schema function
description the OpenAI function-calling / tool-calling API expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class every agent tool must implement.

    Attributes:
        name: The unique, stable identifier the agent uses to invoke this
            tool (also the function name sent to the OpenAI API).
        description: A natural-language description shown to the model so
            it can decide when this tool is relevant.
        parameters_schema: A JSON Schema ``object`` describing the keyword
            arguments accepted by :meth:`run`.

    Example:
        >>> class EchoTool(BaseTool):
        ...     name = "echo"
        ...     description = "Echoes back the given text."
        ...     parameters_schema = {
        ...         "type": "object",
        ...         "properties": {"text": {"type": "string"}},
        ...         "required": ["text"],
        ...     }
        ...     def run(self, **kwargs: Any) -> str:
        ...         return str(kwargs["text"])
        >>> EchoTool().run(text="hi")
        'hi'
    """

    name: str
    description: str
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool synchronously.

        Args:
            **kwargs: Arguments matching :attr:`parameters_schema`.

        Returns:
            A text observation to be fed back to the agent as the result
            of this tool call.

        Raises:
            Exception: Concrete tools may raise; callers (the future tool
                executor in the agent core) are expected to catch and wrap
                failures in ``ToolExecutionError``.
        """
        raise NotImplementedError

    def to_openai_schema(self) -> dict[str, Any]:
        """Render this tool as an OpenAI function-calling tool definition.

        Returns:
            A dict of the shape expected by the OpenAI ``tools`` API
            parameter: ``{"type": "function", "function": {...}}``.

        Example:
            >>> class NoOpTool(BaseTool):
            ...     name = "noop"
            ...     description = "Does nothing."
            ...     def run(self, **kwargs: Any) -> str:
            ...         return ""
            >>> schema = NoOpTool().to_openai_schema()
            >>> schema["function"]["name"]
            'noop'
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
