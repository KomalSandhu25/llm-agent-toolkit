"""Tool abstraction and registry used by the ReAct agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolError(Exception):
    """Raised when a tool cannot complete; message is shown to the agent."""


class Tool(ABC):
    """Base class for all agent tools.

    Subclasses set ``name`` and ``description`` and implement ``run``.

    Example:
        >>> class Echo(Tool):
        ...     name = "echo"
        ...     description = "Return the input text."
        ...     def run(self, **kwargs: Any) -> str:
        ...         return str(kwargs.get("text", ""))
        >>> Echo().run(text="hi")
        'hi'
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            A string observation for the agent.

        Raises:
            ToolError: If the call fails in a way the agent can recover from.
        """

    def describe(self) -> str:
        """Return a one-line description for inclusion in the prompt."""
        return f"{self.name}: {self.description}"


class ToolRegistry:
    """Name-indexed collection of tools.

    Example:
        >>> reg = ToolRegistry()
        >>> len(reg)
        0
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool; raises ValueError on missing or duplicate name."""
        if not tool.name:
            raise ValueError("Tool must define a non-empty name")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Look up a tool by name; raises ToolError if unknown."""
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(
                f"Unknown tool '{name}'. Available: {', '.join(sorted(self._tools))}"
            ) from None

    def call(self, name: str, **kwargs: Any) -> str:
        """Run a tool by name, converting unexpected errors into ToolError."""
        tool = self.get(name)
        try:
            return tool.run(**kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent
            raise ToolError(f"{name} failed: {exc}") from exc

    def describe_all(self) -> str:
        """Return a newline-joined description of every registered tool."""
        return "\n".join(t.describe() for t in self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
