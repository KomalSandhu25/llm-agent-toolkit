"""Registry that maps tool names to tool instances."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agent_toolkit.tools.base import Tool, ToolResult


class ToolRegistry:
    """Holds the set of tools available to an agent.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.names()
        []
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If a tool with the same name already exists.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name.

        Args:
            name: Tool name.

        Returns:
            The tool, or ``None`` if unknown.
        """
        return self._tools.get(name)

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a tool by name, reporting unknown tools as errors.

        Args:
            name: Tool name.
            arguments: Raw arguments for the tool.

        Returns:
            The tool result, or an error result if the tool is unknown.
        """
        tool = self.get(name)
        if tool is None:
            known = ", ".join(self.names()) or "none"
            return ToolResult(output=f"Unknown tool '{name}'. Available: {known}", is_error=True)
        return tool.invoke(arguments)

    def names(self) -> list[str]:
        """Return sorted registered tool names."""
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        """Return prompt specs for every registered tool."""
        return [self._tools[n].spec() for n in self.names()]

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)
