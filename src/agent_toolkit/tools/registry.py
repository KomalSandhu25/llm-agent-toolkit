"""Registry that stores tools and dispatches model tool calls."""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from pydantic import ValidationError

from agent_toolkit.exceptions import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
)
from agent_toolkit.schemas import ToolCall, ToolResult
from agent_toolkit.tools.base import Tool


class ToolRegistry:
    """Holds named tools and executes :class:`ToolCall` requests.

    Example:
        >>> registry = ToolRegistry()
        >>> len(registry)
        0
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        """Create a registry.

        Args:
            tools: Optional tools to register immediately.
        """
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register ``tool`` under its ``name``.

        Raises:
            ToolRegistrationError: If the name is already taken.
        """
        if tool.name in self._tools:
            raise ToolRegistrationError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Look up a tool by name.

        Raises:
            ToolNotFoundError: If no such tool is registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name, list(self._tools)) from None

    def names(self) -> list[str]:
        """Return registered tool names in registration order."""
        return list(self._tools)

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schemas for all tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute ``call`` and capture any failure as an error result.

        Errors are returned (not raised) so the model can observe and
        recover from them in the next ReAct step.

        Args:
            call: The tool call requested by the model.

        Returns:
            A :class:`ToolResult`; ``is_error`` is True on failure.
        """
        try:
            output = self.get(call.name).invoke(call.arguments)
            return ToolResult(call_id=call.id, name=call.name, output=output)
        except ToolNotFoundError as exc:
            return ToolResult(call_id=call.id, name=call.name, output=str(exc), is_error=True)
        except ValidationError as exc:
            msg = f"Invalid arguments for {call.name!r}: {exc.errors(include_url=False)}"
            return ToolResult(call_id=call.id, name=call.name, output=msg, is_error=True)
        except Exception as exc:  # noqa: BLE001 - tool code is untrusted
            err = ToolExecutionError(call.name, exc)
            return ToolResult(call_id=call.id, name=call.name, output=str(err), is_error=True)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())
