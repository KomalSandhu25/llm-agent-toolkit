"""Core extension points of the toolkit.

* :class:`LLMClient` - structural protocol for any chat-completion backend.
* :class:`BaseTool` - abstract base class for tools the agent can call.
* :class:`BaseMemory` - abstract base class for conversation/episodic memory.

Concrete implementations arrive in later modules; code in this module only defines
the contracts and shared behaviour such as argument validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from agent_toolkit.exceptions import ToolExecutionError
from agent_toolkit.schemas import Message, ToolResult


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn a conversation into the next assistant message."""

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Message:
        """Generate the next assistant message.

        Args:
            messages: Conversation so far, oldest first.
            tools: Optional tool specs from :meth:`BaseTool.spec`.
            temperature: Sampling temperature.

        Returns:
            The assistant reply, possibly containing tool calls.
        """
        ...


class BaseTool(ABC):
    """Base class for tools.

    Subclasses set ``name``, ``description`` and ``parameters`` (a JSON-schema
    ``object``) and implement :meth:`run`. Callers use :meth:`execute`, which validates
    arguments and converts failures into :class:`ToolResult` objects so a bad tool call
    never crashes the agent loop.

    Example:
        >>> class Echo(BaseTool):
        ...     name = "echo"
        ...     description = "Return the text unchanged."
        ...     parameters = {
        ...         "type": "object",
        ...         "properties": {"text": {"type": "string"}},
        ...         "required": ["text"],
        ...     }
        ...     def run(self, **kwargs: Any) -> str:
        ...         return str(kwargs["text"])
        >>> Echo().execute("c1", {"text": "hi"}).output
        'hi'
    """

    name: str
    description: str
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Perform the tool's work.

        Args:
            **kwargs: Arguments matching :attr:`parameters`.

        Returns:
            Text observation for the model.
        """

    def spec(self) -> dict[str, Any]:
        """Return an OpenAI-style function spec describing this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Check required and unexpected arguments against :attr:`parameters`.

        Args:
            arguments: Arguments supplied by the model.

        Raises:
            ToolExecutionError: If a required argument is missing or an unknown one
                is supplied.
        """
        properties = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ToolExecutionError(self.name, f"missing required argument(s): {', '.join(missing)}")
        unknown = [key for key in arguments if key not in properties]
        if unknown:
            raise ToolExecutionError(self.name, f"unexpected argument(s): {', '.join(sorted(unknown))}")

    def execute(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Validate arguments and run the tool, never raising.

        Args:
            call_id: Identifier of the originating tool call.
            arguments: Arguments supplied by the model.

        Returns:
            A :class:`ToolResult`; ``is_error`` is ``True`` if validation or execution
            failed.
        """
        try:
            self.validate_arguments(arguments)
            output = self.run(**arguments)
        except ToolExecutionError as exc:
            return ToolResult(call_id=call_id, name=self.name, output=str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 - tools are untrusted; surface as observation
            reason = f"{type(exc).__name__}: {exc}"
            return ToolResult(
                call_id=call_id,
                name=self.name,
                output=str(ToolExecutionError(self.name, reason)),
                is_error=True,
            )
        return ToolResult(call_id=call_id, name=self.name, output=output)


class BaseMemory(ABC):
    """Abstract store for conversation history."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Append a message to memory."""

    @abstractmethod
    def get(self, limit: Optional[int] = None) -> list[Message]:
        """Return stored messages, oldest first.

        Args:
            limit: If given, return only the most recent ``limit`` messages.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored messages."""

    def extend(self, messages: Sequence[Message]) -> None:
        """Append several messages in order."""
        for message in messages:
            self.add(message)

    def __len__(self) -> int:
        return len(self.get())
