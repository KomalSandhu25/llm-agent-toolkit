"""Tests for agent_toolkit.interfaces and exceptions."""

from typing import Any, Optional, Sequence

from agent_toolkit.exceptions import (
    AgentToolkitError,
    MaxStepsExceededError,
    ToolNotFoundError,
)
from agent_toolkit.interfaces import BaseMemory, BaseTool, LLMClient
from agent_toolkit.schemas import Message


class EchoTool(BaseTool):
    name = "echo"
    description = "Return the text unchanged."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, **kwargs: Any) -> str:
        return str(kwargs["text"])


class ExplodingTool(BaseTool):
    name = "boom"
    description = "Always fails."

    def run(self, **kwargs: Any) -> str:
        raise RuntimeError("kaboom")


class ListMemory(BaseMemory):
    def __init__(self) -> None:
        self._items: list[Message] = []

    def add(self, message: Message) -> None:
        self._items.append(message)

    def get(self, limit: Optional[int] = None) -> list[Message]:
        return list(self._items if limit is None else self._items[-limit:])

    def clear(self) -> None:
        self._items.clear()


class FakeLLM:
    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> Message:
        return Message.assistant("done")


def test_tool_execute_success() -> None:
    result = EchoTool().execute("c1", {"text": "hi"})
    assert result.output == "hi"
    assert not result.is_error
    assert result.call_id == "c1"


def test_tool_missing_argument_is_reported_not_raised() -> None:
    result = EchoTool().execute("c1", {})
    assert result.is_error
    assert "missing required" in result.output


def test_tool_unexpected_argument_is_reported() -> None:
    result = EchoTool().execute("c1", {"text": "a", "extra": 1})
    assert result.is_error
    assert "unexpected" in result.output


def test_tool_runtime_exception_is_captured() -> None:
    result = ExplodingTool().execute("c2", {})
    assert result.is_error
    assert "RuntimeError" in result.output and "kaboom" in result.output


def test_tool_spec_shape() -> None:
    spec = EchoTool().spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "echo"
    assert spec["function"]["parameters"]["required"] == ["text"]


def test_memory_helpers() -> None:
    memory = ListMemory()
    memory.extend([Message.user("a"), Message.user("b"), Message.user("c")])
    assert len(memory) == 3
    assert [m.content for m in memory.get(limit=2)] == ["b", "c"]
    memory.clear()
    assert len(memory) == 0


def test_llm_protocol_is_structural() -> None:
    assert isinstance(FakeLLM(), LLMClient)
    assert not isinstance(object(), LLMClient)


def test_exception_hierarchy_and_messages() -> None:
    err = ToolNotFoundError("nope", ["b", "a"])
    assert isinstance(err, AgentToolkitError)
    assert "a, b" in str(err)
    assert "5 steps" in str(MaxStepsExceededError(5))
