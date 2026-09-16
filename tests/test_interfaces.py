"""Tests for the abstract interfaces in llm_agent_toolkit.core.interfaces.

These tests exist to lock down two things that matter for the toolkit's
"swap anything" design goal:

1. The abstract base classes cannot be instantiated directly.
2. A minimal concrete implementation of each interface *can* be
   instantiated and satisfies the contract's method signatures.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from llm_agent_toolkit.core.interfaces import (
    BaseAgent,
    BaseLLMClient,
    BaseMemory,
    BasePlanner,
    BaseTool,
)
from llm_agent_toolkit.core.types import AgentStatus, Message, Role, ToolCall, ToolResult


class _EchoLLMClient(BaseLLMClient):
    async def generate(self, messages: list[Message], **kwargs: Any) -> Message:
        last = messages[-1].content if messages else ""
        return Message(role=Role.ASSISTANT, content=f"echo: {last}")

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        for chunk in ("echo", ": ", messages[-1].content if messages else ""):
            yield chunk


class _CalculatorTool(BaseTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluates a simple arithmetic expression."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        expression = tool_call.arguments["expression"]
        result = eval(expression, {"__builtins__": {}})  # noqa: S307 - test double only
        return ToolResult(tool_call_id=tool_call.id, tool_name=self.name, content=str(result))


class _InMemoryMemory(BaseMemory):
    def __init__(self) -> None:
        self._messages: list[Message] = []

    async def add(self, message: Message) -> None:
        self._messages.append(message)

    async def get_history(self, limit: int | None = None) -> list[Message]:
        return self._messages if limit is None else self._messages[-limit:]

    async def clear(self) -> None:
        self._messages.clear()


class _SingleStepPlanner(BasePlanner):
    async def create_plan(self, goal: str, context: list[Message] | None = None) -> list[str]:
        return [goal]


class _NoopAgent(BaseAgent):
    async def run(self, goal: str) -> AgentStatus:
        return AgentStatus.FINISHED


@pytest.mark.parametrize(
    "abstract_cls",
    [BaseLLMClient, BaseTool, BaseMemory, BasePlanner, BaseAgent],
)
def test_abstract_interfaces_cannot_be_instantiated(abstract_cls: type) -> None:
    """None of the core interfaces should be directly instantiable."""
    with pytest.raises(TypeError):
        abstract_cls()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_llm_client_contract() -> None:
    """A minimal BaseLLMClient implementation should satisfy generate/stream."""
    client = _EchoLLMClient()
    reply = await client.generate([Message(role=Role.USER, content="hi")])
    assert reply.content == "echo: hi"

    chunks = [chunk async for chunk in client.stream([Message(role=Role.USER, content="hi")])]
    assert "".join(chunks) == "echo: hi"


@pytest.mark.asyncio
async def test_tool_contract() -> None:
    """A minimal BaseTool implementation should expose metadata and execute."""
    tool = _CalculatorTool()
    assert tool.name == "calculator"
    assert "arithmetic" in tool.description
    assert tool.parameters_schema["required"] == ["expression"]

    call = ToolCall(tool_name="calculator", arguments={"expression": "2 + 2"})
    result = await tool.execute(call)
    assert result.content == "4"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_memory_contract() -> None:
    """A minimal BaseMemory implementation should add, list, and clear."""
    memory = _InMemoryMemory()
    await memory.add(Message(role=Role.USER, content="first"))
    await memory.add(Message(role=Role.USER, content="second"))

    history = await memory.get_history()
    assert [m.content for m in history] == ["first", "second"]

    limited = await memory.get_history(limit=1)
    assert [m.content for m in limited] == ["second"]

    await memory.clear()
    assert await memory.get_history() == []


@pytest.mark.asyncio
async def test_planner_contract() -> None:
    """A minimal BasePlanner implementation should return an ordered plan."""
    planner = _SingleStepPlanner()
    plan = await planner.create_plan("summarize this repository")
    assert plan == ["summarize this repository"]


@pytest.mark.asyncio
async def test_agent_contract() -> None:
    """A minimal BaseAgent implementation should return a terminal status."""
    agent = _NoopAgent()
    status = await agent.run("do nothing")
    assert status is AgentStatus.FINISHED
