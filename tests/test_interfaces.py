"""Tests for llm_agent_toolkit.core.interfaces.

These tests exercise the abstract contracts using minimal in-memory
implementations, standing in for the real tool/memory/LLM-client/agent
implementations planned for later days. This both proves the interfaces
are usable and pins down the calling convention future implementations
must follow.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_agent_toolkit.core.interfaces import (
    BaseAgent,
    BaseLLMClient,
    BaseMemory,
    BaseTool,
)
from llm_agent_toolkit.core.types import AgentResponse, ToolResult, ToolSpec


class _EchoTool(BaseTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="echo",
            description="Echo back the given text.",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name="echo", success=True, output=str(kwargs.get("text", "")))


class _InMemoryMemory(BaseMemory):
    def __init__(self) -> None:
        self._turns: list[dict[str, str]] = []

    async def add(self, role: str, content: str) -> None:
        self._turns.append({"role": role, "content": content})

    async def get_history(self, *, limit: int | None = None) -> list[dict[str, str]]:
        return self._turns if limit is None else self._turns[-limit:]

    async def clear(self) -> None:
        self._turns.clear()


class _StaticLLMClient(BaseLLMClient):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> str:
        return "static response"


class _OneShotAgent(BaseAgent):
    """Minimal agent: answers immediately without a real ReAct loop."""

    def __init__(self, llm: BaseLLMClient) -> None:
        self._llm = llm

    async def run(self, query: str) -> AgentResponse:
        answer = await self._llm.complete([{"role": "user", "content": query}])
        return AgentResponse(query=query, final_answer=answer, iterations_used=1)


@pytest.mark.parametrize("abstract_cls", [BaseTool, BaseMemory, BaseLLMClient, BaseAgent])
def test_abstract_base_classes_cannot_be_instantiated(abstract_cls: type) -> None:
    with pytest.raises(TypeError):
        abstract_cls()  # type: ignore[abstract]


async def test_echo_tool_round_trips_input() -> None:
    tool = _EchoTool()
    result = await tool.execute(text="hello")
    assert result.success is True
    assert result.output == "hello"
    assert tool.spec.name == "echo"


async def test_in_memory_memory_add_and_get_history() -> None:
    memory = _InMemoryMemory()
    await memory.add("user", "hi")
    await memory.add("assistant", "hello there")
    history = await memory.get_history()
    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]


async def test_in_memory_memory_respects_limit() -> None:
    memory = _InMemoryMemory()
    for i in range(5):
        await memory.add("user", f"turn {i}")
    history = await memory.get_history(limit=2)
    assert history == [{"role": "user", "content": "turn 3"}, {"role": "user", "content": "turn 4"}]


async def test_in_memory_memory_clear() -> None:
    memory = _InMemoryMemory()
    await memory.add("user", "hi")
    await memory.clear()
    assert await memory.get_history() == []


async def test_one_shot_agent_uses_llm_client() -> None:
    agent = _OneShotAgent(_StaticLLMClient())
    response = await agent.run("What is 2 + 2?")
    assert response.final_answer == "static response"
    assert response.succeeded is True
    assert response.iterations_used == 1
