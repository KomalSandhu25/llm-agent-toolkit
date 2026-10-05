"""Unit tests for the tool base class and registry."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agent_toolkit import (
    Message,
    Role,
    Settings,
    Tool,
    ToolCall,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolRegistry,
)
from agent_toolkit.interfaces import LLMClient


class _AddArgs(BaseModel):
    a: int
    b: int


class AddTool(Tool):
    name = "add"
    description = "Add two integers."
    args_schema = _AddArgs

    def run(self, **kwargs: Any) -> str:
        return str(kwargs["a"] + kwargs["b"])


class BoomTool(Tool):
    name = "boom"
    description = "Always fails."
    args_schema = _AddArgs

    def run(self, **kwargs: Any) -> str:
        raise RuntimeError("kaboom")


def test_register_and_lookup() -> None:
    registry = ToolRegistry([AddTool()])
    assert "add" in registry
    assert registry.names() == ["add"]
    assert isinstance(registry.get("add"), AddTool)


def test_duplicate_registration_rejected() -> None:
    registry = ToolRegistry([AddTool()])
    with pytest.raises(ToolRegistrationError):
        registry.register(AddTool())


def test_unknown_tool_raises() -> None:
    with pytest.raises(ToolNotFoundError):
        ToolRegistry().get("nope")


def test_execute_success() -> None:
    call = ToolCall(id="1", name="add", arguments={"a": 2, "b": 3})
    result = ToolRegistry([AddTool()]).execute(call)
    assert result.output == "5"
    assert not result.is_error


def test_execute_validation_error_is_captured() -> None:
    call = ToolCall(id="1", name="add", arguments={"a": "x"})
    result = ToolRegistry([AddTool()]).execute(call)
    assert result.is_error
    assert "Invalid arguments" in result.output


def test_execute_tool_failure_is_captured() -> None:
    call = ToolCall(id="1", name="boom", arguments={"a": 1, "b": 2})
    result = ToolRegistry([BoomTool()]).execute(call)
    assert result.is_error
    assert "kaboom" in result.output


def test_execute_unknown_tool_is_captured() -> None:
    result = ToolRegistry().execute(ToolCall(id="1", name="ghost"))
    assert result.is_error
    assert "Unknown tool" in result.output


def test_openai_schema_shape() -> None:
    schema = AddTool().to_openai_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "add"
    assert set(fn["parameters"]["properties"]) == {"a", "b"}


def test_settings_defaults_and_overrides() -> None:
    s = Settings(_env_file=None, agent_max_steps=3)  # type: ignore[call-arg]
    assert s.agent_max_steps == 3
    assert s.agent_model == "gpt-4o-mini"


def test_llm_client_protocol_is_structural() -> None:
    class Fake:
        def complete(self, messages: Any, tools: Any = None) -> Message:
            return Message(role=Role.ASSISTANT, content="ok")

    assert isinstance(Fake(), LLMClient)
