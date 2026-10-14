"""Tests for the tool base class, registry, types and settings."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agent_toolkit.config import Settings
from agent_toolkit.tools.base import Tool, ToolResult
from agent_toolkit.tools.registry import ToolRegistry
from agent_toolkit.types import AgentRun, Step


class _Args(BaseModel):
    a: int
    b: int


class _Add(Tool):
    name = "add"
    description = "Add two integers."
    args_schema = _Args

    def run(self, args: Any) -> ToolResult:
        return ToolResult(output=str(args.a + args.b))


class _Boom(Tool):
    name = "boom"
    description = "Always fails."
    args_schema = _Args

    def run(self, args: Any) -> ToolResult:
        raise RuntimeError("nope")


def test_invoke_valid() -> None:
    assert _Add().invoke({"a": 2, "b": 3}).output == "5"


def test_invoke_invalid_args_returns_error() -> None:
    result = _Add().invoke({"a": "x"})
    assert result.is_error and "Invalid arguments" in result.output


def test_invoke_exception_is_contained() -> None:
    result = _Boom().invoke({"a": 1, "b": 2})
    assert result.is_error and "nope" in result.output


def test_registry_register_and_call() -> None:
    reg = ToolRegistry()
    reg.register(_Add())
    assert reg.names() == ["add"]
    assert reg.call("add", {"a": 1, "b": 1}).output == "2"
    assert len(reg) == 1


def test_registry_duplicate_rejected() -> None:
    reg = ToolRegistry()
    reg.register(_Add())
    with pytest.raises(ValueError):
        reg.register(_Add())


def test_registry_unknown_tool() -> None:
    result = ToolRegistry().call("ghost", {})
    assert result.is_error and "Unknown tool" in result.output


def test_spec_contains_schema() -> None:
    spec = _Add().spec()
    assert spec["name"] == "add" and "properties" in spec["parameters"]


def test_agent_run_answer() -> None:
    run = AgentRun(question="q", steps=[Step(thought="t", final_answer="done")])
    assert run.answer == "done" and run.steps[-1].is_final


def test_settings_defaults() -> None:
    assert Settings(openai_api_key="k").agent_max_steps == 8
