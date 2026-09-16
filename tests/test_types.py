"""Tests for llm_agent_toolkit.core.types."""

from __future__ import annotations

from llm_agent_toolkit.core.types import (
    AgentStatus,
    AgentStep,
    Message,
    Role,
    ToolCall,
    ToolResult,
)


def test_message_defaults_have_no_tool_linkage() -> None:
    """A plain message should have no name or tool_call_id by default."""
    message = Message(role=Role.USER, content="hello")

    assert message.role is Role.USER
    assert message.content == "hello"
    assert message.name is None
    assert message.tool_call_id is None
    assert message.created_at is not None


def test_tool_call_generates_unique_ids() -> None:
    """Each ToolCall should get a distinct auto-generated id by default."""
    first = ToolCall(tool_name="calculator", arguments={"expression": "1 + 1"})
    second = ToolCall(tool_name="calculator", arguments={"expression": "2 + 2"})

    assert first.id != second.id
    assert first.tool_name == "calculator"
    assert first.arguments == {"expression": "1 + 1"}


def test_tool_result_defaults_to_success() -> None:
    """ToolResult.is_error should default to False."""
    result = ToolResult(tool_call_id="abc", tool_name="calculator", content="2")
    assert result.is_error is False


def test_tool_result_can_represent_an_error() -> None:
    """ToolResult should be able to represent a failed execution."""
    result = ToolResult(
        tool_call_id="abc",
        tool_name="calculator",
        content="division by zero",
        is_error=True,
    )
    assert result.is_error is True


def test_agent_step_action_and_observation_default_to_none() -> None:
    """A step with only a thought should have no action or observation."""
    step = AgentStep(index=0, thought="I should compute 2 + 2.")
    assert step.action is None
    assert step.observation is None


def test_agent_step_can_carry_a_full_cycle() -> None:
    """A step can carry a thought, an action, and its observation together."""
    call = ToolCall(tool_name="calculator", arguments={"expression": "2 + 2"})
    result = ToolResult(tool_call_id=call.id, tool_name="calculator", content="4")
    step = AgentStep(index=1, thought="Let me add those numbers.", action=call, observation=result)

    assert step.action is not None and step.action.tool_name == "calculator"
    assert step.observation is not None and step.observation.content == "4"


def test_agent_status_values() -> None:
    """AgentStatus should expose the four expected terminal/running states."""
    assert {s.value for s in AgentStatus} == {
        "running",
        "finished",
        "error",
        "max_iterations_exceeded",
    }
