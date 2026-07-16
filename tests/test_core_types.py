"""Unit tests for :mod:`src.core.types`."""

from __future__ import annotations

from src.core.types import (
    AgentState,
    AgentStep,
    Message,
    Role,
    StepType,
    ToolCall,
    ToolResult,
)


def test_message_role_constructors() -> None:
    """Each Message classmethod should stamp the correct role."""
    assert Message.system("sys").role is Role.SYSTEM
    assert Message.user("hi").role is Role.USER
    assert Message.assistant("hello").role is Role.ASSISTANT
    tool_msg = Message.tool("42", name="calculator", tool_call_id="abc123")
    assert tool_msg.role is Role.TOOL
    assert tool_msg.name == "calculator"
    assert tool_msg.tool_call_id == "abc123"


def test_tool_call_generates_unique_ids() -> None:
    """Two ToolCall instances should never collide on call_id."""
    call_a = ToolCall(tool_name="search", tool_input={"query": "x"})
    call_b = ToolCall(tool_name="search", tool_input={"query": "y"})
    assert call_a.call_id != call_b.call_id


def test_tool_result_defaults() -> None:
    """ToolResult should default to a successful, zero-latency result."""
    result = ToolResult(call_id="abc", tool_name="calculator", output="4")
    assert result.success is True
    assert result.error is None
    assert result.latency_ms == 0.0


def test_agent_state_add_step_tracks_iteration() -> None:
    """add_step should advance the iteration counter monotonically."""
    state = AgentState(task="2 + 2?")
    state.add_step(AgentStep(step_type=StepType.THOUGHT, content="Add them.", iteration=1))
    state.add_step(
        AgentStep(
            step_type=StepType.ACTION,
            content="Call calculator",
            tool_call=ToolCall(tool_name="calculator", tool_input={"expression": "2+2"}),
            iteration=1,
        )
    )
    assert state.iteration == 1
    assert state.is_complete is False


def test_agent_state_final_answer_marks_complete() -> None:
    """A FINAL_ANSWER step must set is_complete and final_answer."""
    state = AgentState(task="2 + 2?")
    state.add_step(
        AgentStep(step_type=StepType.FINAL_ANSWER, content="4", iteration=2)
    )
    assert state.is_complete is True
    assert state.final_answer == "4"
    assert state.iteration == 2


def test_agent_state_serialization_round_trip() -> None:
    """AgentState should survive a model_dump / model_validate round trip."""
    state = AgentState(task="ping")
    state.messages.append(Message.user("ping"))
    payload = state.model_dump(mode="json")
    restored = AgentState.model_validate(payload)
    assert restored.task == state.task
    assert restored.messages[0].content == "ping"
