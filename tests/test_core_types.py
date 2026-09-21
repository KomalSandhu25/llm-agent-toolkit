"""Unit tests for src.core.types."""

from __future__ import annotations

from src.core.types import AgentState, AgentStep, Message, Role, StepType, ToolCall, ToolResult


def test_message_to_openai_format_without_name() -> None:
    """A plain user message should serialize to role/content only."""
    msg = Message(role=Role.USER, content="hello")
    assert msg.to_openai_format() == {"role": "user", "content": "hello"}


def test_message_to_openai_format_with_name() -> None:
    """A tool message should include its name in the serialized dict."""
    msg = Message(role=Role.TOOL, content="42", name="calculator")
    formatted = msg.to_openai_format()
    assert formatted["name"] == "calculator"
    assert formatted["role"] == "tool"


def test_tool_call_generates_unique_ids() -> None:
    """Each ToolCall should get a distinct auto-generated call_id."""
    call_a = ToolCall(tool_name="search", arguments={"query": "x"})
    call_b = ToolCall(tool_name="search", arguments={"query": "x"})
    assert call_a.call_id != call_b.call_id


def test_agent_state_record_increments_iteration_on_action() -> None:
    """Recording an ACTION step should advance the iteration counter."""
    state = AgentState(task="do something")
    state.record(AgentStep(step_type=StepType.THOUGHT, content="I should act"))
    assert state.iteration == 0
    state.record(
        AgentStep(
            step_type=StepType.ACTION,
            content="call calculator",
            tool_call=ToolCall(tool_name="calculator", arguments={"expr": "1+1"}),
        )
    )
    assert state.iteration == 1
    assert len(state.steps) == 2


def test_agent_state_final_answer_marks_complete() -> None:
    """Recording a FINAL_ANSWER step should set is_complete and final_answer."""
    state = AgentState(task="what is 2+2")
    state.record(AgentStep(step_type=StepType.FINAL_ANSWER, content="4"))
    assert state.is_complete is True
    assert state.final_answer == "4"


def test_tool_result_defaults_to_success() -> None:
    """A ToolResult constructed without an explicit success flag defaults to True."""
    result = ToolResult(call_id="abc123", tool_name="calculator", output="4")
    assert result.success is True
    assert result.error is None
