"""Tests for agent_toolkit.types and exceptions."""

from agent_toolkit import (
    AgentResult,
    AgentStep,
    Message,
    Role,
    ToolCall,
    ToolExecutionError,
    ToolNotFoundError,
    ToolResult,
)


def test_message_to_dict() -> None:
    assert Message(role=Role.USER, content="hi").to_dict() == {"role": "user", "content": "hi"}
    assert Message(role=Role.TOOL, content="4", name="calc").to_dict()["name"] == "calc"


def test_agent_result_counts_steps() -> None:
    step = AgentStep(
        index=0,
        thought="need math",
        action=ToolCall(name="calculator", arguments={"expression": "2+2"}),
        observation=ToolResult(name="calculator", output="4"),
    )
    result = AgentResult(answer="4", steps=[step])
    assert result.num_steps == 1
    assert result.steps[0].observation is not None
    assert not result.steps[0].observation.is_error


def test_tool_errors_carry_context() -> None:
    assert ToolNotFoundError("x").name == "x"
    err = ToolExecutionError("calc", "boom")
    assert err.reason == "boom" and "calc" in str(err)
