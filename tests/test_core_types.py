"""Tests for llm_agent_toolkit.core.types."""

from __future__ import annotations

from llm_agent_toolkit.core.types import (
    AgentResponse,
    AgentStep,
    StepType,
    ToolResult,
    ToolSpec,
)


def test_tool_spec_round_trips_through_json() -> None:
    spec = ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
    )
    restored = ToolSpec.model_validate_json(spec.model_dump_json())
    assert restored == spec


def test_tool_result_defaults() -> None:
    result = ToolResult(tool_name="calculator", success=True, output="4")
    assert result.error is None
    assert result.success is True


def test_agent_step_generates_unique_ids() -> None:
    step_a = AgentStep(step_type=StepType.THOUGHT, content="thinking")
    step_b = AgentStep(step_type=StepType.THOUGHT, content="thinking")
    assert step_a.step_id != step_b.step_id
    assert step_a.created_at > 0


def test_agent_step_carries_tool_result_for_observation() -> None:
    tool_result = ToolResult(tool_name="calculator", success=True, output="4")
    step = AgentStep(
        step_type=StepType.OBSERVATION,
        content="calculator returned 4",
        tool_name="calculator",
        tool_result=tool_result,
    )
    assert step.tool_result is not None
    assert step.tool_result.output == "4"


def test_agent_response_defaults_to_empty_steps() -> None:
    response = AgentResponse(query="2+2?", final_answer="4", iterations_used=1)
    assert response.steps == []
    assert response.succeeded is True


def test_step_type_values_are_stable_strings() -> None:
    assert StepType.THOUGHT.value == "thought"
    assert StepType.ACTION.value == "action"
    assert StepType.OBSERVATION.value == "observation"
    assert StepType.FINAL_ANSWER.value == "final_answer"
