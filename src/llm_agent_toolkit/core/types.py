"""Core data shapes shared across every layer of llm_agent_toolkit.

These are intentionally "dumb" models: no behavior, just validated shape.
Behavior lives on the concrete implementations of the interfaces defined in
:mod:`llm_agent_toolkit.core.interfaces`.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """The kind of step produced during one iteration of the ReAct loop."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class ToolSpec(BaseModel):
    """Describes a tool's calling contract, independent of its implementation.

    This is what gets serialized into the function-calling schema sent to
    the LLM client, and is exposed by every
    :class:`~llm_agent_toolkit.core.interfaces.BaseTool` via its ``spec``
    property.

    Attributes:
        name: Unique, LLM-facing identifier for the tool (e.g. ``"calculator"``).
        description: Natural-language description the LLM uses to decide
            when to invoke the tool.
        parameters: A JSON-Schema-style ``dict`` describing the tool's
            expected keyword arguments.

    Example:
        >>> spec = ToolSpec(
        ...     name="calculator",
        ...     description="Evaluate a basic arithmetic expression.",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {"expression": {"type": "string"}},
        ...         "required": ["expression"],
        ...     },
        ... )
        >>> spec.name
        'calculator'
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing a tool once.

    Attributes:
        tool_name: Name of the tool that produced this result.
        success: Whether the tool call completed without error.
        output: The tool's return value, as a string ready to be fed back
            into the LLM as an OBSERVATION. Empty when ``success`` is
            ``False``.
        error: Error message when ``success`` is ``False``; ``None``
            otherwise.

    Example:
        >>> result = ToolResult(tool_name="calculator", success=True, output="4")
        >>> result.success
        True
    """

    tool_name: str
    success: bool
    output: str = ""
    error: str | None = None


class AgentStep(BaseModel):
    """A single THOUGHT, ACTION, OBSERVATION, or FINAL_ANSWER step.

    A full agent run (see :class:`AgentResponse`) is an ordered list of
    these, which is what makes the run auditable and replayable: nothing
    about *how* the agent reached its answer is discarded.

    Attributes:
        step_id: Unique identifier for this step, auto-generated.
        step_type: Which phase of the ReAct loop produced this step.
        content: Free-text content -- the reasoning text for THOUGHT/
            FINAL_ANSWER steps, or a human-readable description for ACTION/
            OBSERVATION steps.
        tool_name: The tool selected, if ``step_type`` is ``ACTION``.
        tool_arguments: The arguments passed to the tool, if ``step_type``
            is ``ACTION``.
        tool_result: The result of the tool call, if ``step_type`` is
            ``OBSERVATION``.
        created_at: Unix timestamp (seconds) when the step was created.

    Example:
        >>> step = AgentStep(step_type=StepType.THOUGHT, content="I should check the weather.")
        >>> step.step_type
        <StepType.THOUGHT: 'thought'>
    """

    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    step_type: StepType
    content: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result: ToolResult | None = None
    created_at: float = Field(default_factory=time.time)


class AgentResponse(BaseModel):
    """The terminal result of a full agent run.

    Attributes:
        query: The original user query that started the run.
        final_answer: The agent's concluding answer.
        steps: The complete, ordered THOUGHT/ACTION/OBSERVATION history that
            led to ``final_answer``.
        iterations_used: How many THOUGHT/ACTION/OBSERVATION cycles ran
            before a final answer was reached.
        succeeded: Whether the run ended in a final answer rather than an
            error or an iteration-limit cutoff.

    Example:
        >>> response = AgentResponse(
        ...     query="What is 2 + 2?",
        ...     final_answer="4",
        ...     steps=[],
        ...     iterations_used=1,
        ...     succeeded=True,
        ... )
        >>> response.succeeded
        True
    """

    query: str
    final_answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    iterations_used: int = 0
    succeeded: bool = True
