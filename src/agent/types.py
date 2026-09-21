"""Shared data types for the ReAct agent loop.

These types define the vocabulary the rest of the system is built on: a
``Step`` is one iteration of Reason -> Act -> Observe, a ``Thought`` is the
model's reasoning text for that step, a ``ToolCall`` is the action it decided
to take, an ``Observation`` is the result of executing that action, and an
``AgentResult`` is the final outcome returned to the caller once the loop
terminates (successfully, by exhausting iterations, or by error).

Keeping these as plain, well-typed dataclasses (rather than passing dicts
around) keeps the agent loop, planner, API layer, and CLI all speaking the
same language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class StepType(str, Enum):
    """The kind of action taken during a single ReAct iteration."""

    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"


class AgentStatus(str, Enum):
    """Terminal status of an agent run."""

    SUCCESS = "success"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Thought:
    """The model's reasoning text produced during a single ReAct step.

    Attributes:
        content: Free-form reasoning text explaining what the agent is
            trying to do next and why.
        iteration: The 1-indexed loop iteration this thought belongs to.
    """

    content: str
    iteration: int


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation requested by the agent.

    Attributes:
        tool_name: The registered name of the tool to invoke, matching the
            ``name`` attribute of a ``src.tools.base.Tool`` instance.
        arguments: Keyword arguments to pass to the tool's ``run`` method,
            typically produced by parsing the model's structured output.
        call_id: Optional identifier correlating this call with a specific
            provider-side tool-call message, useful for multi-tool-call turns.
    """

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """The result of executing a ``ToolCall``.

    Attributes:
        tool_name: Name of the tool that produced this observation.
        output: The tool's return value, coerced to a string for inclusion
            back into the model's context window.
        success: Whether the tool call completed without raising.
        error: Error message if ``success`` is ``False``, else ``None``.
        latency_ms: Wall-clock time the tool call took, in milliseconds.
    """

    tool_name: str
    output: str
    success: bool
    error: str | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class Step:
    """One full Reason -> Act -> Observe iteration of the agent loop.

    Attributes:
        step_type: Whether this step ended in a tool call or a final answer.
        thought: The reasoning that led to this step's action.
        tool_call: The tool invocation for this step, if ``step_type`` is
            ``StepType.TOOL_CALL``.
        observation: The result of executing ``tool_call``, if applicable.
        final_answer: The agent's answer text, if ``step_type`` is
            ``StepType.FINAL_ANSWER``.
        timestamp: UTC time the step was recorded.
    """

    step_type: StepType
    thought: Thought
    tool_call: ToolCall | None = None
    observation: Observation | None = None
    final_answer: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class AgentResult:
    """The final outcome of an agent run, returned to the caller.

    Attributes:
        status: Terminal status describing how the run ended.
        answer: The agent's final answer text, if any was produced.
        steps: The full ordered trace of ``Step`` objects taken during the run,
            useful for debugging, logging, and UI display.
        total_iterations: Number of ReAct iterations actually executed.
        error: Error message if ``status`` is ``AgentStatus.ERROR``.

    Example:
        >>> result = AgentResult(
        ...     status=AgentStatus.SUCCESS,
        ...     answer="The answer is 42.",
        ...     steps=[],
        ...     total_iterations=1,
        ... )
        >>> result.status
        <AgentStatus.SUCCESS: 'success'>
    """

    status: AgentStatus
    answer: str | None
    steps: list[Step]
    total_iterations: int
    error: str | None = None
