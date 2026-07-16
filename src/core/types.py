"""Core data types shared across the agent toolkit.

These types model the vocabulary of the ReAct loop: conversational
:class:`Message` objects, a single reasoning :class:`AgentStep`
(thought / action / observation / final answer), tool invocation via
:class:`ToolCall` and :class:`ToolResult`, and the accumulated
:class:`AgentState` for one task run.

All types are Pydantic models so they get free validation, JSON
(de)serialization (useful for persisting to Redis in a later module, and
for streaming over the WebSocket API), and clear error messages if a
downstream module constructs one incorrectly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return the current UTC time.

    Returns:
        A timezone-aware :class:`datetime` in UTC.
    """
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """Generate a short, unique identifier.

    Returns:
        A hex UUID4 string suitable for correlating tool calls and results.
    """
    return uuid.uuid4().hex


class Role(str, Enum):
    """Speaker role for a :class:`Message` in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StepType(str, Enum):
    """The kind of reasoning step the agent is currently performing."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


class Message(BaseModel):
    """A single turn in the conversation between user, assistant, and tools.

    Attributes:
        role: Who is speaking in this turn.
        content: The text content of the message.
        name: Optional tool or participant name (used for ``Role.TOOL``
            messages so the model can attribute the observation).
        tool_call_id: When this message is a tool response, the
            :attr:`ToolCall.call_id` it answers.
        timestamp: When the message was created.

    Example:
        >>> msg = Message.user("What's the weather in Paris?")
        >>> msg.role
        <Role.USER: 'user'>
    """

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)

    @classmethod
    def system(cls, content: str) -> "Message":
        """Construct a system message.

        Args:
            content: The system prompt text.

        Returns:
            A new :class:`Message` with ``role=Role.SYSTEM``.
        """
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Construct a user message.

        Args:
            content: The user's input text.

        Returns:
            A new :class:`Message` with ``role=Role.USER``.
        """
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Construct an assistant message.

        Args:
            content: The assistant's response text.

        Returns:
            A new :class:`Message` with ``role=Role.ASSISTANT``.
        """
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool(cls, content: str, *, name: str, tool_call_id: str) -> "Message":
        """Construct a tool-result message.

        Args:
            content: The tool's output, rendered as text.
            name: The name of the tool that produced this result.
            tool_call_id: The id of the :class:`ToolCall` this answers.

        Returns:
            A new :class:`Message` with ``role=Role.TOOL``.
        """
        return cls(role=Role.TOOL, content=content, name=name, tool_call_id=tool_call_id)


class ToolCall(BaseModel):
    """A request from the agent to invoke a specific tool.

    Attributes:
        call_id: Unique identifier correlating this call with its
            :class:`ToolResult`.
        tool_name: The registered name of the tool to invoke.
        tool_input: Keyword arguments to pass to the tool.

    Example:
        >>> call = ToolCall(tool_name="calculator", tool_input={"expression": "2 + 2"})
        >>> call.tool_name
        'calculator'
    """

    call_id: str = Field(default_factory=_new_id)
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing a :class:`ToolCall`.

    Attributes:
        call_id: The :attr:`ToolCall.call_id` this result answers.
        tool_name: The name of the tool that produced this result.
        output: The tool's output, rendered as text for the model to read.
        success: Whether the tool executed without raising an error.
        error: The error message, if ``success`` is ``False``.
        latency_ms: Wall-clock execution time in milliseconds.
    """

    call_id: str
    tool_name: str
    output: str
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0


class AgentStep(BaseModel):
    """A single step in the agent's thought/action/observation loop.

    Attributes:
        step_type: Which phase of the ReAct loop this step represents.
        content: Free-text content of the step (the thought text, or the
            final answer text).
        tool_call: Present when ``step_type == StepType.ACTION``.
        tool_result: Present when ``step_type == StepType.OBSERVATION``.
        iteration: Which loop iteration (1-indexed) this step belongs to.
    """

    step_type: StepType
    content: str
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    iteration: int = Field(ge=1)


class AgentState(BaseModel):
    """The accumulated state of a single agent run on one task.

    Attributes:
        task: The original task/question the agent was asked to solve.
        messages: The full conversation transcript, including tool
            exchanges, in the format the LLM API expects.
        steps: The structured trace of thoughts/actions/observations,
            useful for debugging, evaluation, and streaming to a UI.
        iteration: The current (1-indexed) loop iteration.
        is_complete: Whether the agent has produced a final answer.
        final_answer: The agent's answer once ``is_complete`` is ``True``.

    Example:
        >>> state = AgentState(task="What is 2 + 2?")
        >>> state.add_step(
        ...     AgentStep(step_type=StepType.THOUGHT, content="I should add.", iteration=1)
        ... )
        >>> len(state.steps)
        1
    """

    task: str
    messages: list[Message] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    iteration: int = 0
    is_complete: bool = False
    final_answer: str | None = None

    def add_step(self, step: AgentStep) -> None:
        """Append a step to the trace and advance the iteration counter.

        Args:
            step: The step to record. If ``step.step_type`` is
                :attr:`StepType.FINAL_ANSWER`, the state is marked complete
                and :attr:`final_answer` is set to ``step.content``.
        """
        self.steps.append(step)
        self.iteration = max(self.iteration, step.iteration)
        if step.step_type is StepType.FINAL_ANSWER:
            self.is_complete = True
            self.final_answer = step.content
