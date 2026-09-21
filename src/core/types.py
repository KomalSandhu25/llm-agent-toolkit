"""Core data types for the ReAct agent loop.

Every other package in this project (tools, memory, agent, planning, api)
speaks in terms of the types defined here. Keeping them in one place with
no downstream dependencies avoids circular imports as the agent, tool
registry, and memory backends are built out over the coming days.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Role(str, Enum):
    """Speaker role for a single turn in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StepType(str, Enum):
    """The kind of step the agent produced during one ReAct iteration."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass(slots=True)
class Message:
    """A single conversational turn stored in memory or sent to the LLM.

    Args:
        role: Who produced this message.
        content: The text content of the message.
        name: Optional tool/function name, set when ``role`` is ``TOOL``.
        metadata: Free-form metadata (latency, token counts, etc.).
        timestamp: Unix timestamp the message was created, defaults to now.

    Example:
        >>> msg = Message(role=Role.USER, content="What's 2 + 2?")
        >>> msg.role
        <Role.USER: 'user'>
    """

    role: Role
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_openai_format(self) -> dict[str, str]:
        """Render this message as an OpenAI chat-completions message dict.

        Returns:
            A dict with ``role`` and ``content`` keys (and ``name`` if set),
            suitable for the ``messages`` list of a chat completion call.

        Example:
            >>> Message(role=Role.USER, content="hi").to_openai_format()
            {'role': 'user', 'content': 'hi'}
        """
        payload: dict[str, str] = {"role": self.role.value, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(slots=True)
class ToolCall:
    """A request, emitted by the LLM, to invoke a specific tool.

    Args:
        tool_name: Registered name of the tool to invoke.
        arguments: Keyword arguments to pass to the tool.
        call_id: Unique identifier correlating this call to its result.
    """

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(slots=True)
class ToolResult:
    """The outcome of executing a ``ToolCall``.

    Args:
        call_id: Matches the ``ToolCall.call_id`` this result answers.
        tool_name: Name of the tool that produced this result.
        output: The tool's return value, serialized to a string for the LLM.
        success: Whether the tool executed without raising.
        error: Error message when ``success`` is False, otherwise ``None``.
        latency_ms: Wall-clock execution time in milliseconds.
    """

    call_id: str
    tool_name: str
    output: str
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0


@dataclass(slots=True)
class AgentStep:
    """One recorded step of the agent's thought/action/observation loop.

    Args:
        step_type: Whether this step is a thought, action, observation, or
            the final answer.
        content: Human-readable text for this step (the thought text, or
            the final answer text).
        tool_call: Populated when ``step_type`` is ``ACTION``.
        tool_result: Populated when ``step_type`` is ``OBSERVATION``.
    """

    step_type: StepType
    content: str
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


@dataclass(slots=True)
class AgentState:
    """The full mutable state of one agent run.

    Args:
        task: The user's original task/query.
        history: Conversation messages exchanged with the LLM so far.
        steps: The ordered ReAct steps (thought/action/observation) taken.
        iteration: How many thought/action/observation cycles have run.
        is_complete: Whether the agent has produced a final answer.
        final_answer: The agent's final answer, once ``is_complete`` is True.
    """

    task: str
    history: list[Message] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    iteration: int = 0
    is_complete: bool = False
    final_answer: str | None = None

    def record(self, step: AgentStep) -> None:
        """Append a step to this run's trace and advance the iteration count.

        Args:
            step: The step just produced by the agent loop.
        """
        self.steps.append(step)
        if step.step_type == StepType.ACTION:
            self.iteration += 1
        if step.step_type == StepType.FINAL_ANSWER:
            self.is_complete = True
            self.final_answer = step.content


@runtime_checkable
class Tool(Protocol):
    """Structural interface every pluggable tool must satisfy.

    A tool is any object exposing a ``name``, a natural-language
    ``description`` (used in the LLM's tool-selection prompt), and a
    synchronous ``run`` method. Concrete tools (calculator, web search,
    file I/O, ...) are implemented in ``src/tools`` starting day 2.
    """

    name: str
    description: str

    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result for the LLM to read."""
        ...
