"""Core data models shared across the agent loop, tools, and memory.

These models are intentionally provider-agnostic; adapters convert them to
and from the OpenAI chat format.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """Chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A request from the model to invoke a tool.

    Attributes:
        id: Provider-assigned call identifier.
        name: Name of the tool to invoke.
        arguments: JSON-decoded keyword arguments.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing a :class:`ToolCall`.

    Attributes:
        call_id: Identifier of the originating call.
        name: Tool name.
        output: Stringified tool output shown to the model.
        is_error: True when execution failed and ``output`` holds the error.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    output: str
    is_error: bool = False


class Message(BaseModel):
    """A single chat message.

    Example:
        >>> Message(role=Role.USER, content="hi").role is Role.USER
        True
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class AgentStep(BaseModel):
    """One iteration of the ReAct loop: thought, action, observation.

    Attributes:
        index: Zero-based step number.
        thought: The model's reasoning text for this step.
        tool_calls: Actions requested this step (empty on the final answer).
        observations: Results of executing ``tool_calls``.
        final_answer: Set when the model produced a terminal answer.
    """

    index: int = Field(ge=0)
    thought: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observations: list[ToolResult] = Field(default_factory=list)
    final_answer: str | None = None

    @property
    def is_final(self) -> bool:
        """Whether this step ended the run with a final answer."""
        return self.final_answer is not None
