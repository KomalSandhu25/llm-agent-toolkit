"""Shared data models used across the agent toolkit.

Every value that crosses a boundary between components (an LLM client, a
tool, a memory backend, the agent loop) is one of the models defined here.
Keeping this as the single source of shapes means every other module
imports from this file rather than inventing similar-looking dicts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Who authored a :class:`Message`.

    Example:
        >>> Role.USER.value
        'user'
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single turn in a conversation.

    Attributes:
        role: Who authored the message.
        content: The textual content of the message.
        name: Optional identifier, used e.g. to name the tool that produced
            a ``TOOL``-role message.
        tool_call_id: When this message is the result of a tool call, the
            id of the originating :class:`ToolCall`.
        created_at: UTC timestamp the message was constructed.

    Example:
        >>> msg = Message(role=Role.USER, content="What is 2 + 2?")
        >>> msg.role
        <Role.USER: 'user'>
    """

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolCall(BaseModel):
    """A request, produced by the LLM, to invoke a named tool.

    Attributes:
        id: Unique identifier for this call, used to correlate the eventual
            :class:`ToolResult`.
        tool_name: Name of the tool to invoke, matching a registered
            ``BaseTool.name``.
        arguments: Keyword arguments to pass to the tool, already parsed
            from the LLM's raw output into a plain dict.

    Example:
        >>> call = ToolCall(tool_name="calculator", arguments={"expression": "2 + 2"})
        >>> call.tool_name
        'calculator'
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing a :class:`ToolCall`.

    Attributes:
        tool_call_id: The id of the :class:`ToolCall` this result answers.
        tool_name: Name of the tool that produced this result.
        content: Human-readable result content, suitable for feeding back
            into the LLM as an observation.
        is_error: Whether the tool execution failed.

    Example:
        >>> result = ToolResult(tool_call_id="abc123", tool_name="calculator", content="4")
        >>> result.is_error
        False
    """

    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False


class AgentStatus(str, Enum):
    """The state of an agent run.

    Example:
        >>> AgentStatus.FINISHED.value
        'finished'
    """

    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"


class AgentStep(BaseModel):
    """One iteration of the ReAct thought/action/observation loop.

    Attributes:
        index: Zero-based position of this step within the run.
        thought: The reasoning text produced by the LLM for this step.
        action: The tool call the LLM chose to make, if any. ``None`` when
            the step concluded with a final answer instead of a tool call.
        observation: The result of executing ``action``, if any.

    Example:
        >>> step = AgentStep(index=0, thought="I should compute 2 + 2.")
        >>> step.action is None
        True
    """

    index: int = Field(ge=0)
    thought: str
    action: ToolCall | None = None
    observation: ToolResult | None = None
