"""Data models shared across the toolkit.

These Pydantic models are the wire format between the LLM client, the agent loop,
tools and memory backends. They are deliberately small and fully serializable so an
agent run can be logged, persisted and replayed.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Author of a chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A request from the model to run a tool.

    Attributes:
        id: Unique identifier used to pair the call with its result.
        name: Registered tool name.
        arguments: Keyword arguments for the tool.

    Example:
        >>> ToolCall(name="calculator", arguments={"expression": "2+2"}).name
        'calculator'
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing a :class:`ToolCall`.

    Attributes:
        call_id: Identifier of the originating call.
        name: Name of the tool that ran.
        output: Text returned to the model as the observation.
        is_error: ``True`` when the tool failed and ``output`` holds the error text.
    """

    call_id: str
    name: str
    output: str
    is_error: bool = False


class Message(BaseModel):
    """A single chat message.

    Attributes:
        role: Who produced the message.
        content: Message text.
        tool_calls: Tool invocations requested by an assistant message.
        tool_call_id: For ``Role.TOOL`` messages, the call being answered.
        timestamp: Creation time as a UNIX timestamp.

    Example:
        >>> Message.user("hello").role is Role.USER
        True
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[list[ToolCall]] = None) -> "Message":
        """Create an assistant message, optionally carrying tool calls."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def from_tool_result(cls, result: ToolResult) -> "Message":
        """Create a tool message from a :class:`ToolResult`."""
        return cls(role=Role.TOOL, content=result.output, tool_call_id=result.call_id)


class AgentStep(BaseModel):
    """One iteration of the ReAct loop.

    Attributes:
        index: Zero-based step number.
        thought: The model's reasoning for this step.
        action: Tool call chosen, or ``None`` on the final step.
        observation: Result of the action, or ``None`` on the final step.
        final_answer: Set on the step that terminates the loop.
    """

    index: int = Field(ge=0)
    thought: str = ""
    action: Optional[ToolCall] = None
    observation: Optional[ToolResult] = None
    final_answer: Optional[str] = None

    @property
    def is_final(self) -> bool:
        """Whether this step ended the run with an answer."""
        return self.final_answer is not None
