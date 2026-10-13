"""Core data models shared across the toolkit.

These Pydantic models define the wire format between the agent loop, tools,
memory backends and the API layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Conversation roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single conversation message.

    Args:
        role: Who produced the message.
        content: Text content.
        name: Optional tool name when ``role`` is ``TOOL``.

    Example:
        >>> Message(role=Role.USER, content="hi").to_dict()
        {'role': 'user', 'content': 'hi'}
    """

    role: Role
    content: str
    name: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        """Serialize to the chat-completions message shape.

        Returns:
            A dict with ``role`` and ``content`` (and ``name`` when set).
        """
        data = {"role": self.role.value, "content": self.content}
        if self.name:
            data["name"] = self.name
        return data


class ToolCall(BaseModel):
    """A request from the model to invoke a tool.

    Example:
        >>> ToolCall(name="calculator", arguments={"expression": "2+2"}).name
        'calculator'
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing a tool.

    Attributes:
        name: Tool that produced the result.
        output: Text returned to the model as the observation.
        is_error: True if the tool failed; ``output`` then holds the error text.
    """

    name: str
    output: str
    is_error: bool = False


class AgentStep(BaseModel):
    """One thought/action/observation iteration of the agent loop."""

    index: int = Field(ge=0)
    thought: str = ""
    action: Optional[ToolCall] = None
    observation: Optional[ToolResult] = None


class AgentResult(BaseModel):
    """Final output of an agent run.

    Example:
        >>> AgentResult(answer="4", steps=[]).num_steps
        0
    """

    answer: str
    steps: list[AgentStep] = Field(default_factory=list)

    @property
    def num_steps(self) -> int:
        """Number of steps taken."""
        return len(self.steps)
