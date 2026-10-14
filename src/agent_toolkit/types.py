"""Core data types describing a ReAct agent trace.

A run is a sequence of :class:`Step` objects, each pairing the model's
``Thought`` and chosen ``Action`` with the resulting ``Observation``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """A single chat message.

    Args:
        role: Who produced the message.
        content: Message text.

    Example:
        >>> Message(role="user", content="hi").role
        'user'
    """

    role: Role
    content: str


class Action(BaseModel):
    """A tool invocation requested by the model.

    Args:
        tool: Registered tool name.
        arguments: Keyword arguments passed to the tool.
    """

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    """One Thought -> Action -> Observation iteration.

    Args:
        thought: The model's reasoning text.
        action: Requested tool call, or ``None`` on the final step.
        observation: Tool output fed back to the model.
        final_answer: Set when the model decides it is finished.
    """

    thought: str
    action: Action | None = None
    observation: str | None = None
    final_answer: str | None = None

    @property
    def is_final(self) -> bool:
        """Whether this step terminates the run."""
        return self.final_answer is not None


class AgentRun(BaseModel):
    """Complete record of an agent execution.

    Args:
        question: The original user task.
        steps: Ordered reasoning steps.
    """

    question: str
    steps: list[Step] = Field(default_factory=list)

    @property
    def answer(self) -> str | None:
        """Final answer if the run completed, else ``None``.

        Returns:
            The last step's final answer, if any.
        """
        return self.steps[-1].final_answer if self.steps else None
