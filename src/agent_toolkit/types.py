"""Core data types shared across the agent loop, tools, and memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepKind(str, Enum):
    """Kinds of entries in an agent trace."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL = "final"


@dataclass(frozen=True)
class Step:
    """A single entry in the ReAct trace.

    Args:
        kind: What type of step this is.
        content: Free-text content (thought, observation, or answer).
        tool: Tool name for ACTION steps.
        tool_input: Arguments for ACTION steps.

    Example:
        >>> Step(StepKind.THOUGHT, "I should add the numbers").kind.value
        'thought'
    """

    kind: StepKind
    content: str = ""
    tool: str | None = None
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Ordered record of everything an agent did for one task."""

    task: str
    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> None:
        """Append a step to the trace."""
        self.steps.append(step)

    @property
    def final_answer(self) -> str | None:
        """Return the final answer text if the agent produced one."""
        for step in reversed(self.steps):
            if step.kind is StepKind.FINAL:
                return step.content
        return None
