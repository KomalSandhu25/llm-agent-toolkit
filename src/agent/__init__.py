"""Core agent package: ReAct loop, planner, and shared data types.

This package will host the agent's reasoning loop (``loop.py``, added on
day 4) and the multi-step planner (``planner.py``, added on day 5). For now
it exposes the shared vocabulary -- the dataclasses and enums that every other
component (tools, memory, API, CLI) is built against -- so downstream modules
have a stable interface to import from day one.
"""

from src.agent.types import (
    AgentResult,
    AgentStatus,
    Observation,
    Step,
    StepType,
    Thought,
    ToolCall,
)

__all__ = [
    "AgentResult",
    "AgentStatus",
    "Observation",
    "Step",
    "StepType",
    "Thought",
    "ToolCall",
]
