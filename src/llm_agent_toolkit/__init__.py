"""llm-agent-toolkit: a ReAct-pattern LLM agent framework.

This package provides the building blocks for constructing LLM agents that
reason, act, and observe in a loop, with pluggable tools, memory backends,
and planners. See ``ARCHITECTURE.md`` in the repository root for the full
design rationale.

Public API surface (grows as later milestones land):

Example:
    >>> from llm_agent_toolkit import Settings
    >>> settings = Settings(openai_api_key="sk-...")
    >>> settings.agent_model
    'gpt-4o-mini'
"""

from __future__ import annotations

from llm_agent_toolkit.config import Settings
from llm_agent_toolkit.core.types import (
    AgentStatus,
    AgentStep,
    Message,
    Role,
    ToolCall,
    ToolResult,
)
from llm_agent_toolkit.exceptions import (
    AgentToolkitError,
    ConfigurationError,
    LLMClientError,
    MaxIterationsExceededError,
    MemoryBackendError,
    PlanningError,
    ToolExecutionError,
    ToolNotFoundError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Settings",
    "Role",
    "Message",
    "ToolCall",
    "ToolResult",
    "AgentStep",
    "AgentStatus",
    "AgentToolkitError",
    "ConfigurationError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "MemoryBackendError",
    "PlanningError",
    "LLMClientError",
    "MaxIterationsExceededError",
]
