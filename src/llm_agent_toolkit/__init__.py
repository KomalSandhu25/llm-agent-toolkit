"""llm_agent_toolkit: a ReAct-pattern LLM agent framework.

This package is built up incrementally, one architectural layer at a time:

    core.types        Immutable data shapes shared across every layer.
    core.interfaces   Abstract base classes each concrete component implements.
    config            Environment-driven settings.
    exceptions        The exception hierarchy every layer raises from.
    logging_utils     Rich-based structured logging setup.

Concrete tool, memory, agent, planner, API, and CLI implementations land in
later revisions; see docs/architecture.md for the full roadmap.
"""

from llm_agent_toolkit.core.types import (
    AgentResponse,
    AgentStep,
    StepType,
    ToolResult,
    ToolSpec,
)
from llm_agent_toolkit.core.interfaces import (
    BaseAgent,
    BaseLLMClient,
    BaseMemory,
    BaseTool,
)
from llm_agent_toolkit.exceptions import (
    AgentExecutionError,
    LLMAgentToolkitError,
    MaxIterationsExceededError,
    MemoryBackendError,
    PlanningError,
    ToolExecutionError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core types
    "AgentResponse",
    "AgentStep",
    "StepType",
    "ToolResult",
    "ToolSpec",
    # core interfaces
    "BaseAgent",
    "BaseLLMClient",
    "BaseMemory",
    "BaseTool",
    # exceptions
    "AgentExecutionError",
    "LLMAgentToolkitError",
    "MaxIterationsExceededError",
    "MemoryBackendError",
    "PlanningError",
    "ToolExecutionError",
]
