"""LLM Agent Toolkit.

A ReAct-pattern agent framework with pluggable tools, persistent memory,
multi-step planning and an interactive CLI.

Example:
    >>> from agent_toolkit import AgentConfig
    >>> AgentConfig().max_steps
    10
"""

from agent_toolkit.config import AgentConfig
from agent_toolkit.exceptions import (
    AgentToolkitError,
    ConfigurationError,
    MaxStepsExceededError,
    MemoryBackendError,
    ToolExecutionError,
    ToolNotFoundError,
)
from agent_toolkit.interfaces import BaseMemory, BaseTool, LLMClient
from agent_toolkit.schemas import AgentStep, Message, Role, ToolCall, ToolResult

__version__ = "0.1.0"

__all__ = [
    "AgentConfig",
    "AgentStep",
    "AgentToolkitError",
    "BaseMemory",
    "BaseTool",
    "ConfigurationError",
    "LLMClient",
    "MaxStepsExceededError",
    "MemoryBackendError",
    "Message",
    "Role",
    "ToolCall",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolResult",
    "__version__",
]
