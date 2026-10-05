"""llm-agent-toolkit: a ReAct-pattern LLM agent framework.

Public API re-exports the pieces most users need: configuration, the core
message/step schemas, and the tool abstractions.
"""

from agent_toolkit.config import Settings, get_settings
from agent_toolkit.exceptions import (
    AgentToolkitError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
)
from agent_toolkit.schemas import AgentStep, Message, Role, ToolCall, ToolResult
from agent_toolkit.tools.base import Tool
from agent_toolkit.tools.registry import ToolRegistry

__version__ = "0.1.0"

__all__ = [
    "AgentStep",
    "AgentToolkitError",
    "Message",
    "Role",
    "Settings",
    "Tool",
    "ToolCall",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolResult",
    "__version__",
    "get_settings",
]
