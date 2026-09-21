"""llm-agent-toolkit: a ReAct-pattern LLM agent framework.

This package provides the building blocks for constructing tool-using LLM
agents: a typed core data model (src.core), a pluggable tool registry
(src.tools), pluggable memory backends (src.memory), the ReAct agent loop
itself (src.agent), a multi-step planner (src.planning), a FastAPI +
WebSocket serving layer (src.api), and an interactive Rich CLI (src.cli).
"""

__version__ = "0.1.0"
