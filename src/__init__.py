"""llm-agent-toolkit: a ReAct-pattern LLM agent framework.

This package provides the building blocks for constructing tool-using LLM
agents: a pluggable tool registry, swappable memory backends (in-process or
Redis-backed), a ReAct (Reason + Act) execution loop, a multi-step task
planner, and both a REST/WebSocket API and a Rich-based interactive CLI.

Planned module layout:
    src.config    -- centralized, typed application settings
    src.agent     -- core ReAct loop, planner, and shared types
    src.tools     -- the tool interface and built-in tool registry
    src.memory    -- conversation / episodic memory interfaces and backends
    src.api       -- FastAPI REST + WebSocket interface (added later)
    src.cli       -- Rich-based interactive CLI (added later)
"""

__version__ = "0.1.0"
