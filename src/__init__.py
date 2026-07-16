"""llm-agent-toolkit: a ReAct-pattern LLM agent framework.

This package provides the building blocks for constructing tool-using LLM
agents that follow the ReAct (Reason + Act) loop: the agent alternates
between emitting a *thought*, choosing an *action* (a tool call), and
receiving an *observation* (the tool's result), until it decides it has
enough information to produce a *final answer*.

Planned module layout (built incrementally, day by day):

    src/
        config.py        Centralized, typed application settings.
        core/             Framework-wide types and exceptions shared by
                          every other module (this day's focus).
        tools/            Pluggable tool interface + concrete tools
                          (calculator, search, file I/O) and a registry.
        memory/           Conversation + episodic memory backends
                          (in-memory and Redis-backed).
        agent/            The ReAct agent core: the thought/action/
                          observation loop itself.
        planning/         Multi-step task planning and execution.
        api/              FastAPI REST + WebSocket streaming interface.
        cli/              Interactive Rich-based command-line interface.

Example:
    >>> from src.config import get_settings
    >>> settings = get_settings()
    >>> settings.openai_model
    'gpt-4o-mini'
"""

__version__ = "0.1.0"
