# llm-agent-toolkit

A ReAct-pattern LLM agent framework with pluggable tools, persistent memory,
multi-step planning, and an interactive CLI.

> Status: **in active development** - the scaffold and core contracts are in place;
> tools, memory, the agent loop, planner, API and CLI land in subsequent milestones.

## Design goals

- **Small, typed contracts.** Every moving part (LLM client, tool, memory) is an
  interface defined in `agent_toolkit.interfaces`, validated with Pydantic models from
  `agent_toolkit.schemas`.
- **Provider-agnostic.** The agent talks to an `LLMClient` protocol; OpenAI is just one
  implementation.
- **Testable without network access.** Everything can be exercised with deterministic
  fakes.

## Layout

```
src/agent_toolkit/
    config.py       # AgentConfig, environment-driven configuration
    exceptions.py   # exception hierarchy
    schemas.py      # Message, ToolCall, ToolResult, AgentStep, ...
    interfaces.py   # LLMClient protocol, BaseTool, BaseMemory
tests/
docs/ARCHITECTURE.md
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.
