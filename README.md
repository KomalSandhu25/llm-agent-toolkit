# llm-agent-toolkit

A ReAct-pattern LLM agent framework with pluggable tools, persistent memory,
multi-step planning, and an interactive CLI.

## Why this exists

Most "agent frameworks" hide the ReAct loop (Thought -> Action -> Observation)
behind several layers of abstraction. This project builds it from scratch,
layer by layer, so every piece -- the tool registry, the memory backends, the
planner, the executor, the API, the CLI -- is small enough to read in one
sitting and swap out independently.

## Architecture at a glance

```
+-----------------------------------------------------------+
|                     CLI / REST / WebSocket                |  <- Day 6-7
+-------------------------------------------------------------+
|                    Planner + Executor                     |  <- Day 5
+-------------------------------------------------------------+
|                    ReAct Agent Core                        |  <- Day 4
+------------------------+------------------------------------+
|      Tool Registry     |        Memory Backends            |  <- Day 2-3
+------------------------+------------------------------------+
|                  Core interfaces & types                  |  <- Day 1 (this commit)
+-----------------------------------------------------------+
```

See [`docs/architecture.md`](docs/architecture.md) for the full design
rationale, including why the core interfaces are defined before any concrete
tool, memory backend, or LLM client.

## Project layout

```
src/llm_agent_toolkit/
    config.py          Pydantic settings (env-driven configuration)
    exceptions.py       Exception hierarchy used across every layer
    logging_utils.py    Rich-based structured logging setup
    core/
        types.py         Core dataclasses/enums: AgentStep, ToolResult, etc.
        interfaces.py    Abstract base classes every concrete component implements
tests/
    test_config.py
    test_core_types.py
    test_interfaces.py
docs/
    architecture.md
```

## Roadmap

| Day | Deliverable |
|-----|-------------|
| 1 | Project scaffold and agent architecture design (this commit) |
| 2 | Pluggable tool registry (calculator, search, file tools) |
| 3 | Conversation and episodic memory backends |
| 4 | ReAct agent core (thought/action/observation loop) |
| 5 | Multi-step task planner and agent executor |
| 6 | REST API and streaming WebSocket interface |
| 7 | Interactive Rich CLI and end-to-end demo |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Configuration is environment-driven -- see `src/llm_agent_toolkit/config.py`.
At minimum, set `LLM_AGENT_OPENAI_API_KEY` before running anything that talks
to a real model; the test suite does not require it.
