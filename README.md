# llm-agent-toolkit

A modular toolkit for building tool-using LLM agents. The goal is a small, typed,
testable core: a pluggable tool registry, conversation/episodic memory, a ReAct
agent loop, a task planner, and REST/WebSocket and CLI front-ends.

**Status:** Day 1 of 7 — scaffold, shared types, configuration, and architecture.

## Roadmap

| Day | Deliverable |
|-----|-------------|
| 1 | Scaffold, core types, settings, logging, architecture design |
| 2 | Tool registry (calculator, search, file tools) |
| 3 | Memory backends (conversation, episodic, Redis) |
| 4 | ReAct agent core |
| 5 | Planner and executor |
| 6 | REST API and streaming WebSocket |
| 7 | Rich CLI and demo |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
```

See [docs/architecture.md](docs/architecture.md) for the design.
