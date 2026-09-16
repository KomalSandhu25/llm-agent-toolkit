# llm-agent-toolkit

A ReAct-pattern LLM agent framework with pluggable tools, persistent memory,
multi-step planning, and an interactive CLI.

## What this is

`llm-agent-toolkit` is a small, dependency-light framework for building LLM
agents that **reason, act, and observe** in a loop (the
[ReAct pattern](https://arxiv.org/abs/2210.03629)), rather than a single
black-box "chat and hope" call. Every major piece — the LLM client, the tools
the agent can call, the memory it reads and writes, and the planner that
breaks a goal into steps — is defined as a small interface, so any of them
can be swapped out without touching the agent core.

This repository is being built incrementally, day by day, as a portfolio
project. Day 1 lays down the project scaffold and the core architecture;
later days fill in tools, memory backends, the ReAct loop, planning, a REST
API, and a CLI.

## Architecture at a glance

```
                       ┌─────────────────────┐
                       │        Agent          │  orchestrates the loop
                       │  (thought → action →   │
                       │      observation)      │
                       └──────────┬───────────┘
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
          ┌───────────────┐┌───────────┐┌──────────────────┐
          │   LLM Client    ││  Tool     ││     Memory        │
          │ (OpenAI, etc.)  ││ Registry  ││ (conversation +    │
          │                 ││ (pluggable││  episodic)         │
          │                 ││  tools)   ││                    │
          └───────────────┘└───────────┘└──────────────────┘
                                  ▲
                                  │
                          ┌───────────────┐
                          │    Planner      │  multi-step task
                          │                 │  decomposition
                          └───────────────┘
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design rationale.

## Project layout

```
llm-agent-toolkit/
├── src/
│   └── llm_agent_toolkit/
│       ├── __init__.py       # public package exports
│       ├── config.py         # environment-driven settings (pydantic)
│       ├── exceptions.py     # exception hierarchy
│       └── core/
│           ├── __init__.py
│           ├── types.py       # shared data models (Message, ToolCall, ...)
│           └── interfaces.py  # abstract base classes for every component
├── tests/
│   ├── test_config.py
│   ├── test_types.py
│   └── test_interfaces.py
├── pyproject.toml
└── ARCHITECTURE.md
```

## Status

| Day | Milestone |
|-----|-----------|
| 1 | Project scaffold and architecture design ✅ |
| 2 | Pluggable tool registry (calculator, search, file tools) |
| 3 | Conversation and episodic memory backends |
| 4 | ReAct agent core (thought/action/observation loop) |
| 5 | Multi-step task planner and executor |
| 6 | REST API and streaming WebSocket interface |
| 7 | Interactive Rich CLI and end-to-end demo |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Configuration is read from environment variables (see `config.py`), e.g.:

```bash
export OPENAI_API_KEY=sk-...
export AGENT_MODEL=gpt-4o-mini
export REDIS_URL=redis://localhost:6379/0
```

## License

MIT
