# llm-agent-toolkit

A ReAct-pattern LLM agent framework with pluggable tools, persistent memory,
multi-step planning, and an interactive CLI.

**Status:** Day 1 of 7 - project scaffold and core abstractions.

## Planned architecture

```
            +-----------------+
 user ----> |   CLI / API     |  (rich, fastapi)
            +--------+--------+
                     |
              +------v------+      +------------------+
              |  ReAct loop |<---->|  Memory (Redis)  |
              +------+------+      +------------------+
                     |
          Thought -> Action -> Observation
                     |
              +------v------+
              | ToolRegistry|--> Tool A, Tool B, ... (pydantic-validated)
              +-------------+
```

## What exists so far

- `config.py` - environment-driven settings (`pydantic-settings`)
- `types.py` - `Message`, `Action`, `Step`, `AgentRun` trace models
- `tools/base.py` - `Tool` ABC with schema validation and error containment
- `tools/registry.py` - `ToolRegistry` for registering and dispatching tools

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
```

## Roadmap

1. Scaffold and architecture (this commit)
2. ReAct loop and output parser
3. Built-in tools
4. Persistent memory (Redis)
5. Multi-step planning
6. Interactive CLI
7. API, docs, polish
