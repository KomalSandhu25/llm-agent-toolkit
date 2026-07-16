# llm-agent-toolkit

A ReAct-pattern LLM agent framework with pluggable tools, persistent
memory, multi-step planning, and an interactive CLI.

> **Status:** early scaffold (Day 1 of 7). Core types and configuration are
> in place; tools, memory, the agent loop, planning, and the API/CLI are
> built out over the following days. See the roadmap below.

## Architecture

The agent follows the classic **ReAct** (Reason + Act) loop: it
alternates between reasoning about the task, choosing a tool to act with,
and observing the result, until it has enough information to answer.

```
                     +--------------------+
                     |        Task        |
                     +---------+----------+
                               |
                               v
                     +--------------------+
              +----->|      Thought       |
              |      +---------+----------+
              |                |
              |                v
              |      +--------------------+
              |      |       Action       |----> ToolRegistry ----> BaseTool
              |      +---------+----------+                            |
              |                |                                      |
              |                v                                      |
              |      +--------------------+                           |
              +------|     Observation    |<--------------------------+
                     +---------+----------+
                               |
                          (repeat until
                         confident, or
                     agent_max_iterations)
                               |
                               v
                     +--------------------+
                     |    Final Answer    |
                     +--------------------+

  Cross-cutting:
    Memory   — conversation + episodic state, persisted to Redis
    Planning — decomposes multi-step tasks before the loop starts
    API/CLI  — FastAPI + WebSocket streaming, and a Rich terminal client
```

## Planned module layout

| Module         | Responsibility                                                   | Day |
|----------------|-------------------------------------------------------------------|:---:|
| `src/config.py`| Centralized, typed application settings                          | 1   |
| `src/core/`    | Shared types (`Message`, `AgentStep`, `AgentState`) & exceptions  | 1   |
| `src/tools/`   | Tool interface, concrete tools, and a registry                   | 2   |
| `src/memory/`  | Conversation + episodic memory (in-memory and Redis-backed)       | 3   |
| `src/agent/`   | The ReAct thought/action/observation loop                        | 4   |
| `src/planning/`| Multi-step task planner and executor                             | 5   |
| `src/api/`     | FastAPI REST + streaming WebSocket interface                     | 6   |
| `src/cli/`     | Interactive Rich-based CLI and demo                               | 7   |

## Tech stack

`openai` · `langchain` · `fastapi` · `redis` · `pydantic` (+ `pydantic-settings`) · `rich`

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in OPENAI_API_KEY
pytest
```

## License

MIT
