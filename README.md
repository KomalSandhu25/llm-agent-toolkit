# llm-agent-toolkit

A ReAct-pattern LLM agent framework with pluggable tools, persistent
memory, multi-step planning, and an interactive CLI.

> **Status:** early scaffold (day 1 of 7). Architecture below describes
> the target design; most modules are stubs to be filled in over the
> coming days.

## Architecture

```
                         +----------------------+
                         |   src/cli (day 7)     |
                         |   src/api (day 6)     |
                         +-----------+-----------+
                                     |
                         +-----------v-----------+
                         |  src/planning (day 5)  |  multi-step task decomposition
                         +-----------+-----------+
                                     |
                         +-----------v-----------+
                         |  src/agent (day 4)     |  ReAct loop:
                         |  thought -> action ->  |  thought/action/observation
                         |  observation -> ...    |
                         +------+-----------+-----+
                                |           |
                    +-----------v--+   +----v-----------+
                    | src/tools     |   | src/memory     |
                    | (day 2)       |   | (day 3)        |
                    | calculator,   |   | conversation + |
                    | search, file  |   | episodic, Redis|
                    +---------------+   +----------------+
                                |
                    +-----------v-----------+
                    | src/core/types.py      |  Message, ToolCall,
                    | (day 1 - this commit)  |  ToolResult, AgentState
                    +------------------------+
```

The ReAct pattern (Yao et al., 2022) interleaves *reasoning* ("thoughts")
with *acting* (tool calls) and *observing* (tool results), letting the
agent decide at each step whether it has enough information to answer or
needs to call another tool. `src/core/types.py` defines the shared
vocabulary -- `Message`, `ToolCall`, `ToolResult`, `AgentStep`,
`AgentState`, and the `Tool` protocol -- that every other layer is built
against, so tools, memory, the agent loop, and the API can all be
developed and tested independently.

## Planned components

| Module | Added | Responsibility |
|---|---|---|
| `src/core` | Day 1 | Typed data model shared by every other module |
| `src/tools` | Day 2 | Pluggable tool registry (calculator, search, file I/O) |
| `src/memory` | Day 3 | Conversation + episodic memory, in-process and Redis |
| `src/agent` | Day 4 | ReAct thought/action/observation loop |
| `src/planning` | Day 5 | Multi-step task decomposition and execution |
| `src/api` | Day 6 | FastAPI REST + streaming WebSocket interface |
| `src/cli` | Day 7 | Interactive Rich-based terminal client |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # then set OPENAI_API_KEY
pytest
```

## Configuration

All settings are validated at startup via `src/config.py`'s Pydantic
`Settings` class. See `.env.example` for the full list of environment
variables (LLM provider, agent iteration limits, Redis memory backend,
logging, and API server host/port).

## Tech stack

`openai` . `langchain` . `fastapi` . `redis` . `pydantic` . `rich`
