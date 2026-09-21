# llm-agent-toolkit

A ReAct-pattern LLM agent framework with pluggable tools, persistent memory,
multi-step planning, and an interactive CLI.

> **Status: under active development.** This README describes the target
> architecture; features are being built incrementally (see Roadmap below).

## Architecture

```
                       +-------------------+
                       |   CLI / REST API  |
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       |   Planner (opt.)  |   multi-step task decomposition
                       +---------+---------+
                                 |
                                 v
      +--------------------------------------------------+
      |                  ReAct Agent Loop                  |
      |   Thought -> Action (ToolCall) -> Observation      |
      |   ... repeats until final_answer or max_iterations |
      +---------+----------------------------+-------------+
                |                            |
                v                            v
      +-------------------+       +----------------------+
      |   Tool Registry    |       |   Memory Backend      |
      |  (calculator,       |       |  (in-process / Redis) |
      |   search, files...) |       +----------------------+
      +-------------------+
```

The core loop is the classic **ReAct** pattern (Yao et al., 2022): at each
iteration the LLM produces a *thought* about what to do next, either invokes
a *tool* or emits a *final answer*, and -- if a tool was called -- observes
the result before reasoning again. A memory backend keeps conversation state
across turns and sessions; a planner sits above the loop to break larger
tasks into ordered sub-tasks before execution begins.

## Planned components

| Module | Purpose | Status |
|---|---|---|
| `src.config` | Typed, env-driven settings | done |
| `src.agent.types` | Shared dataclasses (`Step`, `Thought`, `ToolCall`, ...) | done |
| `src.tools` | `Tool` interface + built-in tool registry | interface done, tools day 2 |
| `src.memory` | `Memory` interface + in-process/Redis backends | interface done, backends day 3 |
| `src.agent.loop` | The ReAct execution loop | day 4 |
| `src.agent.planner` | Multi-step task planner | day 5 |
| `src.api` | FastAPI REST + WebSocket streaming interface | day 6 |
| `src.cli` | Rich-based interactive CLI | day 7 |

## Quickstart (once complete)

```bash
pip install -e ".[dev]"
cp .env.example .env   # add your OPENAI_API_KEY
agent-cli
```

## Design principles

- **Backend-agnostic interfaces.** `Tool` and `Memory` are abstract base
  classes so swapping a Redis memory store for an in-process one (e.g. in
  tests) requires no changes to the agent loop.
- **Bounded execution.** The agent loop always respects
  `AGENT_MAX_ITERATIONS`, guaranteeing termination even when the model fails
  to converge on a final answer.
- **Typed everywhere.** All public functions and dataclasses are fully typed
  and documented; `mypy --strict` is expected to pass.

## Roadmap

1. ~~Project scaffold and architecture~~ (this commit)
2. Pluggable tool registry (calculator, search, file tools)
3. Conversation and episodic memory backends
4. ReAct agent core (thought/action/observation loop)
5. Multi-step task planner and executor
6. REST API and streaming WebSocket interface
7. Interactive Rich CLI and end-to-end demo
