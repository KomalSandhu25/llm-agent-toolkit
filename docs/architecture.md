# Architecture

## Goal

Implement the ReAct pattern (Yao et al., 2022: "ReAct: Synergizing Reasoning
and Acting in Language Models") as a small, inspectable set of layers rather
than a monolithic "Agent" class. Each layer depends only on the interfaces
defined in `core/interfaces.py`, never on a concrete implementation of
another layer, so any layer can be swapped without touching the others.

## The ReAct loop this toolkit implements

```
loop until FINAL_ANSWER or max_iterations reached:

  1. THOUGHT       agent reasons about what to do next
  2. ACTION        agent selects a tool + arguments
  3. OBSERVATION   tool executes, result is fed back in
```

Each iteration produces one `AgentStep` (see `core/types.py`). A full run
produces an ordered list of steps plus a terminal `AgentResponse`.

## Layers, and why they are separated this way

### 1. Core types and interfaces (Day 1 -- this commit)

`core/types.py` defines the vocabulary every other layer speaks:
`StepType`, `AgentStep`, `ToolSpec`, `ToolResult`, `AgentResponse`. These are
plain, immutable `pydantic.BaseModel` subclasses -- no behavior, just shape.
Fixing the shape first means the tool registry (Day 2), memory backends
(Day 3), and agent core (Day 4) can all be built against a stable contract
instead of each other.

`core/interfaces.py` defines the four abstract base classes concrete
components will implement later:

- `BaseLLMClient` -- wraps a chat-completion call. Swappable between OpenAI,
  a LangChain-wrapped model, or a local model without changing the agent
  core.
- `BaseTool` -- a named, described, schema-validated capability the agent can
  invoke. The registry (Day 2) is a collection of these.
- `BaseMemory` -- an append/read/search interface over agent history. Two
  backends are planned: an in-process conversation buffer and a persistent
  (Redis-backed) episodic store (Day 3).
- `BaseAgent` -- the thing that actually runs the ReAct loop (Day 4) and,
  later, delegates multi-step goals to a planner (Day 5).

### 2. Tool registry (Day 2)

A `ToolRegistry` will hold `BaseTool` instances, exposing their `ToolSpec`s
to the LLM client as function-calling schemas, and dispatching `ACTION`
steps to the right tool by name.

### 3. Memory (Day 3)

`ConversationMemory` (in-process, bounded) and `EpisodicMemory`
(Redis-backed, persists across process restarts) both implement
`BaseMemory`, so the agent core never needs to know which one it's talking
to.

### 4. Agent core (Day 4)

`ReActAgent(BaseAgent)` orchestrates: pull relevant memory, ask the LLM
client for a `THOUGHT`/`ACTION`, dispatch the action via the tool registry,
record the `OBSERVATION`, repeat until a `FINAL_ANSWER` step or
`max_iterations` is hit (raising `MaxIterationsExceededError`).

### 5. Planner + executor (Day 5)

For goals that need decomposition before any ReAct loop starts, a
`TaskPlanner` will produce an ordered list of sub-goals, and an
`AgentExecutor` will run a `ReActAgent` against each one, aggregating
results.

### 6. API layer (Day 6)

A FastAPI app exposes `POST /agent/run` (single-shot) and a WebSocket
endpoint that streams `AgentStep`s as they're produced, so a client can show
the agent "thinking" in real time.

### 7. CLI (Day 7)

A `rich`-based interactive CLI wraps the same `BaseAgent` interface,
rendering each `AgentStep` live and ending with a worked end-to-end demo.

## Design decisions worth calling out

- **Pydantic everywhere for data shapes.** Every cross-layer payload
  (`ToolSpec`, `AgentStep`, `AgentResponse`, `Settings`) is a `pydantic`
  model. This gives free validation at every boundary and JSON
  serialization for the API layer for free.
- **Exceptions are a hierarchy, not a grab-bag.** `exceptions.py` defines a
  single root (`LLMAgentToolkitError`) so callers can catch broadly or
  narrowly, and each layer raises its own subtype (`ToolExecutionError`,
  `MemoryBackendError`, `MaxIterationsExceededError`, `PlanningError`).
- **No concrete LLM/tool/memory implementation ships on Day 1.** The
  interfaces are deliberately implementation-free so that Day 2-4 code is
  written against a contract that cannot silently drift to match whatever
  the first implementation happened to do.
