# Architecture

## Goals

1. **Swap anything.** The LLM provider, the tools, the memory backend, and
   the planner are each defined behind a small interface. Nothing in the
   agent core imports a concrete implementation directly.
2. **Typed and explicit.** All data that crosses a boundary (a message, a
   tool call, a tool result) is a `pydantic` model, not a loose `dict`. This
   catches shape mistakes at construction time instead of deep in a prompt
   string.
3. **Small core, composable pieces.** The agent loop itself should stay
   short and readable — it delegates to the LLM client, the tool registry,
   and memory rather than accumulating special cases.
4. **Observable by default.** Every step of the reasoning loop (thought,
   action, observation) is a structured object that can be logged, replayed,
   or rendered in a CLI/UI, rather than free text buried in a transcript.

## Layers

### 1. Core types (`core/types.py`)

The vocabulary shared by every other layer:

- `Role` — who authored a message (`system`, `user`, `assistant`, `tool`).
- `Message` — a single turn in a conversation, optionally tagged with a
  tool-call id when it represents a tool result.
- `ToolCall` — a request from the LLM to invoke a named tool with a set of
  arguments.
- `ToolResult` — the outcome of executing a `ToolCall`, including whether it
  errored.
- `AgentStep` — one iteration of the ReAct loop: a thought, the action taken
  (if any), and the resulting observation.
- `AgentStatus` — the terminal or in-progress state of an agent run.

These are intentionally the *only* place shapes are defined. Every other
module imports from here rather than redefining similar-looking dicts.

### 2. Interfaces (`core/interfaces.py`)

Four abstract base classes, one per swappable component:

- **`BaseLLMClient`** — turns a list of `Message` into a new `Message`
  (`generate`) or a stream of text chunks (`stream`). A concrete
  implementation wraps a specific provider's SDK (OpenAI first; the
  interface does not assume any particular provider's request/response
  shape).
- **`BaseTool`** — exposes a `name`, a natural-language `description` (used
  in the LLM's tool-selection prompt), a JSON-schema `parameters_schema`
  (used for argument validation), and an `execute` coroutine that returns a
  `ToolResult`. Day 2 adds the concrete tool registry and the first tools
  (calculator, search, file access).
- **`BaseMemory`** — `add`, `get_history`, and `clear`. Deliberately minimal
  so that a conversation-buffer implementation and a Redis-backed episodic
  implementation can share the same call sites in the agent core. Day 3
  fills this in.
- **`BasePlanner`** — `create_plan` decomposes a high-level goal into an
  ordered list of sub-goals the agent can tackle one at a time. Day 5 fills
  this in; until then, an agent can run without a planner and treat the
  whole goal as a single step.

`BaseAgent` composes the four interfaces above behind a single `run(goal)`
entry point. Day 4 provides the concrete ReAct implementation.

### 3. Configuration (`config.py`)

A single `Settings` object (pydantic-settings `BaseSettings`) reads
everything environment-specific — API keys, model name, Redis URL,
iteration limits, log level — from environment variables or a `.env` file.
Nothing else in the codebase calls `os.environ` directly; if a new
component needs a setting, it gets added here so there is one place that
documents every knob the toolkit exposes.

### 4. Exceptions (`exceptions.py`)

A shallow hierarchy rooted at `AgentToolkitError`, so calling code can catch
`AgentToolkitError` broadly or a specific subtype
(`ToolExecutionError`, `ToolNotFoundError`, `MemoryBackendError`,
`PlanningError`, `LLMClientError`, `MaxIterationsExceededError`) narrowly.

## The ReAct loop (target shape, implemented Day 4)

```
while status is RUNNING and iterations < max_iterations:
    thought   = llm_client.generate(history + [goal])
    if thought proposes a tool call:
        result = tool_registry.execute(thought.tool_call)
        memory.add(observation built from result)
    else:
        status = FINISHED  # the LLM produced a final answer
    iterations += 1
```

Each iteration is captured as an `AgentStep` and appended to a run log, so a
completed or failed run can be inspected step-by-step rather than just as a
final answer.

## Why these choices

- **ABCs over `Protocol`.** The toolkit favors explicit `abc.ABC` +
  `@abstractmethod` for the four swappable components because it gives a
  clear, enforced contract and a natural place to hang shared helper methods
  later (e.g., retry/backoff wrapping in `BaseLLMClient`), at the cost of a
  small amount of extra ceremony compared to a structural `Protocol`.
- **pydantic v2 everywhere data crosses a boundary.** Validation errors
  surface immediately, with a clear message, rather than as a confusing
  `KeyError` three calls deep.
- **No premature provider lock-in.** `BaseLLMClient` does not leak
  OpenAI-specific concepts (e.g., function-calling wire format) into the
  agent core; the concrete OpenAI client is responsible for translating
  to and from the toolkit's own `Message`/`ToolCall` types.

## Non-goals (for now)

- Multi-agent orchestration (one agent, one goal, per run).
- A hosted UI (the CLI, added Day 7, is the only first-party UI).
- Fine-tuning or embedding-based retrieval — out of scope for this toolkit;
  it assumes an existing LLM API and focuses on the agent loop around it.
