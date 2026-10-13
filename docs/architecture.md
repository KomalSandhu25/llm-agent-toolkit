# Architecture

## Components

```
            +-----------+       +----------------+
 CLI / API  |  Agent    | <---> |  LLM client    |
 ---------> |  (ReAct)  |       +----------------+
            |           | <---> +----------------+
            |           |       |  ToolRegistry  |
            |           | <---> +----------------+
            |           |       |  Memory        |
            +-----+-----+       +----------------+
                  ^
                  |
            +-----+-----+
            |  Planner  |
            +-----------+
```

* **types** (`agent_toolkit.types`): Pydantic models shared by every layer
  (`Message`, `ToolCall`, `ToolResult`, `AgentStep`, `AgentResult`).
* **config** (`agent_toolkit.config`): `Settings` loaded from environment variables
  prefixed `AGENT_`, validated once and passed explicitly (no globals).
* **exceptions** (`agent_toolkit.exceptions`): a single `AgentToolkitError` hierarchy
  so callers can catch narrowly or broadly.
* **logging** (`agent_toolkit.logging_utils`): Rich-backed logging setup.

## Design principles

1. Explicit dependencies: components receive collaborators via constructors.
2. Everything crossing a boundary is a validated Pydantic model.
3. Tools never raise to the agent loop; failures become `ToolResult(is_error=True)`
   so the model can observe and recover.
4. The agent loop is bounded by `max_steps` and always terminates.
