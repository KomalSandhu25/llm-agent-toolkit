# Architecture

## The ReAct loop

The agent alternates between reasoning and acting until it can answer:

```
        +-----------+      thought + action       +-------+
 task ->|   LLM     |---------------------------->| Tool  |
        +-----------+                             +-------+
              ^                                       |
              |            observation                |
              +---------------------------------------+
```

1. **Thought** - the model explains what it intends to do next.
2. **Action** - the model selects a tool and arguments (a `ToolCall`).
3. **Observation** - the tool runs; its `ToolResult` is fed back to the model.
4. The loop ends with a **final answer** or when `max_steps` is reached.

Each iteration is recorded as an `AgentStep`, giving a complete, serializable trace.

## Components

| Component | Contract | Responsibility |
|-----------|----------|----------------|
| LLM client | `LLMClient` (Protocol) | Turn a message list into the next assistant `Message` |
| Tools | `BaseTool` (ABC) | Declare a name, description and JSON schema; execute with validated args |
| Memory | `BaseMemory` (ABC) | Store and retrieve `Message`s across turns and sessions |
| Config | `AgentConfig` | Model name, step limits, temperature, memory settings |

## Error model

All library errors derive from `AgentToolkitError`, so callers can catch one type:

- `ConfigurationError` - invalid or missing configuration
- `ToolNotFoundError` / `ToolExecutionError` - tool lookup and runtime failures
- `MemoryBackendError` - persistence failures
- `MaxStepsExceededError` - the loop did not converge

## Roadmap

1. Scaffold and contracts (this milestone)
2. Tool registry with calculator, search, file tools
3. Conversation and episodic memory backends
4. ReAct agent core
5. Multi-step planner and executor
6. REST + streaming WebSocket API
7. Rich CLI and demo
