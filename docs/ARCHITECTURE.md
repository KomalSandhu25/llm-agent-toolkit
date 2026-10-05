# Architecture

```
            +-------------------+
 user ----> |  CLI / FastAPI    |   (later milestone)
            +---------+---------+
                      |
              +-------v--------+      +-----------------+
              |  Agent (ReAct) | ---> |   Planner       |
              +---+--------+---+      +-----------------+
                  |        |
        +---------v--+  +--v-----------+
        | LLMClient  |  | ToolRegistry |  <- Tool, Tool, Tool ...   (day 1)
        | (OpenAI)   |  +--------------+
        +------------+
                  |
              +---v--------+
              |  Memory    |  Redis-backed
              +------------+
```

## ReAct loop

1. Build the prompt from memory + the user goal.
2. Ask the `LLMClient` for the next message.
3. If it contains `tool_calls`, dispatch each through `ToolRegistry.execute`
   and append the observations; otherwise the content is the final answer.
4. Repeat until a final answer or `agent_max_steps` is reached.

Tool failures never crash the loop: they are returned as error observations
so the model can self-correct.

## Design decisions

- **Protocols over base classes** for `LLMClient`, `Memory`, `Planner` so
  tests can inject trivial fakes.
- **Pydantic arg schemas** double as validation and the JSON schema sent to
  the model.
- **Provider-agnostic messages**; adapters translate to the OpenAI format.
