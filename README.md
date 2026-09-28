# llm-agent-toolkit

A ReAct-pattern LLM agent framework with pluggable tools, persistent memory,
multi-step planning, and an interactive CLI.

> Status: Day 1 of 7 — project scaffold and core abstractions.

## Planned architecture

```
            +-----------+
 task ----> |  Agent    |  Thought -> Action -> Observation loop (ReAct)
            +-----+-----+
                  |
     +------------+-------------+
     |            |             |
+----v---+  +-----v-----+  +----v----+
| Tools  |  |  Memory   |  | Planner |
| (base, |  | (Redis)   |  | (multi- |
| calc,..)|  |           |  |  step)  |
+--------+  +-----------+  +---------+
                  |
            +-----v-----+
            | CLI / API |  (rich, FastAPI)
            +-----------+
```

## Implemented so far

- `agent_toolkit.config` — typed settings via pydantic-settings
- `agent_toolkit.types` — `Step`, `StepKind`, `Trace`
- `agent_toolkit.tools` — `Tool` ABC, `ToolRegistry`, safe `CalculatorTool`

## Development

```bash
pip install -e ".[dev]"
pytest
```
