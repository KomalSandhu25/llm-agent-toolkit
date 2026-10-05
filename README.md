# llm-agent-toolkit

A ReAct-pattern LLM agent framework with pluggable tools, persistent memory,
multi-step planning, and an interactive CLI.

> Status: in active development. The first milestone delivers the project
> scaffold, core schemas, and the tool abstraction layer.

## Features (planned)

- Pluggable tools with Pydantic-validated arguments
- ReAct reasoning loop with bounded steps and error recovery
- Redis-backed persistent conversation memory
- Multi-step planner
- Rich interactive CLI and FastAPI server

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
```

## Defining a tool

```python
from typing import Any
from pydantic import BaseModel
from agent_toolkit import Tool, ToolRegistry, ToolCall

class Args(BaseModel):
    text: str

class Echo(Tool):
    name = "echo"
    description = "Echo the given text."
    args_schema = Args

    def run(self, **kwargs: Any) -> str:
        return str(kwargs["text"])

registry = ToolRegistry([Echo()])
print(registry.execute(ToolCall(id="1", name="echo", arguments={"text": "hi"})).output)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.
