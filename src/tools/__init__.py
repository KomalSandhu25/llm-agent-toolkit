"""Pluggable tool system for the agent.

A ``Tool`` is anything the agent can invoke to interact with the outside
world (a calculator, a web search, a file reader, ...). This package defines
the common ``Tool`` interface (``base.py``); concrete tool implementations
and the ``ToolRegistry`` that the agent loop queries at runtime are added on
day 2.
"""

from src.tools.base import Tool, ToolResult

__all__ = ["Tool", "ToolResult"]
