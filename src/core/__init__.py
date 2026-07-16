"""Framework-wide types and exceptions.

Every other module in ``llm-agent-toolkit`` (tools, memory, the agent loop,
planning, and the API/CLI layers) is built on top of the shared vocabulary
defined here: :mod:`src.core.types` and :mod:`src.core.exceptions`. Keeping
these in one place, defined before any concrete tool or backend exists,
lets later modules be implemented against a stable contract.
"""
