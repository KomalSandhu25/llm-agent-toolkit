"""Core data types and interfaces shared across the agent toolkit.

This subpackage is the toolkit's vocabulary: :mod:`core.types` defines the
data that flows between components, and :mod:`core.interfaces` defines the
abstract contracts each swappable component (LLM client, tool, memory
backend, planner, agent) must satisfy.
"""

from __future__ import annotations
