"""Tests for core trace types."""

from agent_toolkit.types import Step, StepKind, Trace


def test_trace_final_answer() -> None:
    t = Trace(task="x")
    assert t.final_answer is None
    t.add(Step(StepKind.THOUGHT, "thinking"))
    t.add(Step(StepKind.FINAL, "done"))
    assert t.final_answer == "done"
