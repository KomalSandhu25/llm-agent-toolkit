"""Tests for the tool registry and calculator."""

import pytest

from agent_toolkit.tools import ToolError, ToolRegistry
from agent_toolkit.tools.calculator import CalculatorTool


def test_calculator_precedence() -> None:
    assert CalculatorTool().run(expression="2 + 3 * 4") == "14"


def test_calculator_float() -> None:
    assert CalculatorTool().run(expression="1 / 4") == "0.25"


def test_calculator_rejects_code() -> None:
    with pytest.raises(ToolError):
        CalculatorTool().run(expression="__import__('os').system('ls')")


def test_calculator_div_zero() -> None:
    with pytest.raises(ToolError):
        CalculatorTool().run(expression="1/0")


def test_registry_register_and_call() -> None:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    assert "calculator" in reg
    assert reg.call("calculator", expression="6*7") == "42"


def test_registry_duplicate_and_unknown() -> None:
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    with pytest.raises(ValueError):
        reg.register(CalculatorTool())
    with pytest.raises(ToolError):
        reg.get("nope")
