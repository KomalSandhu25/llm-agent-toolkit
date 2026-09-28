"""Safe arithmetic calculator tool (AST-based, no eval)."""

from __future__ import annotations

import ast
import operator
from typing import Any, Callable

from agent_toolkit.tools.base import Tool, ToolError

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ToolError("Unsupported expression")


class CalculatorTool(Tool):
    """Evaluate arithmetic expressions safely.

    Example:
        >>> CalculatorTool().run(expression="2 + 3 * 4")
        '14'
    """

    name = "calculator"
    description = "Evaluate an arithmetic expression, e.g. '2 + 3 * 4'."

    def run(self, **kwargs: Any) -> str:
        """Evaluate ``expression``; returns the numeric result as a string."""
        expr = str(kwargs.get("expression", "")).strip()
        if not expr:
            raise ToolError("Missing 'expression'")
        try:
            result = _eval(ast.parse(expr, mode="eval").body)
        except ZeroDivisionError:
            raise ToolError("Division by zero") from None
        except SyntaxError:
            raise ToolError("Invalid expression syntax") from None
        return str(int(result)) if float(result).is_integer() else str(result)
