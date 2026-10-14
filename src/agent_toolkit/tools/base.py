"""Abstract base class and result type for agent tools.

A tool is a named, described callable with a pydantic-validated argument
schema. The description and JSON schema are shown to the model so it can
choose and parameterise tools correctly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError


class ToolResult(BaseModel):
    """Outcome of a tool call.

    Args:
        output: Text returned to the model as the observation.
        is_error: True when the call failed; the model can then retry.
    """

    output: str
    is_error: bool = False


class Tool(ABC):
    """Base class for all tools.

    Subclasses set ``name``, ``description`` and ``args_schema`` and
    implement :meth:`run`.

    Example:
        >>> class Echo(Tool):
        ...     name = "echo"
        ...     description = "Repeat the input."
        ...     class Args(BaseModel):
        ...         text: str
        ...     args_schema = Args
        ...     def run(self, args):
        ...         return ToolResult(output=args.text)
        >>> Echo().invoke({"text": "hi"}).output
        'hi'
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, args: Any) -> ToolResult:
        """Execute the tool with validated arguments.

        Args:
            args: Instance of ``args_schema``.

        Returns:
            The tool result.
        """

    def invoke(self, raw_args: dict[str, Any]) -> ToolResult:
        """Validate raw arguments, run the tool, and never raise.

        Args:
            raw_args: Unvalidated arguments from the model.

        Returns:
            A :class:`ToolResult`; validation or runtime failures are
            returned with ``is_error=True`` so the agent can recover.
        """
        try:
            args = self.args_schema.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult(output=f"Invalid arguments for {self.name}: {exc}", is_error=True)
        try:
            return self.run(args)
        except Exception as exc:  # noqa: BLE001 - tools must not crash the loop
            return ToolResult(output=f"{self.name} failed: {exc}", is_error=True)

    def spec(self) -> dict[str, Any]:
        """Describe the tool for inclusion in the model prompt.

        Returns:
            Dict with ``name``, ``description`` and JSON ``parameters``.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }
