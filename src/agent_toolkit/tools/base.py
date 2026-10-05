"""Base class for pluggable agent tools.

A tool declares a name, a description, and a Pydantic model for its
arguments. The model's JSON schema is exposed to the LLM through the
OpenAI function-calling format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class Tool(ABC):
    """Abstract base class for tools the agent may call.

    Subclasses set ``name``, ``description`` and ``args_schema`` and
    implement :meth:`run`.

    Example:
        >>> class Echo(Tool):
        ...     name = "echo"
        ...     description = "Echo the given text."
        ...     class args_schema(BaseModel):
        ...         text: str
        ...     def run(self, **kwargs: Any) -> str:
        ...         return str(kwargs["text"])
        >>> Echo().invoke({"text": "hi"})
        'hi'
    """

    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool with validated keyword arguments.

        Args:
            **kwargs: Arguments matching ``args_schema``.

        Returns:
            A string observation to feed back to the model.
        """

    def invoke(self, arguments: dict[str, Any]) -> str:
        """Validate raw ``arguments`` against ``args_schema`` then run.

        Args:
            arguments: JSON-decoded arguments from the model.

        Returns:
            The tool's string output.

        Raises:
            pydantic.ValidationError: If ``arguments`` do not match the schema.
        """
        parsed = self.args_schema.model_validate(arguments)
        return self.run(**parsed.model_dump())

    def to_openai_schema(self) -> dict[str, Any]:
        """Render this tool as an OpenAI function-calling schema.

        Returns:
            A dict of the form ``{"type": "function", "function": {...}}``.
        """
        schema = self.args_schema.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }
