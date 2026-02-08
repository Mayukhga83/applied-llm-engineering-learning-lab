"""Typed local tool registry with argument validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, type]
    handler: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        type_names = {str: "string", int: "integer", float: "number", bool: "boolean"}
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": list(self.parameters),
                "properties": {
                    key: {"type": type_names.get(value, "object")}
                    for key, value in self.parameters.items()
                },
            },
        }

    def invoke(self, arguments: Mapping[str, Any]) -> Any:
        missing = set(self.parameters) - set(arguments)
        extra = set(arguments) - set(self.parameters)
        if missing or extra:
            raise ValueError(f"invalid arguments; missing={sorted(missing)}, extra={sorted(extra)}")
        for name, expected_type in self.parameters.items():
            if not isinstance(arguments[name], expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")
        return self.handler(**arguments)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in sorted(self._tools)]

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Any:
        try:
            tool = self._tools[name]
        except KeyError as error:
            raise KeyError(f"unknown tool: {name}") from error
        return tool.invoke(arguments)
