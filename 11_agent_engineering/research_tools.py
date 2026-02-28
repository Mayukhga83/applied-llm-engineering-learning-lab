"""Typed, deterministic research tools for agent exercises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: Mapping[str, type]
    handler: Callable[..., Any]
    risk: str = "read"
    requires_approval: bool = False

    def invoke(self, arguments: Mapping[str, Any]) -> Any:
        if set(arguments) != set(self.parameters):
            raise ValueError(f"{self.name} requires arguments {sorted(self.parameters)}")
        for key, expected in self.parameters.items():
            if not isinstance(arguments[key], expected):
                raise TypeError(f"{key} must be {expected.__name__}")
        return self.handler(**arguments)

    def mcp_schema(self) -> dict[str, Any]:
        types = {str: "string", int: "integer", float: "number", bool: "boolean"}
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": list(self.parameters),
                "properties": {name: {"type": types.get(kind, "object")} for name, kind in self.parameters.items()},
            },
        }


def build_research_tools(documents: Mapping[str, str]) -> dict[str, AgentTool]:
    def search(query: str) -> list[dict[str, str]]:
        terms = set(query.lower().split())
        ranked = sorted(documents.items(), key=lambda item: -len(terms & set(item[1].lower().split())))
        return [{"id": doc_id, "text": text} for doc_id, text in ranked[:3]]

    def read_document(document_id: str) -> str:
        if document_id not in documents:
            raise KeyError(f"unknown document: {document_id}")
        return documents[document_id]

    def save_note(title: str, body: str) -> dict[str, str]:
        return {"status": "proposed", "title": title, "body": body}

    return {
        "search": AgentTool("search", "Search the learning corpus", {"query": str}, search),
        "read_document": AgentTool("read_document", "Read one document", {"document_id": str}, read_document),
        "save_note": AgentTool("save_note", "Propose a persistent research note", {"title": str, "body": str}, save_note, "write", True),
    }
