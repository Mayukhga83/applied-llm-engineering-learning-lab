"""Offline structured-generation walkthrough."""

import json

from schemas import GroundedAnswer
from tools import ToolDefinition, ToolRegistry
from validation import parse_and_validate


def lookup_source(source_id: str) -> str:
    sources = {"rag-notes": "Hybrid retrieval combines lexical and semantic signals."}
    return sources.get(source_id, "source not found")


def main() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="lookup_source",
            description="Retrieve a source by identifier",
            parameters={"source_id": str},
            handler=lookup_source,
        )
    )
    quote = registry.invoke("lookup_source", {"source_id": "rag-notes"})
    simulated_model_output = json.dumps(
        {
            "answer": "Hybrid retrieval uses more than one retrieval signal.",
            "citations": [{"source_id": "rag-notes", "quote": quote}],
            "confidence": 0.95,
        }
    )
    result = parse_and_validate(simulated_model_output, GroundedAnswer.from_mapping)
    print(json.dumps(result.value.to_dict(), indent=2))


if __name__ == "__main__":
    main()
