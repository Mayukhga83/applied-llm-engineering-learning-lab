"""Explicit response schemas for grounded research answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Citation:
    source_id: str
    quote: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Citation":
        source_id = value.get("source_id")
        quote = value.get("quote")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("citation.source_id must be a non-empty string")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("citation.quote must be a non-empty string")
        return cls(source_id=source_id.strip(), quote=quote.strip())


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: tuple[Citation, ...]
    confidence: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GroundedAnswer":
        answer = value.get("answer")
        raw_citations = value.get("citations")
        confidence = value.get("confidence")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string")
        if not isinstance(raw_citations, Sequence) or isinstance(raw_citations, (str, bytes)):
            raise ValueError("citations must be an array")
        citations = tuple(Citation.from_mapping(item) for item in raw_citations)
        if not citations:
            raise ValueError("at least one citation is required")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return cls(answer=answer.strip(), citations=citations, confidence=float(confidence))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GROUNDED_ANSWER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "citations", "confidence"],
    "properties": {
        "answer": {"type": "string", "minLength": 1},
        "citations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source_id", "quote"],
                "properties": {
                    "source_id": {"type": "string", "minLength": 1},
                    "quote": {"type": "string", "minLength": 1},
                },
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
