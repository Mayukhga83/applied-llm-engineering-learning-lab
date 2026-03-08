"""Shared evaluation data contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    expected: str = ""
    output: str = ""
    relevant_ids: tuple[str, ...] = ()
    retrieved_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trace: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "EvalCase":
        return cls(
            id=str(item["id"]),
            input=str(item["input"]),
            expected=str(item.get("expected", "")),
            output=str(item.get("output", "")),
            relevant_ids=tuple(item.get("relevant_ids", [])),
            retrieved_ids=tuple(item.get("retrieved_ids", [])),
            metadata=dict(item.get("metadata", {})),
            trace=tuple(item.get("trace", [])),
        )


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    grader: str
    score: float
    passed: bool
    details: Mapping[str, Any] = field(default_factory=dict)


def load_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8") as source:
        return [EvalCase.from_mapping(json.loads(line)) for line in source if line.strip()]
