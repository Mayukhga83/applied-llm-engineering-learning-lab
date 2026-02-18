"""Loading contracts for the versioned retrieval corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Question:
    id: str
    query: str
    relevant_ids: tuple[str, ...]


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_documents(path: Path) -> list[Document]:
    return [Document(item["id"], item["title"], item["text"], tuple(item.get("tags", []))) for item in _jsonl(path)]


def load_questions(path: Path) -> list[Question]:
    return [Question(item["id"], item["query"], tuple(item["relevant_ids"])) for item in _jsonl(path)]
