"""Deterministic hashing-vector baseline behind a dense retriever interface."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from corpus import Document
from lexical import SearchResult, tokenize

Vector = Sequence[float]


def hashing_embedding(text: str, dimensions: int = 128) -> tuple[float, ...]:
    """Create an offline feature-hashing vector; replaceable with real embeddings."""
    vector = [0.0] * dimensions
    tokens = tokenize(text)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(value / norm for value in vector)


def cosine(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    return sum(a * b for a, b in zip(left, right))


@dataclass
class DenseRetriever:
    documents: list[Document]
    embed: Callable[[str], Vector] = hashing_embedding

    def __init__(self, documents: Iterable[Document], embed: Callable[[str], Vector] = hashing_embedding) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("documents must not be empty")
        self.embed = embed
        self._vectors = [embed(f"{doc.title} {doc.text}") for doc in self.documents]

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        query_vector = self.embed(query)
        results = [SearchResult(doc, cosine(query_vector, vector)) for doc, vector in zip(self.documents, self._vectors)]
        return sorted(results, key=lambda result: (-result.score, result.document.id))[:limit]
