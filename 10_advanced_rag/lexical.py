"""Compact BM25 implementation for inspecting lexical retrieval behavior."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from corpus import Document

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float


class BM25Retriever:
    def __init__(self, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("documents must not be empty")
        self.k1, self.b = k1, b
        self.tokens = [tokenize(f"{doc.title} {doc.text}") for doc in self.documents]
        self.counts = [Counter(tokens) for tokens in self.tokens]
        self.avg_length = sum(map(len, self.tokens)) / len(self.tokens)
        self.document_frequency = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    def _idf(self, term: str) -> float:
        count = self.document_frequency[term]
        return math.log(1 + (len(self.documents) - count + 0.5) / (count + 0.5))

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        terms = tokenize(query)
        scored: list[SearchResult] = []
        for doc, counts, tokens in zip(self.documents, self.counts, self.tokens):
            score = 0.0
            for term in terms:
                frequency = counts[term]
                denominator = frequency + self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_length)
                if denominator:
                    score += self._idf(term) * frequency * (self.k1 + 1) / denominator
            scored.append(SearchResult(doc, score))
        return sorted(scored, key=lambda result: (-result.score, result.document.id))[:limit]
