"""Rank fusion utilities for combining heterogeneous retrievers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Protocol

from lexical import SearchResult


class Retriever(Protocol):
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        ...


def reciprocal_rank_fusion(result_lists: Iterable[list[SearchResult]], k: int = 60, limit: int = 5) -> list[SearchResult]:
    scores: dict[str, float] = defaultdict(float)
    documents = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.document.id] += 1.0 / (k + rank)
            documents[result.document.id] = result.document
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:limit]
    return [SearchResult(documents[doc_id], scores[doc_id]) for doc_id in ordered]


class HybridRetriever:
    def __init__(self, *retrievers: Retriever, rrf_k: int = 60) -> None:
        if len(retrievers) < 2:
            raise ValueError("hybrid retrieval needs at least two retrievers")
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        candidate_limit = max(limit * 2, 10)
        lists = [retriever.search(query, candidate_limit) for retriever in self.retrievers]
        return reciprocal_rank_fusion(lists, self.rrf_k, limit)
