"""Evaluate retrievers against the checked-in relevance judgments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Protocol

from corpus import load_documents, load_questions
from dense import DenseRetriever
from fusion import HybridRetriever
from lexical import BM25Retriever, SearchResult
from reranking import rerank


class Retriever(Protocol):
    def search(self, query: str, limit: int = 5) -> list[SearchResult]: ...


class RerankedRetriever:
    def __init__(self, base: Retriever) -> None:
        self.base = base

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return rerank(query, self.base.search(query, max(10, limit * 2)), limit)


def evaluate(retriever: Retriever, questions, limit: int = 3) -> dict[str, float]:
    recalls, reciprocal_ranks = [], []
    for question in questions:
        ids = [result.document.id for result in retriever.search(question.query, limit)]
        relevant = set(question.relevant_ids)
        recalls.append(len(relevant & set(ids)) / len(relevant))
        ranks = [index + 1 for index, doc_id in enumerate(ids) if doc_id in relevant]
        reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
    return {
        f"recall@{limit}": round(sum(recalls) / len(recalls), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
    }


def run(data_dir: Path) -> dict[str, dict[str, float]]:
    documents = load_documents(data_dir / "documents.jsonl")
    questions = load_questions(data_dir / "questions.jsonl")
    lexical = BM25Retriever(documents)
    dense = DenseRetriever(documents)
    hybrid = HybridRetriever(lexical, dense)
    systems = {"bm25": lexical, "dense_hashing_baseline": dense, "hybrid_rrf": hybrid, "hybrid_reranked": RerankedRetriever(hybrid)}
    return {name: evaluate(retriever, questions) for name, retriever in systems.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
