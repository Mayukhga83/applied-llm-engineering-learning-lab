"""Retrieval metrics that preserve per-case failure details."""

from __future__ import annotations

import math
from typing import Mapping

from .cases import EvalCase, EvalResult


def recall_at_k(case: EvalCase, k: int = 5) -> EvalResult:
    relevant = set(case.relevant_ids)
    retrieved = set(case.retrieved_ids[:k])
    score = len(relevant & retrieved) / len(relevant) if relevant else 0.0
    return EvalResult(case.id, f"recall@{k}", score, score == 1.0, {"missed": sorted(relevant - retrieved)})


def reciprocal_rank(case: EvalCase) -> EvalResult:
    relevant = set(case.relevant_ids)
    rank = next((index for index, item in enumerate(case.retrieved_ids, start=1) if item in relevant), None)
    score = 1 / rank if rank else 0.0
    return EvalResult(case.id, "reciprocal_rank", score, score > 0, {"rank": rank})


def ndcg_at_k(case: EvalCase, k: int = 5) -> EvalResult:
    relevant = set(case.relevant_ids)
    gains = [1.0 if item in relevant else 0.0 for item in case.retrieved_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(relevant), k)))
    score = dcg / ideal if ideal else 0.0
    return EvalResult(case.id, f"ndcg@{k}", score, score >= 0.8, {"dcg": dcg, "ideal_dcg": ideal})


def context_precision(case: EvalCase) -> EvalResult:
    relevant = set(case.relevant_ids)
    retrieved = set(case.retrieved_ids)
    score = len(relevant & retrieved) / len(retrieved) if retrieved else 0.0
    return EvalResult(case.id, "context_precision", score, score >= 0.5, {"irrelevant": sorted(retrieved - relevant)})


def citation_validity(case: EvalCase, source_text: Mapping[str, str]) -> EvalResult:
    """Check structured citations stored in metadata against source text."""
    citations = list(case.metadata.get("citations", []))
    valid = 0
    failures = []
    for citation in citations:
        source_id, quote = citation.get("source_id", ""), citation.get("quote", "")
        source = source_text.get(source_id, "")
        if quote and quote.casefold() in source.casefold():
            valid += 1
        else:
            failures.append(source_id)
    score = valid / len(citations) if citations else 0.0
    return EvalResult(case.id, "citation_validity", score, bool(citations) and score == 1.0, {"invalid_sources": failures})
