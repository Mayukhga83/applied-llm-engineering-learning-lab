"""Second-stage reranking over a deliberately small candidate set."""

from __future__ import annotations

from typing import Callable, Iterable

from lexical import SearchResult, tokenize


def interaction_score(query: str, passage: str) -> float:
    """Transparent baseline approximating richer query-document interaction."""
    query_tokens = tokenize(query)
    passage_tokens = tokenize(passage)
    if not query_tokens:
        return 0.0
    passage_set = set(passage_tokens)
    unigram = sum(token in passage_set for token in query_tokens) / len(query_tokens)
    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    passage_bigrams = set(zip(passage_tokens, passage_tokens[1:]))
    bigram = len(query_bigrams & passage_bigrams) / (len(query_bigrams) or 1)
    return 0.75 * unigram + 0.25 * bigram


def rerank(
    query: str,
    candidates: Iterable[SearchResult],
    limit: int = 5,
    scorer: Callable[[str, str], float] = interaction_score,
) -> list[SearchResult]:
    rescored = []
    for candidate in candidates:
        passage = f"{candidate.document.title}. {candidate.document.text}"
        relevance = scorer(query, passage)
        combined = 0.85 * relevance + 0.15 * candidate.score
        rescored.append(SearchResult(candidate.document, combined))
    return sorted(rescored, key=lambda result: (-result.score, result.document.id))[:limit]
