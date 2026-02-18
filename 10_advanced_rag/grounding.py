"""Simple query decomposition and citation support checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from corpus import Document
from lexical import tokenize


def decompose_query(query: str) -> list[str]:
    """Split explicit multi-part questions without inventing hidden subproblems."""
    normalized = re.sub(r"\s+", " ", query).strip(" ?")
    parts = re.split(r"\s+(?:and|versus|vs\.?|then)\s+", normalized, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


@dataclass(frozen=True)
class CitationCheck:
    source_id: str
    quote_found: bool
    claim_overlap: float
    supported: bool


def verify_citation(claim: str, source_id: str, quote: str, documents: Mapping[str, Document], minimum_overlap: float = 0.35) -> CitationCheck:
    document = documents.get(source_id)
    if document is None:
        return CitationCheck(source_id, False, 0.0, False)
    quote_found = quote.casefold() in document.text.casefold()
    claim_terms = set(tokenize(claim))
    evidence_terms = set(tokenize(quote))
    overlap = len(claim_terms & evidence_terms) / (len(claim_terms) or 1)
    return CitationCheck(source_id, quote_found, overlap, quote_found and overlap >= minimum_overlap)


def citation_coverage(checks: Iterable[CitationCheck]) -> float:
    checks = list(checks)
    return sum(check.supported for check in checks) / len(checks) if checks else 0.0
