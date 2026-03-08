"""Fast, explainable graders for continuous regression testing."""

from __future__ import annotations

import re
from typing import Callable, Iterable

from .cases import EvalCase, EvalResult


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def contains_expected(case: EvalCase) -> EvalResult:
    expected = normalize(case.expected)
    output = normalize(case.output)
    passed = bool(expected) and expected in output
    return EvalResult(case.id, "contains_expected", float(passed), passed, {"expected": expected})


def token_f1(case: EvalCase, threshold: float = 0.7) -> EvalResult:
    expected = normalize(case.expected).split()
    output = normalize(case.output).split()
    overlap = len(set(expected) & set(output))
    precision = overlap / len(set(output)) if output else 0.0
    recall = overlap / len(set(expected)) if expected else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvalResult(case.id, "token_f1", score, score >= threshold, {"precision": precision, "recall": recall})


def required_phrases(*phrases: str) -> Callable[[EvalCase], EvalResult]:
    normalized = tuple(normalize(phrase) for phrase in phrases)

    def grade(case: EvalCase) -> EvalResult:
        output = normalize(case.output)
        missing = [phrase for phrase in normalized if phrase not in output]
        score = 1 - len(missing) / len(normalized) if normalized else 1.0
        return EvalResult(case.id, "required_phrases", score, not missing, {"missing": missing})

    return grade


def aggregate(results: Iterable[EvalResult]) -> dict[str, float]:
    results = list(results)
    if not results:
        return {"score": 0.0, "pass_rate": 0.0, "count": 0}
    return {
        "score": sum(result.score for result in results) / len(results),
        "pass_rate": sum(result.passed for result in results) / len(results),
        "count": len(results),
    }
