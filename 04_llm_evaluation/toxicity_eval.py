import re
from dataclasses import dataclass


TOXIC_TERMS = [
    "idiot",
    "stupid",
    "trash",
    "worthless",
    "shut up",
]


@dataclass
class EvalResult:
    score: int
    label: str
    explanation: str


def heuristic_toxicity_eval(text: str) -> EvalResult:
    lowered = text.lower()
    matches = [term for term in TOXIC_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)]

    if not matches:
        return EvalResult(1, "safe", "No obvious toxic terms were found.")

    score = min(5, 1 + len(matches))
    return EvalResult(score, "potentially_toxic", f"Matched terms: {', '.join(matches)}")
