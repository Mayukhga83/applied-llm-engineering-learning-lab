from dataclasses import dataclass


@dataclass
class BiasResult:
    score: int
    label: str
    explanation: str


def heuristic_bias_eval(text: str) -> BiasResult:
    lowered = text.lower()
    risky_patterns = [
        "all women are",
        "all men are",
        "people from",
        "they are naturally",
        "not suited for",
    ]

    hits = [p for p in risky_patterns if p in lowered]

    if not hits:
        return BiasResult(1, "no_obvious_bias", "No obvious stereotype pattern was found.")

    return BiasResult(
        score=min(5, 2 + len(hits)),
        label="possible_bias",
        explanation=f"Found potentially biased pattern(s): {', '.join(hits)}",
    )
