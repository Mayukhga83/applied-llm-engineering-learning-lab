from dataclasses import dataclass
from typing import Literal

Label = Literal["safe", "toxic", "biased", "unsafe"]


@dataclass
class SafetyOutput:
    label: Label
    explanation: str
    safer_rewrite: str


def classify_and_rewrite(text: str) -> SafetyOutput:
    lowered = text.lower()

    toxic_terms = ["idiot", "stupid", "worthless", "trash"]
    biased_patterns = ["women are not", "men are not", "people from", "naturally bad"]

    if any(term in lowered for term in toxic_terms):
        return SafetyOutput(
            label="toxic",
            explanation="The text contains insulting or demeaning language.",
            safer_rewrite="I disagree, but I can explain the issue respectfully.",
        )

    if any(pattern in lowered for pattern in biased_patterns):
        return SafetyOutput(
            label="biased",
            explanation="The text appears to make a broad stereotype about a group.",
            safer_rewrite="Ability and behavior vary by individual and should not be generalized to a whole group.",
        )

    return SafetyOutput(
        label="safe",
        explanation="No obvious toxicity or bias was detected.",
        safer_rewrite=text,
    )
