"""Model-judge adapter with label calibration and structured outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .cases import EvalCase, EvalResult

Judge = Callable[[str], Mapping[str, object]]


@dataclass(frozen=True)
class CalibrationExample:
    case: EvalCase
    human_passed: bool


def build_prompt(case: EvalCase, rubric: str) -> str:
    return (
        "Evaluate the response using only the rubric and supplied reference. "
        "Return an object with score from 0 to 1 and a short reason.\n"
        f"Rubric: {rubric}\nInput: {case.input}\nReference: {case.expected}\nResponse: {case.output}"
    )


def grade_with_model(case: EvalCase, judge: Judge, rubric: str, threshold: float = 0.7) -> EvalResult:
    raw = judge(build_prompt(case, rubric))
    score = raw.get("score")
    reason = raw.get("reason", "")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
        raise ValueError("judge score must be a number between 0 and 1")
    if not isinstance(reason, str):
        raise ValueError("judge reason must be a string")
    return EvalResult(case.id, "model_judge", float(score), score >= threshold, {"reason": reason})


def calibration_report(examples: Iterable[CalibrationExample], judge: Judge, rubric: str, threshold: float = 0.7) -> dict[str, float]:
    examples = list(examples)
    if not examples:
        raise ValueError("calibration needs labeled examples")
    predictions = [grade_with_model(example.case, judge, rubric, threshold).passed for example in examples]
    labels = [example.human_passed for example in examples]
    accuracy = sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)
    positive_agreement = sum(prediction and label for prediction, label in zip(predictions, labels))
    predicted_or_labeled = sum(predictions) + sum(labels)
    f1 = 2 * positive_agreement / predicted_or_labeled if predicted_or_labeled else 1.0
    return {"accuracy": accuracy, "positive_f1": f1, "count": len(labels)}
