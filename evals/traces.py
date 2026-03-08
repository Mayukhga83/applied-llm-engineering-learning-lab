"""Structural graders for agent execution traces."""

from __future__ import annotations

from collections import Counter

from .cases import EvalCase, EvalResult


def trace_structure(case: EvalCase, max_steps: int = 8) -> EvalResult:
    events = [str(event.get("event", "")) for event in case.trace]
    counts = Counter(events)
    failures = []
    if not events:
        failures.append("empty_trace")
    if counts["decision"] == 0:
        failures.append("missing_decision")
    if counts["tool_result"] > counts["decision"]:
        failures.append("orphan_tool_result")
    if len(case.trace) > max_steps * 2 + 1:
        failures.append("step_budget_exceeded")
    if any(event.get("event") == "tool_call" and not event.get("tool") for event in case.trace):
        failures.append("unnamed_tool")
    score = max(0.0, 1.0 - len(failures) * 0.25)
    return EvalResult(case.id, "trace_structure", score, not failures, {"failures": failures, "events": dict(counts)})


def tool_argument_quality(case: EvalCase) -> EvalResult:
    calls = [event for event in case.trace if event.get("event") in {"tool_call", "decision"} and event.get("tool")]
    invalid = []
    for index, call in enumerate(calls):
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            invalid.append(index)
        elif any(value in (None, "") for value in arguments.values()):
            invalid.append(index)
    score = 1 - len(invalid) / len(calls) if calls else 0.0
    return EvalResult(case.id, "tool_argument_quality", score, bool(calls) and not invalid, {"invalid_calls": invalid})


def excessive_retries(case: EvalCase, allowed_per_tool: int = 2) -> EvalResult:
    calls = [str(event.get("tool")) for event in case.trace if event.get("event") in {"tool_call", "decision"} and event.get("tool")]
    counts = Counter(calls)
    excessive = {tool: count for tool, count in counts.items() if count > allowed_per_tool}
    score = 1.0 if not excessive else 0.0
    return EvalResult(case.id, "excessive_retries", score, not excessive, {"excessive": excessive})
