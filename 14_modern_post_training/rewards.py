"""Deterministic rewards for small verifiable-reasoning experiments."""

from __future__ import annotations

import json
import math
import re
from typing import Callable, Iterable

Reward = Callable[[str, str], float]


def exact_answer_reward(completion: str, reference: str) -> float:
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9.-]+", value.casefold()))
    return float(normalize(completion) == normalize(reference))


def numeric_answer_reward(completion: str, reference: str, tolerance: float = 1e-6) -> float:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", completion)
    expected = re.findall(r"[-+]?\d+(?:\.\d+)?", reference)
    if not numbers or not expected:
        return 0.0
    return float(math.isclose(float(numbers[-1]), float(expected[-1]), abs_tol=tolerance, rel_tol=tolerance))


def structured_answer_reward(completion: str, reference: str = "") -> float:
    try:
        value = json.loads(completion)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(value, dict):
        return 0.0
    required = {"answer", "citations"}
    if not required <= set(value) or not isinstance(value["citations"], list):
        return 0.25
    return 1.0 if value["answer"] and value["citations"] else 0.5


def weighted_reward(completion: str, reference: str, rewards: Iterable[tuple[Reward, float]]) -> float:
    rewards = list(rewards)
    total_weight = sum(weight for _, weight in rewards)
    if total_weight <= 0:
        raise ValueError("reward weights must sum to a positive value")
    score = sum(reward(completion, reference) * weight for reward, weight in rewards)
    return score / total_weight


def batch_reward(completions: list[str], references: list[str], reward: Reward) -> list[float]:
    if len(completions) != len(references):
        raise ValueError("completions and references must have equal length")
    return [reward(completion, reference) for completion, reference in zip(completions, references)]
