"""Bounded parsing and repair for structured model responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class StructuredOutputError(ValueError):
    pass


@dataclass(frozen=True)
class ValidationResult(Generic[T]):
    value: T
    repaired: bool


def extract_json_object(text: str) -> str:
    """Extract one top-level JSON object, accepting a fenced response."""
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise StructuredOutputError("response does not contain a JSON object")
    return candidate[start : end + 1]


def parse_and_validate(text: str, validator: Callable[[dict], T]) -> ValidationResult[T]:
    try:
        raw = json.loads(text)
        repaired = False
    except json.JSONDecodeError:
        try:
            raw = json.loads(extract_json_object(text))
            repaired = True
        except (json.JSONDecodeError, StructuredOutputError) as error:
            raise StructuredOutputError(f"invalid JSON response: {error}") from error
    if not isinstance(raw, dict):
        raise StructuredOutputError("top-level response must be an object")
    try:
        value = validator(raw)
    except (TypeError, ValueError, KeyError) as error:
        raise StructuredOutputError(f"response failed schema validation: {error}") from error
    return ValidationResult(value=value, repaired=repaired)


def generate_with_retries(generate: Callable[[str], str], prompt: str, validator: Callable[[dict], T], max_attempts: int = 2) -> ValidationResult[T]:
    """Retry invalid model output with a concise validation instruction."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    current_prompt = prompt
    last_error: StructuredOutputError | None = None
    for _ in range(max_attempts):
        output = generate(current_prompt)
        try:
            return parse_and_validate(output, validator)
        except StructuredOutputError as error:
            last_error = error
            current_prompt = f"{prompt}\nReturn only valid JSON. Previous error: {error}"
    raise StructuredOutputError(f"structured generation failed after {max_attempts} attempts: {last_error}")
