"""Content labeling and heuristic triage for injection-focused exercises."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ContentEnvelope:
    content: str
    origin: str
    trusted: bool = False
    content_type: str = "text/plain"

    def for_model(self) -> str:
        trust = "trusted" if self.trusted else "UNTRUSTED"
        return f"<{trust}_CONTENT origin={self.origin!r}>\n{self.content}\n</{trust}_CONTENT>"


@dataclass(frozen=True)
class TriageResult:
    action: str
    matched_rules: tuple[str, ...]


RULES = {
    "instruction_override": re.compile(
        r"(?:\b(ignore|override|disregard)\b.{0,40}\b(instruction|system|previous)\b|\b(system|instruction)\b.{0,20}\boverride\b)",
        re.IGNORECASE,
    ),
    "secret_request": re.compile(r"\b(api[_ -]?key|token|password|system prompt|credential)\b", re.IGNORECASE),
    "external_transfer": re.compile(r"\b(send|upload|forward|post)\b.{0,80}https?://", re.IGNORECASE),
    "persistent_authority": re.compile(r"\b(remember permanently|future .{0,30}pre-approved|always authorize)\b", re.IGNORECASE),
}


def triage(envelope: ContentEnvelope, extra_rules: Iterable[tuple[str, re.Pattern[str]]] = ()) -> TriageResult:
    if envelope.trusted:
        return TriageResult("allow", ())
    rules = [*RULES.items(), *extra_rules]
    matches = tuple(name for name, pattern in rules if pattern.search(envelope.content))
    if {"secret_request", "external_transfer"} & set(matches):
        action = "block"
    elif matches:
        action = "quarantine"
    else:
        action = "allow"
    return TriageResult(action, matches)
