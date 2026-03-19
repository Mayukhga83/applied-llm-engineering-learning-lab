"""Token, cost, latency, retry, and failure accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class UsageEvent:
    operation: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    failed: bool = False


@dataclass(frozen=True)
class Budget:
    max_cost_usd: float = 1.0
    max_tokens: int = 50_000
    max_latency_ms: float = 60_000
    max_retries: int = 3


@dataclass
class UsageLedger:
    budget: Budget
    events: list[UsageEvent] = field(default_factory=list)

    def record(self, event: UsageEvent) -> None:
        self.events.append(event)
        violations = self.violations()
        if violations:
            raise RuntimeError(f"execution budget exceeded: {', '.join(violations)}")

    def totals(self) -> dict[str, float]:
        tokens = sum(event.input_tokens + event.output_tokens for event in self.events)
        latencies = [event.latency_ms for event in self.events]
        return {
            "requests": len(self.events),
            "tokens": tokens,
            "cost_usd": round(sum(event.cost_usd for event in self.events), 6),
            "latency_ms": sum(latencies),
            "mean_latency_ms": mean(latencies) if latencies else 0.0,
            "retries": sum(event.retries for event in self.events),
            "failures": sum(event.failed for event in self.events),
        }

    def violations(self) -> list[str]:
        totals = self.totals()
        violations = []
        if totals["cost_usd"] > self.budget.max_cost_usd:
            violations.append("cost")
        if totals["tokens"] > self.budget.max_tokens:
            violations.append("tokens")
        if totals["latency_ms"] > self.budget.max_latency_ms:
            violations.append("latency")
        if totals["retries"] > self.budget.max_retries:
            violations.append("retries")
        return violations


def percentile(values: Iterable[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]
