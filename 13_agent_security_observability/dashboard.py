"""Render a compact operational dashboard from usage events."""

from __future__ import annotations

from typing import Iterable

from metrics import UsageEvent, percentile


def render(events: Iterable[UsageEvent]) -> str:
    events = list(events)
    tokens = sum(event.input_tokens + event.output_tokens for event in events)
    cost = sum(event.cost_usd for event in events)
    failures = sum(event.failed for event in events)
    retries = sum(event.retries for event in events)
    failure_rate = failures / len(events) if events else 0.0
    p95 = percentile((event.latency_ms for event in events), 0.95)
    return "\n".join(
        [
            "# LLM Operations Snapshot",
            "",
            "| Signal | Value |",
            "|---|---:|",
            f"| Requests | {len(events)} |",
            f"| Tokens | {tokens} |",
            f"| Estimated cost | ${cost:.6f} |",
            f"| P95 latency | {p95:.1f} ms |",
            f"| Retries | {retries} |",
            f"| Failure rate | {failure_rate:.1%} |",
        ]
    )


if __name__ == "__main__":
    sample = [UsageEvent("retrieve", 42), UsageEvent("chat", 380, 420, 90, 0.0012)]
    print(render(sample))
