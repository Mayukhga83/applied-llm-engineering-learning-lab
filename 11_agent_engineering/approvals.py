"""Human-approval policies kept outside model reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from research_tools import AgentTool


@dataclass(frozen=True)
class ApprovalRecord:
    tool: str
    arguments: Mapping[str, Any]
    approved: bool
    reason: str
    recorded_at: str


@dataclass
class ApprovalGate:
    """Apply deterministic rules before optionally asking a human callback."""

    reviewer: Callable[[AgentTool, Mapping[str, Any]], tuple[bool, str]] | None = None
    denied_risks: set[str] = field(default_factory=lambda: {"destructive"})
    records: list[ApprovalRecord] = field(default_factory=list)

    def __call__(self, tool: AgentTool, arguments: Mapping[str, Any]) -> bool:
        if tool.risk in self.denied_risks:
            decision = (False, f"risk level {tool.risk} is denied by policy")
        elif not tool.requires_approval:
            decision = (True, "tool is pre-approved read-only capability")
        elif self.reviewer is None:
            decision = (False, "human approval is required but no reviewer is available")
        else:
            decision = self.reviewer(tool, arguments)
        self.records.append(
            ApprovalRecord(
                tool=tool.name,
                arguments=dict(arguments),
                approved=decision[0],
                reason=decision[1],
                recorded_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        return decision[0]
