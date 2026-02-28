"""A bounded, inspectable tool-using agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from research_tools import AgentTool


@dataclass(frozen=True)
class Decision:
    kind: str
    content: str = ""
    tool_name: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceEvent:
    step: int
    event: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class AgentResult:
    answer: str
    trace: tuple[TraceEvent, ...]
    stopped_reason: str


Policy = Callable[[str, tuple[TraceEvent, ...], list[dict[str, Any]]], Decision]
Approval = Callable[[AgentTool, Mapping[str, Any]], bool]


class BoundedAgent:
    def __init__(self, tools: Mapping[str, AgentTool], policy: Policy, approval: Approval | None = None, max_steps: int = 6, max_result_chars: int = 4_000) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.tools = dict(tools)
        self.policy = policy
        self.approval = approval or (lambda tool, arguments: not tool.requires_approval)
        self.max_steps = max_steps
        self.max_result_chars = max_result_chars

    def run(self, objective: str) -> AgentResult:
        trace: list[TraceEvent] = []
        schemas = [tool.mcp_schema() for tool in self.tools.values()]
        for step in range(1, self.max_steps + 1):
            decision = self.policy(objective, tuple(trace), schemas)
            trace.append(TraceEvent(step, "decision", {"kind": decision.kind, "tool": decision.tool_name}))
            if decision.kind == "finish":
                return AgentResult(decision.content, tuple(trace), "completed")
            if decision.kind != "tool":
                return AgentResult("", tuple(trace), "invalid_decision")
            tool = self.tools.get(decision.tool_name)
            if tool is None:
                trace.append(TraceEvent(step, "tool_error", {"error": "unknown_tool"}))
                continue
            if not self.approval(tool, decision.arguments):
                trace.append(TraceEvent(step, "approval_denied", {"tool": tool.name}))
                continue
            try:
                result = tool.invoke(decision.arguments)
                serialized = json.dumps(result, ensure_ascii=False, default=str)[: self.max_result_chars]
                trace.append(TraceEvent(step, "tool_result", {"tool": tool.name, "result": serialized}))
            except Exception as error:
                trace.append(TraceEvent(step, "tool_error", {"tool": tool.name, "error": type(error).__name__, "message": str(error)}))
        return AgentResult("", tuple(trace), "step_limit")
