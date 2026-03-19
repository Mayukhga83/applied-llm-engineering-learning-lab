"""Deterministic least-privilege policy for model-proposed tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlparse


@dataclass(frozen=True)
class Principal:
    id: str
    allowed_tools: frozenset[str]
    writable_roots: tuple[str, ...] = ()
    allowed_hosts: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    arguments: Mapping[str, Any]
    risk: str = "read"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def evaluate(principal: Principal, request: ToolRequest) -> PolicyDecision:
    reasons: list[str] = []
    if request.tool not in principal.allowed_tools:
        reasons.append("tool_not_allowed")
    path = request.arguments.get("path")
    if path is not None:
        if not isinstance(path, str) or ".." in PurePosixPath(path).parts:
            reasons.append("invalid_path")
        elif request.risk == "write" and not any(PurePosixPath(path).is_relative_to(PurePosixPath(root)) for root in principal.writable_roots):
            reasons.append("path_outside_writable_roots")
    url = request.arguments.get("url")
    if url is not None:
        host = urlparse(str(url)).hostname
        if not host or host not in principal.allowed_hosts:
            reasons.append("host_not_allowed")
    requires_approval = request.risk in {"write", "destructive", "external_side_effect"}
    if request.risk == "destructive":
        reasons.append("destructive_operation_denied")
    return PolicyDecision(not reasons, requires_approval and not reasons, tuple(reasons))
