"""Serializable agent state and explicit checkpoint boundaries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class AgentState:
    run_id: str
    objective: str
    status: str = "running"
    step: int = 0
    observations: list[Mapping[str, Any]] = field(default_factory=list)
    pending_approval: Mapping[str, Any] | None = None

    def observe(self, event: Mapping[str, Any]) -> None:
        self.observations.append(dict(event))
        self.step += 1

    def checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def restore(cls, path: Path) -> "AgentState":
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"run_id", "objective", "status", "step", "observations", "pending_approval"}
        if set(data) != required:
            raise ValueError("checkpoint has an unexpected schema")
        return cls(**data)


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._states: dict[str, str] = {}

    def save(self, state: AgentState) -> None:
        self._states[state.run_id] = json.dumps(asdict(state), default=str)

    def load(self, run_id: str) -> AgentState:
        if run_id not in self._states:
            raise KeyError(run_id)
        return AgentState(**json.loads(self._states[run_id]))
