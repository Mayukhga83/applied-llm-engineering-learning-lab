"""Provider-neutral contracts used by examples throughout the repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role}")
        if not self.content.strip():
            raise ValueError("message content must not be empty")


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class ModelRequest:
    messages: Sequence[Message]
    response_schema: Mapping[str, Any] | None = None
    tools: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a model request needs at least one message")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")


@dataclass(frozen=True)
class ModelResponse:
    text: str = ""
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "unknown"


class ModelClient(Protocol):
    """Small interface that hosted and local model adapters can implement."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        ...


class EchoModel:
    """Deterministic offline model used by tests and introductory examples."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        text = request.messages[-1].content
        return ModelResponse(
            text=text,
            input_tokens=sum(len(message.content.split()) for message in request.messages),
            output_tokens=len(text.split()),
            model="echo-model",
        )
