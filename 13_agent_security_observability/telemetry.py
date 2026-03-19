"""Small JSON tracer using GenAI-oriented semantic attribute names."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Mapping

SENSITIVE_MARKERS = ("authorization", "api_key", "password", "secret", "token")


def redact(attributes: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in attributes.items():
        lowered = key.casefold()
        output[key] = "[REDACTED]" if any(marker in lowered for marker in SENSITIVE_MARKERS) else value
    return output


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    started_ns: int
    ended_ns: int = 0
    status: str = "unset"
    attributes: dict[str, Any] = field(default_factory=dict)
    error_type: str = ""

    @property
    def duration_ms(self) -> float:
        return (self.ended_ns - self.started_ns) / 1_000_000 if self.ended_ns else 0.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["duration_ms"] = self.duration_ms
        value["attributes"] = redact(self.attributes)
        return value


class JsonTracer:
    def __init__(self, export: Callable[[Mapping[str, Any]], None] | None = None, record_content: bool = False) -> None:
        self.export = export or (lambda span: print(json.dumps(span, default=str)))
        self.record_content = record_content

    @contextmanager
    def span(self, name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Span]:
        span = Span(name, uuid.uuid4().hex, uuid.uuid4().hex[:16], time.time_ns(), attributes=dict(attributes or {}))
        if not self.record_content:
            span.attributes = {key: value for key, value in span.attributes.items() if not key.endswith(("prompt", "completion", "content"))}
        try:
            yield span
            span.status = "ok"
        except Exception as error:
            span.status = "error"
            span.error_type = type(error).__name__
            raise
        finally:
            span.ended_ns = time.time_ns()
            self.export(span.to_dict())


def model_attributes(model: str, provider: str, input_tokens: int = 0, output_tokens: int = 0) -> dict[str, Any]:
    return {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model,
        "gen_ai.provider.name": provider,
        "gen_ai.usage.input_tokens": input_tokens,
        "gen_ai.usage.output_tokens": output_tokens,
    }
