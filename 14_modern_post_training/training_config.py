"""Validated configuration shared by post-training launchers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    model: str
    dataset: str
    output_dir: str
    seed: int
    learning_rate: float
    epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    use_peft: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        config = cls(**value)
        if config.method not in {"sft", "dpo", "grpo"}:
            raise ValueError(f"unsupported method: {config.method}")
        if config.learning_rate <= 0 or config.epochs < 1 or config.per_device_batch_size < 1:
            raise ValueError("training values must be positive")
        return config


def load_config(path: Path) -> TrainingConfig:
    return TrainingConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))
