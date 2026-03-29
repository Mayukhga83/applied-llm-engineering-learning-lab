"""Validate and split pairwise preference data reproducibly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

REQUIRED = ("id", "prompt", "chosen", "rejected")


def validate(item: Mapping[str, object]) -> dict[str, str]:
    missing = [key for key in REQUIRED if not isinstance(item.get(key), str) or not str(item[key]).strip()]
    if missing:
        raise ValueError(f"missing non-empty fields: {missing}")
    normalized = {key: str(item[key]).strip() for key in REQUIRED}
    if normalized["chosen"] == normalized["rejected"]:
        raise ValueError("chosen and rejected responses must differ")
    return normalized


def deterministic_split(items: Iterable[Mapping[str, object]], eval_fraction: float = 0.2) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")
    train, evaluation, seen = [], [], set()
    for raw in items:
        item = validate(raw)
        fingerprint = hashlib.sha256((item["prompt"] + item["chosen"] + item["rejected"]).encode()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        bucket = int(fingerprint[:8], 16) / 0xFFFFFFFF
        (evaluation if bucket < eval_fraction else train).append(item)
    return train, evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, nargs="?", default=Path(__file__).parent / "data" / "preferences.jsonl")
    args = parser.parse_args()
    items = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    train, evaluation = deterministic_split(items)
    print(json.dumps({"valid": len(train) + len(evaluation), "train": len(train), "evaluation": len(evaluation)}, indent=2))


if __name__ == "__main__":
    main()
