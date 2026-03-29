"""Compare held-out predictions produced by base, SFT, DPO, and GRPO models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

from rewards import exact_answer_reward, structured_answer_reward


def summarize(rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        required = {"checkpoint", "prompt", "completion", "reference"}
        if not required <= set(row):
            raise ValueError(f"prediction row is missing {sorted(required - set(row))}")
        groups.setdefault(str(row["checkpoint"]), []).append(row)
    report = {}
    for checkpoint, items in sorted(groups.items()):
        exact = [exact_answer_reward(str(item["completion"]), str(item["reference"])) for item in items]
        structured = [structured_answer_reward(str(item["completion"])) for item in items]
        report[checkpoint] = {
            "cases": len(items),
            "exact_match": sum(exact) / len(exact),
            "structured_output": sum(structured) / len(structured),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
