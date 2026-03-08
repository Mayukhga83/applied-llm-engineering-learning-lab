"""Command-line regression gate for checked-in evaluation cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .cases import EvalResult, load_cases
from .deterministic import aggregate, contains_expected
from .retrieval import recall_at_k
from .traces import trace_structure


def evaluate(path: Path) -> tuple[list[EvalResult], dict[str, float]]:
    results: list[EvalResult] = []
    for case in load_cases(path):
        if case.expected and case.output:
            results.append(contains_expected(case))
        if case.relevant_ids:
            results.append(recall_at_k(case, 3))
        if case.trace:
            results.append(trace_structure(case))
    return results, aggregate(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="?", default=Path(__file__).parent / "data" / "baseline_cases.jsonl")
    parser.add_argument("--minimum-pass-rate", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results, summary = evaluate(args.cases)
    report = {"summary": summary, "results": [asdict(result) for result in results]}
    rendered = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if summary["pass_rate"] < args.minimum_pass_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
