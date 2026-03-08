import unittest
from pathlib import Path

from evals.cases import EvalCase
from evals.deterministic import contains_expected, token_f1
from evals.model_judge import CalibrationExample, calibration_report, grade_with_model
from evals.retrieval import citation_validity, ndcg_at_k, recall_at_k
from evals.run import evaluate
from evals.traces import excessive_retries, trace_structure


class EvaluationTests(unittest.TestCase):
    def test_checked_in_baseline_passes(self) -> None:
        path = Path(__file__).parents[1] / "evals" / "data" / "baseline_cases.jsonl"
        _, summary = evaluate(path)
        self.assertEqual(summary["pass_rate"], 1.0)

    def test_response_graders(self) -> None:
        case = EvalCase("x", "question", "hybrid retrieval", "Hybrid retrieval combines signals")
        self.assertTrue(contains_expected(case).passed)
        self.assertGreater(token_f1(case, threshold=0.4).score, 0.4)

    def test_retrieval_metrics(self) -> None:
        case = EvalCase("x", "query", relevant_ids=("a", "b"), retrieved_ids=("a", "c", "b"))
        self.assertEqual(recall_at_k(case, 3).score, 1.0)
        self.assertGreater(ndcg_at_k(case, 3).score, 0.8)

    def test_citation_validity(self) -> None:
        case = EvalCase("x", "q", metadata={"citations": [{"source_id": "d", "quote": "supported claim"}]})
        self.assertTrue(citation_validity(case, {"d": "The supported claim appears here."}).passed)

    def test_trace_graders_flag_loops(self) -> None:
        trace = tuple({"event": "decision", "tool": "search", "arguments": {"query": "x"}} for _ in range(3))
        case = EvalCase("x", "q", trace=trace)
        self.assertTrue(trace_structure(case).passed)
        self.assertFalse(excessive_retries(case).passed)

    def test_model_judge_is_calibrated_against_labels(self) -> None:
        judge = lambda prompt: {"score": 0.9 if "Response: correct" in prompt else 0.1, "reason": "fixture"}
        positive = EvalCase("yes", "q", "reference", "correct")
        negative = EvalCase("no", "q", "reference", "wrong")
        self.assertTrue(grade_with_model(positive, judge, "correctness").passed)
        report = calibration_report([CalibrationExample(positive, True), CalibrationExample(negative, False)], judge, "correctness")
        self.assertEqual(report["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
