import json
import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[1] / "09_structured_generation"
sys.path.insert(0, str(MODULE))

from schemas import GroundedAnswer  # noqa: E402
from tools import ToolDefinition, ToolRegistry  # noqa: E402
from validation import StructuredOutputError, generate_with_retries, parse_and_validate  # noqa: E402


VALID = {
    "answer": "Hybrid retrieval combines complementary signals.",
    "citations": [{"source_id": "doc-1", "quote": "combine lexical and semantic retrieval"}],
    "confidence": 0.9,
}


class StructuredGenerationTests(unittest.TestCase):
    def test_valid_answer(self) -> None:
        result = parse_and_validate(json.dumps(VALID), GroundedAnswer.from_mapping)
        self.assertFalse(result.repaired)
        self.assertEqual(result.value.citations[0].source_id, "doc-1")

    def test_fenced_answer_is_repaired(self) -> None:
        result = parse_and_validate(f"```json\n{json.dumps(VALID)}\n```", GroundedAnswer.from_mapping)
        self.assertTrue(result.repaired)

    def test_missing_citations_are_rejected(self) -> None:
        invalid = {**VALID, "citations": []}
        with self.assertRaises(StructuredOutputError):
            parse_and_validate(json.dumps(invalid), GroundedAnswer.from_mapping)

    def test_retry_receives_validation_feedback(self) -> None:
        outputs = iter(["not json", json.dumps(VALID)])
        prompts: list[str] = []

        def generate(prompt: str) -> str:
            prompts.append(prompt)
            return next(outputs)

        result = generate_with_retries(generate, "answer", GroundedAnswer.from_mapping)
        self.assertEqual(result.value.confidence, 0.9)
        self.assertIn("Previous error", prompts[1])

    def test_tool_arguments_are_checked(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition("add", "Add integers", {"a": int, "b": int}, lambda a, b: a + b))
        self.assertEqual(registry.invoke("add", {"a": 2, "b": 3}), 5)
        with self.assertRaises(TypeError):
            registry.invoke("add", {"a": "2", "b": 3})


if __name__ == "__main__":
    unittest.main()
