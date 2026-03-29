import importlib.util
import json
import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[1] / "14_modern_post_training"
sys.path.insert(0, str(MODULE))

from compare_checkpoints import summarize  # noqa: E402
from prepare_preferences import deterministic_split, validate  # noqa: E402
from rewards import exact_answer_reward, numeric_answer_reward, structured_answer_reward  # noqa: E402
from training_config import load_config  # noqa: E402


class PostTrainingTests(unittest.TestCase):
    def test_preference_validation_and_deduplication(self) -> None:
        row = {"id": "1", "prompt": "p", "chosen": "good", "rejected": "bad"}
        self.assertEqual(validate(row)["chosen"], "good")
        train, evaluation = deterministic_split([row, row], eval_fraction=0.5)
        self.assertEqual(len(train) + len(evaluation), 1)

    def test_every_training_config_is_valid(self) -> None:
        for name in ("sft", "dpo", "grpo"):
            config = load_config(MODULE / "configs" / f"{name}.json")
            self.assertEqual(config.method, name)
            self.assertEqual(config.seed, 42)

    def test_verifiable_rewards(self) -> None:
        self.assertEqual(exact_answer_reward("Paris", "paris"), 1.0)
        self.assertEqual(numeric_answer_reward("The answer is 4", "4"), 1.0)
        self.assertEqual(structured_answer_reward(json.dumps({"answer": "x", "citations": ["d"]})), 1.0)

    def test_checkpoint_comparison_groups_results(self) -> None:
        rows = [
            {"checkpoint": "base", "prompt": "p", "completion": "wrong", "reference": "right"},
            {"checkpoint": "sft", "prompt": "p", "completion": "right", "reference": "right"},
        ]
        report = summarize(rows)
        self.assertEqual(report["base"]["exact_match"], 0.0)
        self.assertEqual(report["sft"]["exact_match"], 1.0)


if __name__ == "__main__":
    unittest.main()
