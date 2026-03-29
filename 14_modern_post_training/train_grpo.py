"""Validate or execute a deliberately small GRPO experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rewards import exact_answer_reward
from training_config import load_config


def accuracy_reward(completions: list[str], reference: list[str], **_: object) -> list[float]:
    return [exact_answer_reward(completion, expected) for completion, expected in zip(completions, reference)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "grpo.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.method != "grpo":
        raise ValueError("GRPO launcher requires a grpo configuration")
    if not args.execute:
        print(json.dumps({"status": "validated", "warning": "Run only where answers have deterministic rewards", **config.__dict__}, indent=2))
        return
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise SystemExit("Install the optional post-training dependencies before using --execute") from error
    dataset = load_dataset("json", data_files=config.dataset, split="train").map(lambda row: {"reference": row["chosen"]})
    trainer = GRPOTrainer(
        model=config.model,
        reward_funcs=accuracy_reward,
        train_dataset=dataset,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"),
        args=GRPOConfig(
            output_dir=config.output_dir,
            seed=config.seed,
            learning_rate=config.learning_rate,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_completion_length=512,
            num_generations=4,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
