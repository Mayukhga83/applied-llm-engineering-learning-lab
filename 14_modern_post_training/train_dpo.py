"""Validate or execute pairwise Direct Preference Optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "dpo.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.method != "dpo":
        raise ValueError("DPO launcher requires a dpo configuration")
    if not args.execute:
        print(json.dumps({"status": "validated", "precondition": "evaluate the SFT checkpoint first", **config.__dict__}, indent=2))
        return
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as error:
        raise SystemExit("Install the optional post-training dependencies before using --execute") from error
    dataset = load_dataset("json", data_files=config.dataset, split="train")
    trainer = DPOTrainer(
        model=config.model,
        train_dataset=dataset,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"),
        args=DPOConfig(
            output_dir=config.output_dir,
            seed=config.seed,
            learning_rate=config.learning_rate,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=config.max_length,
            beta=0.1,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
