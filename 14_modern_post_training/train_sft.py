"""Validate or execute the small supervised fine-tuning baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "sft.json")
    parser.add_argument("--execute", action="store_true", help="Run training instead of validating the plan")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.method != "sft":
        raise ValueError("SFT launcher requires an sft configuration")
    if not args.execute:
        print(json.dumps({"status": "validated", **config.__dict__}, indent=2))
        return
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise SystemExit("Install the optional post-training dependencies before using --execute") from error
    dataset = load_dataset("json", data_files=config.dataset, split="train")
    dataset = dataset.map(lambda row: {"text": f"User: {row['prompt']}\nAssistant: {row['chosen']}"})
    trainer = SFTTrainer(
        model=config.model,
        train_dataset=dataset,
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir=config.output_dir,
            seed=config.seed,
            learning_rate=config.learning_rate,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=config.max_length,
            dataset_text_field="text",
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
