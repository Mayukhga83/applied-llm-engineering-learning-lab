# Post-Training Experiment Card

This card prevents configuration files from being mistaken for completed experiments. Populate it only after producing held-out predictions from every checkpoint.

## Required checkpoints

- untuned base model;
- SFT checkpoint;
- DPO checkpoint initialized from SFT;
- GRPO checkpoint initialized from the same evaluated SFT baseline.

## Controls

- fixed seed and data split;
- identical held-out prompts;
- identical decoding settings;
- checkpoint-specific training data excluded from evaluation;
- deterministic rewards tested independently;
- wall time, hardware, peak memory, and token counts recorded.

## Results

Status: **not executed**. The repository includes validated configurations and a comparison harness, but no training results are claimed.

Run `python 14_modern_post_training/compare_checkpoints.py <predictions.jsonl>` after generating prediction rows containing `checkpoint`, `prompt`, `completion`, and `reference`.
