import json
from pathlib import Path


def convert_to_jsonl(input_examples, output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for ex in input_examples:
            record = {
                "messages": [
                    {"role": "system", "content": "You are a helpful and responsible AI assistant."},
                    {"role": "user", "content": ex["instruction"]},
                    {"role": "assistant", "content": ex["response"]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    examples = [
        {
            "instruction": "Explain LoRA in simple terms.",
            "response": "LoRA fine-tunes small adapter layers instead of changing the full model."
        }
    ]
    convert_to_jsonl(examples, "data/train.jsonl")
