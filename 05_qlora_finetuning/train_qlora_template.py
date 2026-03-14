# QLoRA training template.
#
# This is a learning template. For real training, run on a CUDA machine and adjust:
# - model_name
# - dataset path
# - batch size
# - sequence length

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model

try:
    from trl import SFTTrainer
except ImportError:
    SFTTrainer = None


def main():
    if SFTTrainer is None:
        raise ImportError("Install trl first: pip install trl")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset = load_dataset("json", data_files="data/train.jsonl")["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir="outputs/qwen-responsible-ai-qlora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=args,
        tokenizer=tokenizer,
        max_seq_length=1024,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
