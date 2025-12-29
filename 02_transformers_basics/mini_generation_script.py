from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generate_text(prompt: str, model_name: str = "distilgpt2", max_new_tokens: int = 60):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(generate_text("A retrieval augmented generation system is"))
