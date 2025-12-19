from transformers import AutoTokenizer


def inspect_tokenization(model_name: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)

    print("Text:", text)
    print("Tokens:", tokens)
    print("Token IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))


if __name__ == "__main__":
    inspect_tokenization(
        "distilbert-base-uncased",
        "Large language models transform text into tokens."
    )
