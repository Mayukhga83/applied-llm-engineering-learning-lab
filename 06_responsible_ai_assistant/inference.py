from safety_classifier import classify_and_rewrite


def main():
    while True:
        text = input("Enter text to evaluate, or 'quit': ")
        if text.lower().strip() == "quit":
            break

        result = classify_and_rewrite(text)
        print(result)


if __name__ == "__main__":
    main()
