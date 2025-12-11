from dataclasses import dataclass
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class TextExample:
    text: str
    label: int


def train_toy_classifier(examples: List[TextExample]):
    texts = [ex.text for ex in examples]
    labels = [ex.label for ex in examples]

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(x, labels)
    return vectorizer, model


if __name__ == "__main__":
    data = [
        TextExample("This answer is helpful and respectful.", 1),
        TextExample("This response is abusive and harmful.", 0),
        TextExample("Clear explanation with useful details.", 1),
        TextExample("Insulting and toxic language.", 0),
    ]

    vectorizer, model = train_toy_classifier(data)
    query = ["This is a useful and respectful answer."]
    pred = model.predict(vectorizer.transform(query))[0]
    print("Prediction:", pred)
