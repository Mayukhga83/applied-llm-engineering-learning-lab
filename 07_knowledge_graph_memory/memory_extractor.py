import re
from dataclasses import dataclass


@dataclass
class MemoryTriple:
    subject: str
    relation: str
    object: str


def simple_extract(text: str):
    pattern = r"(\b[A-Z][a-zA-Z]+)\s+(likes|studies|uses|builds|learns)\s+([A-Za-z0-9_\- ]+)"
    triples = []
    for match in re.finditer(pattern, text):
        triples.append(MemoryTriple(match.group(1), match.group(2), match.group(3).strip()))
    return triples


if __name__ == "__main__":
    print(simple_extract("Mayukh studies LLM evaluation. Mayukh builds RAG systems."))
