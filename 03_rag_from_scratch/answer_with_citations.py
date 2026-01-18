from typing import List, Tuple
from openai import OpenAI
from retrieval import DocumentChunk


def build_context(results: List[Tuple[DocumentChunk, float]]) -> str:
    blocks = []
    for idx, (chunk, score) in enumerate(results, start=1):
        source = f"{chunk.source}, page {chunk.page}" if chunk.page else chunk.source
        blocks.append(f"[{idx}] Source: {source}\nScore: {score:.3f}\n{chunk.text}")
    return "\n\n".join(blocks)


def answer_question(question: str, results: List[Tuple[DocumentChunk, float]], model: str = "gpt-4o-mini") -> str:
    client = OpenAI()
    context = build_context(results)

    prompt = f"""
Answer the question using only the context below.
Cite sources using bracket numbers like [1] or [2].
If the context is insufficient, say so.

Question:
{question}

Context:
{context}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful citation-grounded research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
