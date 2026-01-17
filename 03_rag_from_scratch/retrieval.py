from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int | None = None


class SimpleVectorIndex:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings = None

    def add(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        self.chunks.extend(chunks)
        self.embeddings = embeddings if self.embeddings is None else np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.embeddings is None or not self.chunks:
            return []

        scores = self.embeddings @ query_embedding.reshape(-1)
        top_ids = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_ids]
