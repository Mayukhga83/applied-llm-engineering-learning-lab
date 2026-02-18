# Advanced Retrieval-Augmented Generation

This module turns the original RAG demo into a measurable retrieval laboratory. It compares lexical and semantic retrieval, reciprocal-rank fusion, reranking, query decomposition, and citation verification against one versioned corpus.

All baseline implementations are intentionally small and dependency-free. They expose the mechanics clearly and provide stable test fixtures; production systems can replace individual retrievers without changing the evaluation contract.

Run the comparison with:

```bash
python 10_advanced_rag/benchmark.py
```

The hashing-vector retriever is a reproducible offline baseline, not a claim that feature hashing is a modern embedding model. Pass a production embedding function to `DenseRetriever` when comparing real systems.
