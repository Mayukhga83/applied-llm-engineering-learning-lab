# Initial Advanced-Track Baseline

This report records the deterministic fixtures used while introducing the shared evaluation package. It is a software-regression baseline, not a claim about external model quality.

## Shared evaluation gate

| Metric | Result |
|---|---:|
| Cases graded | 3 |
| Mean score | 1.0000 |
| Pass rate | 1.0000 |

## Retrieval fixture

| System | Recall@3 | MRR |
|---|---:|---:|
| BM25 | 0.8333 | 0.7500 |
| Hashing-vector offline baseline | 1.0000 | 0.5833 |
| Hybrid reciprocal-rank fusion | 0.8333 | 0.7500 |
| Hybrid plus interaction reranking | 0.8333 | 0.7500 |

The six-document corpus is intentionally too small for comparative conclusions. Its purpose is to make metric behavior testable before replacing the corpus and dense encoder in a real experiment.
