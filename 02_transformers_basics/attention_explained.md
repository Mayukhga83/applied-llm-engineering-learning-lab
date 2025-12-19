# Transformer Attention

Self-attention lets each token look at other tokens in the same sequence.

A simplified view:

1. Each token is projected into query, key, and value vectors.
2. Query-key similarity estimates relevance.
3. Similarities are normalized with softmax.
4. The output is a weighted combination of value vectors.

This is why transformers can model long-range dependencies better than older sequential models.
