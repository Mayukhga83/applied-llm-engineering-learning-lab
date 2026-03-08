# Evaluation-Driven Development

Evaluation is the integration layer for the advanced track. A case captures inputs, references, retrieved evidence, output, and execution trace. Graders remain small and composable so failures identify a specific layer rather than collapsing into one opaque score.

The examples prefer deterministic checks. Model-based grading is reserved for qualities that cannot be expressed as programmatic assertions and must be calibrated against labeled examples.
