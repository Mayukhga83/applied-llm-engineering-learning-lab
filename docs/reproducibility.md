# Reproducibility Contract

Every reported experiment should preserve enough information to explain and rerun the result.

## Record before execution

- immutable dataset revision and split procedure;
- model and tokenizer identifiers or local artifact hashes;
- prompt, tool schema, and system-policy versions;
- random seeds and decoding parameters;
- training configuration and dependency versions;
- hardware, precision, quantization, and memory limits;
- primary metric, acceptance threshold, and stopping rule.

## Record after execution

- per-case predictions rather than aggregate scores alone;
- retrieval results and agent traces needed for error attribution;
- latency, tokens, estimated cost, retries, and failures;
- checkpoint hashes and evaluation dataset revision;
- deviations from the original plan;
- known limitations and examples of remaining failures.

## Comparison rules

Change one experimental factor at a time where possible. Use identical held-out cases and decoding settings across checkpoints. Calibrate model judges against human labels, keep deterministic graders in the loop, and never report fixture results as representative model benchmarks.

## Historical metadata

The project is a retrospective learning archive. Historical commit dates encode the intended study sequence and must remain disclosed in the README and timeline. They are not evidence that the work was performed on those dates.
