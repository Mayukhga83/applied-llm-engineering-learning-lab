# Structured Generation

This module introduces explicit contracts between a language model and the application around it. The examples use standard-library dataclasses and JSON so the concepts remain independent of any model provider.

## Learning objectives

- represent model requests and responses without provider-specific types;
- validate JSON-shaped outputs before application code consumes them;
- repair bounded, well-understood formatting failures;
- expose tools through typed schemas;
- distinguish retryable generation failures from application failures.

Run the examples from the repository root and use `python -m unittest discover -s tests` to execute the test suite.
