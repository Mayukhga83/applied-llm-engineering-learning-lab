# Agent Security and Observability

This module treats the model, retrieved content, tool output, and memory as untrusted inputs. Security controls live at deterministic application boundaries: authorization, argument validation, output validation, resource budgets, and audit records.

The telemetry examples emit vendor-neutral JSON spans inspired by OpenTelemetry semantic conventions. They avoid recording prompt content by default because observability data can itself become a sensitive-data sink.
