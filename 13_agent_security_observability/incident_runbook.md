# Agent Incident Runbook

## First response

1. Disable write-capable and external-side-effect tools through the policy layer.
2. Preserve redacted traces, approval records, tool requests, and deployment identifiers.
3. Rotate credentials when a trace suggests disclosure or unauthorized tool access.
4. Quarantine affected memory records and retrieval documents without destroying evidence.
5. Reproduce the behavior with the smallest safe fixture.

## Classification

- **Prompt injection:** identify the untrusted origin and the missing trust boundary.
- **Tool misuse:** inspect authorization, validated arguments, credentials, and approval records.
- **Memory poisoning:** identify writer, provenance, readers, and expiration policy.
- **Runaway execution:** inspect step, time, token, retry, and monetary budgets.
- **Sensitive telemetry:** stop content recording and remove access before repairing retention.

## Recovery criteria

- an automated regression fixture reproduces the original failure;
- the control blocks the fixture without breaking the benign control case;
- affected credentials and stored data have been remediated;
- evaluation and security suites pass before tools are re-enabled;
- the incident record documents impact, cause, corrective action, and remaining risk.
