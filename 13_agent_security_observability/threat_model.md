# Threat Model

## Protected assets

- model and tool credentials;
- private retrieved documents;
- persistent user and agent memory;
- integrity of tool calls and saved artifacts;
- availability and inference budget;
- trace data, which may contain sensitive context.

## Trust boundaries

1. User input enters the application.
2. External documents enter retrieval context.
3. A model proposes structured output or a tool call.
4. Application code authorizes and executes the tool.
5. Tool output returns to model context.
6. Selected information may enter persistent memory.
7. Redacted telemetry leaves the request path.

## Abuse cases

| Threat | Example | Required control |
|---|---|---|
| Direct injection | User requests hidden instructions | Treat prompts as data; enforce policy in code |
| Indirect injection | Retrieved page asks the agent to export secrets | Mark external content untrusted and prohibit instruction following |
| Tool misuse | Model calls a write tool with an unauthorized path | Validate arguments and authorize against user identity |
| Data exfiltration | Tool result is sent to an arbitrary URL | Destination allowlist and data classification |
| Memory poisoning | Untrusted text is stored as a durable fact | Provenance, review, expiration, and namespace isolation |
| Denial of wallet | Agent repeatedly retries a failing tool | Step, token, time, and cost budgets |
| Trace leakage | Prompts or secrets appear in logs | Content recording off by default and field-level redaction |

No prompt can serve as an authorization boundary. High-impact operations require deterministic policy and explicit approval outside the model.
