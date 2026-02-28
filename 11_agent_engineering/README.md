# Agent Engineering and MCP

This module builds an agent as a bounded state machine instead of an open-ended prompt loop. Tools have explicit schemas, risk levels, time budgets, and approval requirements. A small JSON-RPC learning server demonstrates the core Model Context Protocol tool flow without hiding it behind a framework.

The MCP example implements only the methods required by the exercise. Use an official SDK for a production MCP server.

Run an in-process protocol demonstration with:

```bash
python 11_agent_engineering/mcp_client.py
```
