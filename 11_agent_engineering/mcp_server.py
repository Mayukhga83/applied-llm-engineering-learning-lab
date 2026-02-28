"""Minimal JSON-RPC MCP learning server for the tools/list and tools/call flow."""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, TextIO

from research_tools import AgentTool, build_research_tools


class LearningMCPServer:
    protocol_version = "2025-11-25"

    def __init__(self, tools: Mapping[str, AgentTool]) -> None:
        self.tools = dict(tools)

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "learning-lab", "version": "1.0.0"},
                }
            elif method == "tools/list":
                result = {"tools": [self.tools[name].mcp_schema() for name in sorted(self.tools)]}
            elif method == "tools/call":
                name = params.get("name")
                if name not in self.tools:
                    raise KeyError(f"unknown tool: {name}")
                output = self.tools[name].invoke(params.get("arguments", {}))
                result = {"content": [{"type": "text", "text": json.dumps(output, default=str)}], "isError": False}
            else:
                return self._error(request_id, -32601, f"method not found: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (KeyError, TypeError, ValueError) as error:
            return self._error(request_id, -32602, str(error))
        except Exception as error:
            return self._error(request_id, -32603, f"internal error: {type(error).__name__}")

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    def serve(self, reader: TextIO = sys.stdin, writer: TextIO = sys.stdout) -> None:
        for line in reader:
            if line.strip():
                writer.write(json.dumps(self.handle(json.loads(line))) + "\n")
                writer.flush()


def main() -> None:
    documents = {"rag": "Hybrid retrieval combines lexical and semantic evidence.", "evals": "Trace graders inspect tool calls and results."}
    LearningMCPServer(build_research_tools(documents)).serve()


if __name__ == "__main__":
    main()
