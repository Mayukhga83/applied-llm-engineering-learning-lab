"""Transport-independent client for the learning MCP server."""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable, Mapping

SendRequest = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class MCPClientError(RuntimeError):
    pass


class LearningMCPClient:
    def __init__(self, send: SendRequest) -> None:
        self.send = send
        self._ids = itertools.count(1)
        self.server_info: Mapping[str, Any] = {}

    def _request(self, method: str, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        response = self.send({"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": dict(params or {})})
        if "error" in response:
            error = response["error"]
            raise MCPClientError(f"MCP {error.get('code')}: {error.get('message')}")
        return response["result"]

    def initialize(self) -> Mapping[str, Any]:
        result = self._request("initialize", {"protocolVersion": "2025-11-25", "clientInfo": {"name": "learning-client", "version": "1.0.0"}})
        self.server_info = result.get("serverInfo", {})
        return result

    def list_tools(self) -> list[Mapping[str, Any]]:
        return list(self._request("tools/list").get("tools", []))

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        blocks = result.get("content", [])
        if result.get("isError") or not blocks:
            raise MCPClientError(f"tool failed: {name}")
        return json.loads(blocks[0]["text"])


def in_process_demo() -> None:
    from mcp_server import LearningMCPServer
    from research_tools import build_research_tools

    server = LearningMCPServer(build_research_tools({"rag": "Hybrid retrieval combines signals."}))
    client = LearningMCPClient(server.handle)
    client.initialize()
    print(json.dumps(client.list_tools(), indent=2))
    print(client.call_tool("search", {"query": "hybrid retrieval"}))


if __name__ == "__main__":
    in_process_demo()
