import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[1] / "11_agent_engineering"
sys.path.insert(0, str(MODULE))

from agent_loop import BoundedAgent, Decision  # noqa: E402
from approvals import ApprovalGate  # noqa: E402
from mcp_client import LearningMCPClient, MCPClientError  # noqa: E402
from mcp_server import LearningMCPServer  # noqa: E402
from research_tools import build_research_tools  # noqa: E402
from state import AgentState, InMemoryCheckpointStore  # noqa: E402


class AgentEngineeringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = build_research_tools({"d1": "Hybrid retrieval combines lexical and semantic signals."})

    def test_agent_records_tool_error_then_recovers(self) -> None:
        decisions = iter(
            [
                Decision("tool", tool_name="missing", arguments={}),
                Decision("tool", tool_name="search", arguments={"query": "hybrid"}),
                Decision("finish", content="Recovered with evidence."),
            ]
        )
        result = BoundedAgent(self.tools, lambda objective, trace, schemas: next(decisions)).run("research")
        self.assertEqual(result.stopped_reason, "completed")
        self.assertTrue(any(event.event == "tool_error" for event in result.trace))
        self.assertTrue(any(event.event == "tool_result" for event in result.trace))

    def test_step_limit_stops_runaway_policy(self) -> None:
        policy = lambda objective, trace, schemas: Decision("tool", tool_name="search", arguments={"query": "x"})
        result = BoundedAgent(self.tools, policy, max_steps=2).run("loop")
        self.assertEqual(result.stopped_reason, "step_limit")

    def test_write_tool_is_denied_without_reviewer(self) -> None:
        gate = ApprovalGate()
        self.assertFalse(gate(self.tools["save_note"], {"title": "x", "body": "y"}))
        self.assertTrue(gate(self.tools["search"], {"query": "x"}))

    def test_checkpoint_round_trip(self) -> None:
        store = InMemoryCheckpointStore()
        state = AgentState("run-1", "research")
        state.observe({"event": "started"})
        store.save(state)
        self.assertEqual(store.load("run-1").step, 1)

    def test_mcp_tool_flow_and_error(self) -> None:
        server = LearningMCPServer(self.tools)
        client = LearningMCPClient(server.handle)
        self.assertEqual(client.initialize()["serverInfo"]["name"], "learning-lab")
        self.assertEqual(len(client.list_tools()), 3)
        self.assertEqual(client.call_tool("read_document", {"document_id": "d1"})[:6], "Hybrid")
        with self.assertRaises(MCPClientError):
            client.call_tool("missing", {})


if __name__ == "__main__":
    unittest.main()
