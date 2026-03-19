import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[1] / "13_agent_security_observability"
sys.path.insert(0, str(MODULE))

from dashboard import render  # noqa: E402
from metrics import Budget, UsageEvent, UsageLedger  # noqa: E402
from telemetry import JsonTracer, model_attributes  # noqa: E402


class ObservabilityTests(unittest.TestCase):
    def test_tracer_redacts_secrets_and_omits_content(self) -> None:
        exported = []
        tracer = JsonTracer(exported.append)
        with tracer.span("chat", {"api_key": "secret", "gen_ai.prompt": "private", **model_attributes("test", "local") } ):
            pass
        self.assertEqual(exported[0]["attributes"]["api_key"], "[REDACTED]")
        self.assertNotIn("gen_ai.prompt", exported[0]["attributes"])

    def test_budget_stops_excessive_cost(self) -> None:
        ledger = UsageLedger(Budget(max_cost_usd=0.01))
        with self.assertRaises(RuntimeError):
            ledger.record(UsageEvent("chat", 10, cost_usd=0.02))

    def test_dashboard_reports_operational_signals(self) -> None:
        output = render([UsageEvent("chat", 120, 10, 5, 0.001, retries=1)])
        self.assertIn("P95 latency", output)
        self.assertIn("15", output)


if __name__ == "__main__":
    unittest.main()
