import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[1] / "10_advanced_rag"
sys.path.insert(0, str(MODULE))

from benchmark import run  # noqa: E402
from corpus import Document  # noqa: E402
from grounding import decompose_query, verify_citation  # noqa: E402


class AdvancedRagTests(unittest.TestCase):
    def test_all_retrievers_recall_the_fixture_answers(self) -> None:
        metrics = run(MODULE / "data")
        for result in metrics.values():
            self.assertGreaterEqual(result["recall@3"], 0.8)

    def test_query_decomposition_is_conservative(self) -> None:
        self.assertEqual(decompose_query("compare lexical and semantic retrieval"), ["compare lexical", "semantic retrieval"])
        self.assertEqual(decompose_query("explain reranking"), ["explain reranking"])

    def test_citation_requires_quote_and_claim_overlap(self) -> None:
        document = Document("d1", "Test", "Hybrid retrieval combines lexical and semantic signals.")
        check = verify_citation("Hybrid retrieval combines lexical signals", "d1", "Hybrid retrieval combines lexical and semantic signals.", {"d1": document})
        self.assertTrue(check.supported)


if __name__ == "__main__":
    unittest.main()
