from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import confidence
from starter.agent import Agent


ROOT = Path(__file__).resolve().parents[1]
# 0.891084, not the 0.891111 recorded elsewhere: src/retrieval.py's BM25
# ORDER BY now breaks exact bm25()-score ties by parent_asin instead of
# leaving them to SQLite's unspecified order (see src/retrieval.py::_run).
# Isolated to one session (public_0052, intent_override, rank 7 -> 8, same
# hit/turn) by toggling that single line with the rest of the tree held
# fixed. tests/test_frozen_baseline.py is the integration gate and is not
# edited here.
EXPECTED_SCORE = 0.891084


class SoftmaxEntropyTest(unittest.TestCase):
    def test_softmax_sums_to_one_and_preserves_order(self) -> None:
        probabilities = confidence.softmax([1.0, 3.0, 2.0])
        self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
        self.assertLess(probabilities[0], probabilities[2])
        self.assertLess(probabilities[2], probabilities[1])

    def test_softmax_is_stable_for_large_scores(self) -> None:
        probabilities = confidence.softmax([1000.0, 1000.5, 999.0])
        self.assertTrue(all(math.isfinite(p) for p in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0, places=9)

    def test_softmax_empty_is_empty(self) -> None:
        self.assertEqual(confidence.softmax([]), [])

    def test_entropy_of_uniform_distribution_is_log_n(self) -> None:
        n = 8
        uniform = [1.0 / n] * n
        self.assertAlmostEqual(confidence.entropy(uniform), math.log(n), places=9)

    def test_entropy_of_certain_distribution_is_zero(self) -> None:
        self.assertAlmostEqual(confidence.entropy([1.0, 0.0, 0.0]), 0.0, places=12)


class NormalizedConfidenceTest(unittest.TestCase):
    def test_empty_pool_is_zero(self) -> None:
        self.assertEqual(confidence.normalized_confidence([]), 0.0)

    def test_single_candidate_is_certain(self) -> None:
        self.assertEqual(confidence.normalized_confidence([0.42]), 1.0)

    def test_uniform_scores_are_low_confidence(self) -> None:
        self.assertLess(confidence.normalized_confidence([1.0] * 50), 0.05)

    def test_one_dominant_score_is_high_confidence(self) -> None:
        scores = [10.0] + [0.0] * 49
        self.assertGreater(confidence.normalized_confidence(scores), 0.9)

    def test_output_is_always_within_unit_interval(self) -> None:
        for scores in ([], [1.0], [1.0, 1.0], [-5.0, 3.0, 100.0], [0.0] * 10):
            value = confidence.normalized_confidence(scores)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class FusedPoolScoresParityTest(unittest.TestCase):
    """Proves Agent delegates confidence to the src.confidence module."""

    def _agent(self, directory: Path) -> Agent:
        path = directory / "catalog.jsonl"
        rows = [{"parent_asin": f"A{index:03d}", "title": f"Product {index}"} for index in range(30)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return Agent(path, enable_dense=False)

    def test_agent_delegates_to_member_four_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            route_results = {
                "bm25": [(f"A{index:03d}", 100.0 - index) for index in range(20)],
                "exact": [("A005", 1.0), ("A011", 1.0)],
                "bucket": [("A002", 1.0)],
                "dense": [("A007", 0.9), ("A000", 0.1)],
            }

            with patch(
                "src.confidence.confidence_from_route_results", return_value=0.37
            ) as integrated:
                self.assertEqual(agent._confidence_from_routes(route_results), 0.37)

            integrated.assert_called_once_with(
                route_results,
                exact_match_boost=agent.exact_match_boost,
                bucket_match_boost=agent.bucket_match_boost,
                dense_similarity_weight=agent.dense_similarity_weight,
                pool_limit=500,
            )

    def test_integrated_result_matches_pure_module_with_nonzero_dense_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                (Path(directory) / "catalog.jsonl"),
                enable_dense=False,
                dense_similarity_weight=0.20,
            )
            (Path(directory) / "catalog.jsonl").write_text(
                "".join(
                    json.dumps({"parent_asin": f"A{i:03d}", "title": f"P{i}"}) + "\n"
                    for i in range(15)
                ),
                encoding="utf-8",
            )
            route_results = {
                "bm25": [(f"A{i:03d}", 50.0 - i) for i in range(15)],
                "exact": [],
                "bucket": [],
                "dense": [("A003", 0.8), ("A009", 0.2)],
            }

            mine = confidence.confidence_from_route_results(
                route_results,
                exact_match_boost=agent.exact_match_boost,
                bucket_match_boost=agent.bucket_match_boost,
                dense_similarity_weight=agent.dense_similarity_weight,
            )
            theirs = agent._confidence_from_routes(route_results)

            self.assertEqual(mine, theirs)

    def test_empty_bm25_pool_yields_no_scores(self) -> None:
        self.assertEqual(
            confidence.fused_pool_scores({"bm25": [], "exact": [], "bucket": [], "dense": []}),
            [],
        )


class FrozenBaselineInvarianceTest(unittest.TestCase):
    """Acceptance test for the Day 4 confidence task: flipping ENABLE_CONFIDENCE
    on, with nothing consuming the value, must not move the official score by
    a single unit in the last decimal place. If it does, the layer is not
    read-only and that is a bug, not a tuning opportunity.

    Verified current frozen value: 0.891084 after the deterministic SQLite
    ASIN tie-break.
    The 0.877011 figure in CLAUDE.md's Day 3 snapshot predates the exact-route
    three-state constraint fix and is superseded on disk.
    """

    def test_enable_confidence_does_not_change_the_official_score(self) -> None:
        catalog_path = ROOT / "data" / "catalog.jsonl"
        public_path = ROOT / "data" / "public_set.jsonl"
        if not catalog_path.exists() or not public_path.exists():
            self.skipTest("frozen catalog/public set are not available")

        from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

        samples = load_jsonl(public_path)
        catalog_ids, categories, products = catalog_index(catalog_path)

        baseline = evaluate(
            Agent(catalog_path, enable_confidence=False),
            samples,
            catalog_ids,
            categories,
            products,
        )
        with_confidence = evaluate(
            Agent(catalog_path, enable_confidence=True),
            samples,
            catalog_ids,
            categories,
            products,
        )

        self.assertAlmostEqual(
            baseline["recommended_technical_score"], EXPECTED_SCORE, places=9
        )
        self.assertEqual(
            baseline["recommended_technical_score"],
            with_confidence["recommended_technical_score"],
        )
        self.assertEqual(baseline["sessions"], with_confidence["sessions"])


if __name__ == "__main__":
    unittest.main()
