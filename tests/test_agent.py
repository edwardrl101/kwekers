from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


class AgentShellTest(unittest.TestCase):
    def _catalog(self, root: Path, count: int = 20) -> Path:
        path = root / "catalog.jsonl"
        rows = [
            {"parent_asin": f"A{index:03d}", "title": f"Product {index}"}
            for index in range(count)
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_random_floor_always_returns_ten_and_asks_other(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            agent.reset("session", {})
            response = agent.respond("session", "I need shoes", 1, 10)
            ids = [item["parent_asin"] for item in response["recommendations"]]
            self.assertEqual(response["ask_attribute"], "other")
            self.assertEqual(len(ids), 10)
            self.assertEqual(len(set(ids)), 10)

    def test_respond_does_not_crash_without_reset_or_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "missing.jsonl"
            agent = Agent(catalog_path, enable_dense=False)
            response = agent.respond("unknown", "", 1, 10)
            self.assertIsInstance(response["message"], str)
            self.assertEqual(response["ask_attribute"], "other")
            self.assertEqual(len(response["recommendations"]), 10)

    def test_missing_dense_cache_never_triggers_an_automatic_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)))

            self.assertIsNone(agent._dense_route)
            self.assertIn("Dense cache not found", agent._route_errors["dense"])

    def test_broken_route_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            agent._route_exact = lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
            agent.reset("session", {})
            response = agent.respond("session", "test", 1, 10)
            self.assertEqual(len(response["recommendations"]), 10)

    def test_random_fill_is_stable_across_session_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            profile = {"summary": "same customer"}
            agent.reset("first", profile)
            agent.reset("second", profile)
            first = agent.respond("first", "message", 2, 10)["recommendations"]
            second = agent.respond("second", "message", 2, 10)["recommendations"]
            self.assertEqual(first, second)

    def test_route_adapters_delegate_and_preserve_scores(self) -> None:
        class FakeRoute:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def query(self, text: str, limit: int) -> list[tuple[str, float]]:
                self.calls.append((text, limit))
                return [("A001", 0.9), ("A002", 0.5)]

        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            route = FakeRoute()
            for attribute, adapter in (
                ("_bucket_route", agent._route_bucket),
                ("_exact_route", agent._route_exact),
                ("_bm25_route", agent._route_bm25),
                ("_dense_route", agent._route_dense),
            ):
                setattr(agent, attribute, route)
                self.assertEqual(
                    adapter({}, "query", 1),
                    [("A001", 0.9), ("A002", 0.5)],
                )

            self.assertEqual(route.calls, [("query", 500)] * 4)

    def test_route_collection_keeps_route_identity_and_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            agent._route_bucket = lambda *_: [("A001", 1.0)]
            agent._route_exact = lambda *_: [("A002", 0.9)]
            agent._route_bm25 = lambda *_: [("A003", 12.5)]
            agent._route_dense = lambda *_: [("A004", 0.75)]

            results = agent._route_candidates({}, "query", 1)

            self.assertEqual(results["bucket"], [("A001", 1.0)])
            self.assertEqual(results["exact"], [("A002", 0.9)])
            self.assertEqual(results["bm25"], [("A003", 12.5)])
            self.assertEqual(results["dense"], [("A004", 0.75)])

    def test_fusion_uses_only_bm25_pool_and_promotes_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            route_results = {
                "bm25": [(f"A{index:03d}", 100.0 - index) for index in range(10)],
                "exact": [("A003", 1.0), ("A019", 1.0)],
                "bucket": [],
                "dense": [("A019", 0.99)],
            }

            fused = agent._fuse_bm25_pool(route_results)

            self.assertEqual(fused[0], "A003")
            self.assertNotIn("A019", fused)
            self.assertEqual(set(fused), {f"A{index:03d}" for index in range(10)})

    def test_dense_evidence_can_promote_a_bm25_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            route_results = {
                "bm25": [(f"A{index:03d}", 100.0 - index) for index in range(10)],
                "exact": [],
                "bucket": [],
                "dense": [("A002", 0.9), ("A000", 0.1)],
            }

            fused = agent._fuse_bm25_pool(route_results)

            self.assertLess(fused.index("A002"), 2)
            self.assertNotEqual(fused, [f"A{index:03d}" for index in range(10)])

    def test_respond_builds_an_accumulated_query_across_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            queries: list[str] = []

            def bm25(_session: dict, query: str, _turn: int):
                queries.append(query)
                return [(f"A{index:03d}", 100.0 - index) for index in range(10)]

            agent._route_bm25 = bm25
            agent._route_bucket = lambda *_: []
            agent._route_exact = lambda *_: []
            agent._route_dense = lambda *_: []
            agent.reset("session", {})

            agent.respond(
                "session",
                "I'm looking for Shoes. A key requirement is: leather.",
                1,
                10,
            )
            agent.respond(
                "session",
                "For that, what matters is: color: black.",
                2,
                10,
            )

            self.assertEqual(queries[0], "Shoes leather")
            self.assertEqual(queries[1], "Shoes leather color: black")

    def test_accumulated_query_removes_overridden_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            agent.reset("session", {})
            session = agent._sessions["session"]

            first = agent._update_retrieval_context(
                session, "I'm looking for Shirts. Department: Womens", 1
            )
            overridden = agent._update_retrieval_context(
                session,
                "Actually, ignore my earlier preference. What I need is: wool.",
                3,
            )

            self.assertIn("Department: Womens", first)
            self.assertNotIn("Department: Womens", overridden)
            self.assertIn("wool", overridden)


if __name__ == "__main__":
    unittest.main()
