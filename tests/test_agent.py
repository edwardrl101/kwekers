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

    def test_freshness_avoids_repeats_across_normal_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                self._catalog(Path(directory), count=30), enable_dense=False
            )
            agent._route_bucket = None
            agent._exact_route = None
            agent._dense_route = None
            agent._route_bm25 = lambda *_: [
                (f"A{index:03d}", 100.0 - index) for index in range(30)
            ]
            agent.reset("session", {})

            first = agent.respond(
                "session",
                "I'm looking for Shoes. A key requirement is: leather.",
                1,
                10,
            )
            second = agent.respond(
                "session",
                "For that, what matters is: soft.",
                2,
                10,
            )
            first_ids = {
                item["parent_asin"] for item in first["recommendations"]
            }
            second_ids = {
                item["parent_asin"] for item in second["recommendations"]
            }

            self.assertEqual(len(first_ids), 10)
            self.assertEqual(len(second_ids), 10)
            self.assertTrue(first_ids.isdisjoint(second_ids))
            self.assertEqual(
                agent._sessions["session"]["shown"], first_ids | second_ids
            )

    def test_override_clears_freshness_before_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                self._catalog(Path(directory), count=30), enable_dense=False
            )
            agent._route_bucket = None
            agent._route_exact = None
            agent._dense_route = None
            agent._route_bm25 = lambda *_: [
                (f"A{index:03d}", 100.0 - index) for index in range(30)
            ]
            agent.reset("session", {})

            first = agent.respond(
                "session",
                "I'm looking for Shirts. Department: Womens",
                1,
                10,
            )
            agent.respond(
                "session",
                "For that, what matters is: color: black.",
                2,
                10,
            )
            override = agent.respond(
                "session",
                "Actually, ignore my earlier preference. What I need is: wool.",
                3,
                10,
            )
            first_ids = [item["parent_asin"] for item in first["recommendations"]]
            override_ids = [
                item["parent_asin"] for item in override["recommendations"]
            ]

            self.assertEqual(override_ids, first_ids)
            self.assertEqual(agent._sessions["session"]["shown"], set(first_ids))

    def test_random_fill_excludes_shown_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(
                self._catalog(Path(directory), count=25), enable_dense=False
            )
            excluded = {f"A{index:03d}" for index in range(10)}
            session = {"seed_key": "freshness-test", "shown": excluded}

            result = agent._random_fill(["A000", "A010"], session, 2)

            self.assertEqual(len(result), 10)
            self.assertEqual(len(set(result)), 10)
            self.assertTrue(set(result).isdisjoint(excluded))
            self.assertEqual(result[0], "A010")

    def test_reset_initializes_independent_freshness_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            agent.reset("first", {})
            agent.reset("second", {})

            agent._sessions["first"]["shown"].add("A000")

            self.assertEqual(agent._sessions["first"]["shown"], {"A000"})
            self.assertEqual(agent._sessions["second"]["shown"], set())

    def test_route_adapters_delegate_and_preserve_scores(self) -> None:
        class FakeRoute:
            def __init__(self) -> None:
                self.calls: list[tuple[str | list[str], int]] = []

            def query(
                self, text: str | list[str], limit: int
            ) -> list[tuple[str, float]]:
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

    def test_exact_adapter_passes_all_active_constraints_in_one_call(self) -> None:
        class FakeExactRoute:
            def __init__(self) -> None:
                self.calls: list[tuple[str | list[str], int]] = []

            def query(
                self, text: str | list[str], limit: int
            ) -> list[tuple[str, float]]:
                self.calls.append((text, limit))
                return [("A003", 1.0)]

        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)), enable_dense=False)
            route = FakeExactRoute()
            agent._exact_route = route

            result = agent._route_exact(
                {"active_constraints": ["cotton", "soft"]},
                "ignored cumulative query",
                2,
            )

            self.assertEqual(result, [("A003", 1.0)])
            self.assertEqual(route.calls, [(["cotton", "soft"], 500)])

    def test_real_exact_route_intersection_reaches_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            rows = [
                {
                    "parent_asin": f"A{index:03d}",
                    "title": f"Product {index}",
                    "features": (
                        ["cotton", "soft"]
                        if index == 3
                        else ["cotton"]
                        if index == 4
                        else ["soft"]
                        if index == 5
                        else ["unrelated"]
                    ),
                }
                for index in range(20)
            ]
            catalog_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            agent = Agent(catalog_path, enable_dense=False)
            agent._route_bucket = None
            agent._dense_route = None
            agent._route_bm25 = lambda *_: [
                (f"A{index:03d}", 100.0 - index) for index in range(10)
            ]
            session = {"active_constraints": ["cotton", "soft"]}

            route_results = agent._route_candidates(session, "query", 2)
            fused = agent._fuse_bm25_pool(route_results)

            self.assertEqual(route_results["exact"], [("A003", 1.0)])
            self.assertEqual(fused[0], "A003")
            session["active_constraints"] = ["cotton", "missing"]
            self.assertEqual(agent._route_exact(session, "query", 2), [])

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
