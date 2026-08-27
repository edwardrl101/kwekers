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
            agent = Agent(self._catalog(Path(directory)))
            agent.reset("session", {})
            response = agent.respond("session", "I need shoes", 1, 10)
            ids = [item["parent_asin"] for item in response["recommendations"]]
            self.assertEqual(response["ask_attribute"], "other")
            self.assertEqual(len(ids), 10)
            self.assertEqual(len(set(ids)), 10)

    def test_respond_does_not_crash_without_reset_or_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(Path(directory) / "missing.jsonl")
            response = agent.respond("unknown", "", 1, 10)
            self.assertIsInstance(response["message"], str)
            self.assertEqual(response["ask_attribute"], "other")
            self.assertEqual(len(response["recommendations"]), 10)

    def test_broken_route_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)))
            agent._route_exact = lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
            agent.reset("session", {})
            response = agent.respond("session", "test", 1, 10)
            self.assertEqual(len(response["recommendations"]), 10)

    def test_random_fill_is_stable_across_session_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = Agent(self._catalog(Path(directory)))
            profile = {"summary": "same customer"}
            agent.reset("first", profile)
            agent.reset("second", profile)
            first = agent.respond("first", "message", 2, 10)["recommendations"]
            second = agent.respond("second", "message", 2, 10)["recommendations"]
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
