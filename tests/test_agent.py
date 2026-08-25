from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


class AgentStateIntegrationTest(unittest.TestCase):
    def test_reset_and_respond_record_typed_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text(
                json.dumps({
                    "parent_asin": "A",
                    "title": "Black running shoes",
                    "categories": ["Clothing", "Shoes"],
                    "features": ["lightweight"],
                    "details": {"department": "mens"},
                    "store": "Example",
                    "description": ["good for training"],
                }) + "\n",
                encoding="utf-8",
            )

            agent = Agent(catalog_path)
            agent.reset("session-1", {"summary": "prefers comfort"})
            response = agent.respond("session-1", "I want running shoes", 1, 10)

            self.assertEqual(response["ask_attribute"], None)
            self.assertEqual(response["recommendations"], [{"parent_asin": "A"}])
            state = agent._sessions["session-1"]
            self.assertEqual(state.turn_number, 1)
            self.assertEqual(state.current_search_goal, "I want running shoes")
            self.assertEqual(state.turns[-1].recommendations, ("A",))


if __name__ == "__main__":
    unittest.main()
