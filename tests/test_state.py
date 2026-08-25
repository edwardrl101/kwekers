from __future__ import annotations

import unittest

from shopping_copilot.state import ConversationState, Intent


class ConversationStateTest(unittest.TestCase):
    def test_record_user_turn_accumulates_history(self) -> None:
        state = ConversationState.start("session-1", {"summary": "prefers comfort"})

        state.record_user_turn(1, "I want shoes")
        state.record_user_turn(2, "Running shoes under $100")

        self.assertEqual(state.turn_number, 2)
        self.assertEqual(state.current_search_goal, "Running shoes under $100")
        self.assertEqual(state.previous_queries, ["I want shoes", "Running shoes under $100"])
        self.assertEqual([turn.turn_number for turn in state.turns], [1, 2])

    def test_record_agent_response_updates_latest_turn(self) -> None:
        state = ConversationState.start("session-1", {"summary": "prefers comfort"})
        state.record_user_turn(1, "I want shoes")

        state.record_agent_response(
            "Here are the closest matches I found.",
            None,
            ["A", "B"],
        )

        self.assertEqual(state.candidate_count, 2)
        self.assertEqual(state.turns[-1].recommendations, ("A", "B"))
        self.assertEqual(state.turns[-1].agent_message, "Here are the closest matches I found.")

    def test_begin_new_goal_clears_constraints_for_override(self) -> None:
        state = ConversationState.start("session-1", {"summary": "prefers comfort"})
        state.constraints.category = "running shoes"
        state.constraints.color = "black"
        state.constraints.hard_constraints = ("black", "under $100")

        state.begin_new_goal("Actually, let's look for jackets", Intent.BUYING)

        self.assertEqual(state.intent, Intent.BUYING)
        self.assertEqual(state.current_search_goal, "Actually, let's look for jackets")
        self.assertIsNone(state.constraints.category)
        self.assertIsNone(state.constraints.color)
        self.assertEqual(state.constraints.hard_constraints, ())


if __name__ == "__main__":
    unittest.main()
