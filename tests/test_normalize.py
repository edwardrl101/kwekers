from __future__ import annotations

import unittest

from src.dialog import SlotState
from src.normalize import detect_override, normalize_constraints


class FakeExactRoute:
    def query(self, values: list[str], limit: int = 1) -> list[tuple[str, float]]:
        return [("A001", 1.0)] if values == ["cotton"] else []


class LLMNormalizationTest(unittest.TestCase):
    def test_regex_match_never_calls_llm(self) -> None:
        def forbidden(*_args):
            raise AssertionError("LLM must not run after a regex match")

        result = normalize_constraints(
            "For that, what matters is: cotton; soft.",
            FakeExactRoute(),
            turn=2,
            llm_call=forbidden,
        )

        self.assertEqual(result, ["cotton", "soft"])

    def test_llm_candidates_must_match_the_exact_index(self) -> None:
        result = normalize_constraints(
            "Something breathable, with mostly natural fibers.",
            FakeExactRoute(),
            turn=2,
            llm_call=lambda *_args: '["cotton", "invented feature"]',
        )

        self.assertEqual(result, ["cotton"])

    def test_invalid_llm_json_falls_back_to_no_constraints(self) -> None:
        result = normalize_constraints(
            "Something breathable.",
            FakeExactRoute(),
            llm_call=lambda *_args: "cotton",
        )

        self.assertEqual(result, [])

    def test_known_no_constraint_templates_never_call_llm(self) -> None:
        def forbidden(*_args):
            raise AssertionError("known wrapper must not call the LLM")

        for message in (
            "I'm looking for Shirts, but I'm still exploring.",
            "I don't have a preference for color; please use your judgment.",
            "I don't have an additional preference for material.",
        ):
            self.assertEqual(
                normalize_constraints(
                    message, FakeExactRoute(), llm_call=forbidden
                ),
                [],
            )

    def test_override_regex_precedes_llm(self) -> None:
        def forbidden(*_args):
            raise AssertionError("LLM must not run after a regex match")

        self.assertTrue(
            detect_override(
                "Actually, ignore my earlier preference. What I need is: wool.",
                llm_call=forbidden,
            )
        )

    def test_override_llm_is_conservative(self) -> None:
        self.assertTrue(
            detect_override(
                "Forget the previous style; cotton is the priority now.",
                llm_call=lambda *_args: "true",
            )
        )
        self.assertFalse(
            detect_override(
                "Cotton would also be nice.", llm_call=lambda *_args: "maybe"
            )
        )

    def test_ordinary_messages_do_not_trigger_override_llm(self) -> None:
        def forbidden(*_args):
            raise AssertionError("ordinary messages must not call the LLM")

        self.assertFalse(detect_override("Cotton would also be nice.", llm_call=forbidden))

    def test_external_override_demotes_old_slot_and_adds_validated_value(self) -> None:
        state = SlotState("session")
        state.update("I'm looking for Shirts. Department: Womens", 1)

        added = state.add_external_constraints(
            ["cotton"], 3, is_override=True
        )

        self.assertEqual([item.text for item in added], ["cotton"])
        self.assertEqual([item.text for item in state.soft_constraints], ["Department: Womens"])
        self.assertEqual(
            [item.text for item in state.constraints if item.active], ["cotton"]
        )


if __name__ == "__main__":
    unittest.main()
