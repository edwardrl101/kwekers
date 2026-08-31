from __future__ import annotations

import json
import unittest

from src.dialog import SlotState
from src.exact import ExactRoute
from src.normalize import _catalog_candidates, detect_override, normalize_constraints


def json_response(value: str) -> str:
    return json.dumps({"constraints": [value]})


class LLMNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.route = ExactRoute(
            {
                "A001": {
                    "title": "Cotton shirt",
                    "features": ["95% Cotton, 5% Spandex", "Machine washable"],
                    "details": {"Fit": "Regular"},
                    "price": 20.0,
                },
                "A002": {
                    "title": "Wool shirt",
                    "features": ["Dry clean only"],
                    "details": {"Fit": "Slim"},
                    "price": 40.0,
                },
            }
        )

    def test_regex_match_never_calls_llm(self) -> None:
        def forbidden(*_args):
            raise AssertionError("LLM must not run after a regex match")

        result = normalize_constraints(
            "For that, what matters is: cotton; soft.",
            self.route,
            turn=2,
            llm_call=forbidden,
        )

        self.assertEqual(result, ["cotton", "soft"])

    def test_llm_candidates_must_match_the_exact_index(self) -> None:
        result = normalize_constraints(
            "Something made from mostly cotton with a bit of stretch.",
            self.route,
            turn=2,
            llm_call=lambda *_args: '{"constraints":["cotton","invented feature"]}',
        )

        self.assertEqual(result, ["cotton"])

    def test_invalid_llm_json_falls_back_to_no_constraints(self) -> None:
        result = normalize_constraints(
            "Something breathable.",
            self.route,
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
                    message, self.route, llm_call=forbidden
                ),
                [],
            )

    def test_llm_can_only_select_from_catalog_candidate_list(self) -> None:
        result = normalize_constraints(
            "A mostly cotton option, please.",
            self.route,
            llm_call=lambda *_args: '{"constraints":["Machine washable"]}',
        )

        self.assertEqual(result, [])

    def test_candidate_list_is_deterministic_and_catalog_derived(self) -> None:
        message = "Mostly cotton, with a little stretch, and machine safe."
        first = _catalog_candidates(message, self.route)
        second = _catalog_candidates(message, self.route)

        self.assertEqual(first, second)
        self.assertIn("cotton", [item.casefold() for item in first])
        self.assertIn("95% cotton, 5% spandex", [item.casefold() for item in first])
        self.assertIn("machine washable", [item.casefold() for item in first])

    def test_semantic_level4_examples_recover_supported_constraints(self) -> None:
        cases = (
            (
                "I want something mostly cotton with a bit of stretch.",
                "cotton",
            ),
            (
                "It needs to be safe for the washing machine.",
                "Machine washable",
            ),
            (
                "I can spend no more than about 20 dollars.",
                "budget around $20",
            ),
        )

        for message, selected in cases:
            with self.subTest(message=message):
                result = normalize_constraints(
                    message,
                    self.route,
                    llm_call=lambda *_args, value=selected: json_response(value),
                )
                self.assertEqual(result, [selected])

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

    def test_normal_public_words_do_not_trigger_override_llm(self) -> None:
        def forbidden(*_args):
            raise AssertionError("ordinary constraints must not call the LLM")

        for message in (
            "Available now in cotton.",
            "A quick-change buckle would be useful.",
            "For that, what matters is: switch closure.",
        ):
            with self.subTest(message=message):
                # A constraint containing an isolated cue word is not itself
                # evidence that the shopper replaced an earlier preference.
                self.assertFalse(detect_override(message, llm_call=forbidden))

    def test_paraphrased_override_cues_can_reach_llm_fallback(self) -> None:
        for message in (
            "On second thought, wool would be better.",
            "Scratch that; make it wool.",
            "I no longer want cotton.",
        ):
            with self.subTest(message=message):
                self.assertTrue(
                    detect_override(message, llm_call=lambda *_args: "true")
                )

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

    def test_what_i_need_override_demotes_previous_preference(self) -> None:
        state = SlotState("session")
        state.update("A key requirement is: cotton.", 1)

        added = state.update("What I need is: wool.", 2)

        self.assertEqual([item.text for item in added], ["wool"])
        self.assertEqual([item.text for item in state.constraints if item.active], ["wool"])
        self.assertEqual([item.text for item in state.soft_constraints], ["cotton"])


if __name__ == "__main__":
    unittest.main()
