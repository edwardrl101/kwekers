from __future__ import annotations

import unittest

from src.dialog import (
    POLICY_ALWAYS_OTHER,
    POLICY_OTHER_TWICE_ROTATE,
    QuestionPolicy,
    SAFE_ASK_ATTRIBUTES,
    SlotState,
    policy_evidence_summary,
)
from src.explain import deterministic_explanation, explain


class Member5Day4Test(unittest.TestCase):
    def test_default_policy_is_constant_other(self) -> None:
        state = SlotState("session")
        policy = QuestionPolicy()
        for _ in range(6):
            ask = policy.next_attribute(state)
            self.assertEqual(ask, "other")
            state.record_ask(ask)

    def test_rotation_policy_is_exercised_fallback(self) -> None:
        state = SlotState("session")
        policy = QuestionPolicy(POLICY_OTHER_TWICE_ROTATE)
        sequence = []
        for _ in range(7):
            ask = policy.next_attribute(state)
            sequence.append(ask)
            state.record_ask(ask)
        self.assertEqual(
            sequence,
            ["other", "other", "feature", "material", "color", "style", "feature"],
        )

    def test_both_policies_only_emit_safe_attributes(self) -> None:
        for mode in (POLICY_ALWAYS_OTHER, POLICY_OTHER_TWICE_ROTATE):
            state = SlotState(mode)
            policy = QuestionPolicy(mode)
            for _ in range(20):
                ask = policy.next_attribute(state)
                self.assertIn(ask, SAFE_ASK_ATTRIBUTES)
                self.assertNotIn(ask, {"brand", "category", None})
                state.record_ask(ask)

    def test_policy_summary_uses_equivalence_claim(self) -> None:
        summary = policy_evidence_summary()
        self.assertIn("0.0015", summary)
        self.assertIn("±0.019", summary)
        self.assertIn("equivalent", summary)

    def test_explanation_is_constraint_and_confidence_aware(self) -> None:
        high = deterministic_explanation(["cotton", "budget around $45"], 0.9)
        low = deterministic_explanation([], 0.1)
        self.assertIn("cotton", high)
        self.assertIn("budget around $45", high)
        self.assertIn("strong match", high)
        self.assertIn("still narrowing down", low)
        self.assertTrue(high.endswith("What else matters most to you?"))

    def test_llm_failure_is_zero_risk_fallback(self) -> None:
        expected = deterministic_explanation(["leather"], 0.4)
        actual = explain(
            ["leather"],
            [{"title": "Leather Belt", "price": 20}],
            0.4,
            use_llm=True,
            llm_call=lambda *_args: None,
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
