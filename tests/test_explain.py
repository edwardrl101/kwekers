from __future__ import annotations

import unittest

from src.explain import deterministic_explanation, explain


class ExplanationTest(unittest.TestCase):
    def test_deterministic_message_uses_constraints_and_confidence(self) -> None:
        message = deterministic_explanation(["cotton", "budget around $40"], 0.8)

        self.assertIn("cotton", message)
        self.assertIn("budget around $40", message)
        self.assertIn("strong match", message)

    def test_llm_failure_returns_deterministic_fallback(self) -> None:
        message = explain(
            ["cotton"],
            [{"title": "Shirt", "price": 20}],
            0.2,
            use_llm=True,
            llm_call=lambda *_args: None,
        )

        self.assertEqual(message, deterministic_explanation(["cotton"], 0.2))

    def test_llm_message_is_length_limited(self) -> None:
        message = explain(
            [],
            [],
            0.0,
            use_llm=True,
            llm_call=lambda *_args: "x" * 241,
        )

        self.assertEqual(message, deterministic_explanation([], 0.0))


if __name__ == "__main__":
    unittest.main()
