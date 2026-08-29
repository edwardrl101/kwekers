from __future__ import annotations

import unittest

from src.exact import ExactRoute


class ExactRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.route = ExactRoute(
            {
                "A003": {"features": ["cotton", "soft"]},
                "A001": {"features": ["cotton"]},
                "A002": {"features": ["cotton", "soft"]},
                "A004": {"features": ["soft"]},
            }
        )

    def test_query_intersects_all_constraints(self) -> None:
        self.assertEqual(
            self.route.query(["cotton", "soft"]),
            [("A002", 1.0), ("A003", 1.0)],
        )

    def test_query_is_deterministic_before_limit(self) -> None:
        self.assertEqual(
            self.route.query(["cotton"], limit=2),
            [("A001", 1.0), ("A002", 1.0)],
        )

    def test_unmatched_constraint_forces_empty_intersection(self) -> None:
        self.assertEqual(self.route.query(["cotton", "missing"]), [])


if __name__ == "__main__":
    unittest.main()
