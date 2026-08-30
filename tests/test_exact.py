import unittest

from src.exact import ExactRoute, clean_constraint


class ExactRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.route = ExactRoute(
            {
                "A1": {
                    "title": "Cotton everyday shirt",
                    "features": ["Machine washable"],
                    "details": {"Fit": "Regular", "Color": "Black"},
                    "price": 20.0,
                },
                "A2": {
                    "title": "Cotton slim shirt",
                    "features": [],
                    "details": {"Fit": "Slim"},
                    "price": 30.0,
                },
                "A3": {
                    "title": "Wool regular shirt",
                    "features": [],
                    "details": {"Fit": "Regular"},
                    "price": 40.0,
                },
            }
        )

    def test_clean_constraint_matches_evaluator_shape(self) -> None:
        self.assertEqual(clean_constraint("  -- Machine   washable...  "), "Machine washable")

    def test_intersects_all_indexed_constraints(self) -> None:
        self.assertEqual(
            self.route.exact_matches(["cotton", "Fit: Regular"]),
            ["A1"],
        )

    def test_unindexed_constraint_does_not_veto_known_evidence(self) -> None:
        self.assertEqual(
            self.route.exact_matches(["cotton", "unknown constraint"]),
            ["A1", "A2"],
        )

    def test_partial_color_index_is_not_used_as_exact_evidence(self) -> None:
        self.assertEqual(
            self.route.exact_matches(["cotton", "color: black"]),
            ["A1", "A2"],
        )

    def test_all_unindexed_constraints_return_no_exact_evidence(self) -> None:
        self.assertEqual(self.route.exact_matches(["color: chartreuse"]), [])

    def test_supported_constraint_with_zero_matches_vetoes_intersection(self) -> None:
        self.assertEqual(
            self.route.exact_matches(["cotton", "budget around $10"]),
            [],
        )
    def test_query_intersects_accumulated_constraint_list(self) -> None:
        self.assertEqual(
            self.route.query(["cotton", "Fit: Regular"]),
            [("A1", 1.0)],
        )

    def test_query_honours_limit_and_catalog_order(self) -> None:
        self.assertEqual(self.route.query("cotton", limit=1), [("A1", 1.0)])

    def test_non_positive_limit_returns_nothing(self) -> None:
        self.assertEqual(self.route.query("cotton", limit=0), [])


if __name__ == "__main__":
    unittest.main()
