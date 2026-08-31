from __future__ import annotations

import unittest

import numpy as np

from src.retrieval import BM25Route, _top_k, extract_constraints, norm, strip_boilerplate


class NormalizationTest(unittest.TestCase):
    def test_nfkc_folds_fullwidth_and_decorative_variants(self) -> None:
        # Fullwidth "A" (U+FF21) and a circled "1" (U+2460) both fold to
        # plain ASCII under NFKC - this is why norm() matters for ~20% of
        # catalog rows that carry non-ASCII in features/description.
        self.assertEqual(norm("ＡＢＣ"), "abc")
        self.assertIn("1", norm("①"))

    def test_strips_decorative_symbols_and_collapses_whitespace(self) -> None:
        self.assertEqual(norm("Black  ***  Cotton!!\tShirt"), "black cotton shirt")

    def test_empty_and_none_like_input_is_safe(self) -> None:
        self.assertEqual(norm(""), "")


class ConstraintExtractionTest(unittest.TestCase):
    def test_extracts_key_requirement_marker(self) -> None:
        message = "I'm looking for Sandals. A key requirement is: 95% Cotton, 5% Spandex."
        self.assertEqual(extract_constraints(message), ["95% cotton, 5% spandex"])

    def test_extracts_multiple_matters_constraints(self) -> None:
        message = "For that, what matters is: cotton; color: black."
        self.assertEqual(extract_constraints(message), ["cotton", "color: black"])

    def test_browsing_opener_falls_back_to_stripped_boilerplate(self) -> None:
        message = "I'm looking for Women Dresses, but I'm still exploring."
        self.assertEqual(extract_constraints(message), ["women dresses"])

    def test_strip_boilerplate_removes_every_known_phrase(self) -> None:
        message = "I don't have a preference for color; please use your judgment."
        self.assertEqual(strip_boilerplate(message), "color")


class TopKDeterminismTest(unittest.TestCase):
    """np.argpartition/np.argsort default tie-breaking is unspecified; a tie
    sitting exactly at the `limit` boundary can change WHICH candidates are
    returned, not just their order. _top_k must be immune to this."""

    def test_tied_scores_break_deterministically_by_asin(self) -> None:
        asins = np.array(["B003", "B001", "A999", "C000"])
        scores = np.array([5.0, 5.0, 5.0, 1.0])

        result = _top_k(asins, scores, limit=3)

        self.assertEqual([asin for asin, _score in result], ["A999", "B001", "B003"])

    def test_tie_at_the_cutoff_selects_the_lexicographically_smaller_asin(self) -> None:
        asins = np.array(["Z999", "A111", "M555"])
        scores = np.array([9.0, 9.0, 9.0])

        result = _top_k(asins, scores, limit=2)

        self.assertEqual([asin for asin, _score in result], ["A111", "M555"])

    def test_matches_prior_behavior_when_no_ties_exist(self) -> None:
        asins = np.array([f"P{i:02d}" for i in range(10)])
        scores = np.array([float(10 - i) for i in range(10)])

        result = _top_k(asins, scores, limit=4)

        self.assertEqual(
            [asin for asin, _score in result], ["P00", "P01", "P02", "P03"]
        )

    def test_zero_or_negative_scores_are_excluded(self) -> None:
        asins = np.array(["A", "B", "C"])
        scores = np.array([1.0, 0.0, -1.0])

        result = _top_k(asins, scores, limit=3)

        self.assertEqual([asin for asin, _score in result], ["A"])


class BM25TieDeterminismTest(unittest.TestCase):
    """Two catalog rows with byte-identical indexed text force an exact
    bm25() score tie. Before the ORDER BY s, parent_asin fix, SQLite's
    tie-break order was unspecified; this pins the now-guaranteed order."""

    def _catalog(self) -> dict[str, dict]:
        twin_row = {
            "title": "Cotton Everyday Crew Neck Shirt",
            "categories": ["Clothing", "Shirts"],
            "features": ["Machine washable", "Breathable"],
            "details": {"Fit": "Regular"},
            "store": "Example",
            "description": ["A comfortable everyday shirt."],
        }
        return {
            "TIE_B": dict(twin_row),
            "TIE_A": dict(twin_row),
            "OTHER": {
                "title": "Leather Hiking Boots",
                "categories": ["Clothing", "Boots"],
                "features": ["Waterproof"],
                "details": {"Fit": "Wide"},
                "store": "Example",
                "description": ["Rugged outdoor boots."],
            },
        }

    def test_exact_score_tie_breaks_by_ascending_parent_asin(self) -> None:
        route = BM25Route(self._catalog())
        results = route.query("cotton everyday crew neck shirt", limit=10)
        ids = [asin for asin, _score in results]

        self.assertIn("TIE_A", ids)
        self.assertIn("TIE_B", ids)
        tied_scores = {score for asin, score in results if asin in ("TIE_A", "TIE_B")}
        self.assertEqual(len(tied_scores), 1, "test fixture did not produce a real tie")
        self.assertLess(ids.index("TIE_A"), ids.index("TIE_B"))

    def test_repeated_queries_return_identical_order(self) -> None:
        route = BM25Route(self._catalog())
        first = route.query("cotton shirt", limit=10)
        second = route.query("cotton shirt", limit=10)
        self.assertEqual(first, second)


class BM25RouteBasicTest(unittest.TestCase):
    def test_query_ranks_lexical_match_above_unrelated_product(self) -> None:
        catalog = {
            "MATCH": {
                "title": "Black Leather Running Shoes",
                "categories": ["Clothing", "Shoes"],
                "features": ["Cushioned sole"],
                "details": {},
                "store": "Example",
                "description": [],
            },
            "OTHER": {
                "title": "Silk Evening Dress",
                "categories": ["Clothing", "Dresses"],
                "features": [],
                "details": {},
                "store": "Example",
                "description": [],
            },
        }
        route = BM25Route(catalog)
        results = route.query("I'm looking for Shoes. A key requirement is: leather.", limit=10)
        self.assertEqual(results[0][0], "MATCH")

    def test_empty_query_returns_no_results(self) -> None:
        route = BM25Route({"A": {"title": "Item"}})
        self.assertEqual(route.query("", limit=10), [])


if __name__ == "__main__":
    unittest.main()
