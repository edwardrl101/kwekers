import json
import tempfile
import unittest
from pathlib import Path

from src.buckets import (
    BucketRoute,
    _opening_message,
    analyze_sessions,
    bucket_band,
    evaluate_browsing_filter,
    extract_category_phrase,
    load_catalog,
)


def product(asin: str, categories: list[str]) -> dict:
    return {
        "parent_asin": asin,
        "title": "Example",
        "categories": categories,
        "features": ["cotton", "soft"],
        "details": {},
    }


class FakeBM25:
    def __init__(self, results: list[str]):
        self.results = results

    def query(self, _message: str, limit: int = 500) -> list[tuple[str, float]]:
        return [(asin, 1.0) for asin in self.results[:limit]]


class LoadCatalogTests(unittest.TestCase):
    def test_loads_jsonl_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [product("A1", ["Women", "Shoes"]), product("A2", ["Men", "Shirts"])]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")
            self.assertEqual(list(load_catalog(path)), ["A1", "A2"])

    def test_reports_malformed_json_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps(product("A1", ["Women"])) + "\n{bad}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"line 2"):
                load_catalog(path)

    def test_rejects_duplicate_asins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            row = json.dumps(product("A1", ["Women"]))
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate parent_asin"):
                load_catalog(path)


class ExtractCategoryTests(unittest.TestCase):
    def test_extracts_all_opening_shapes(self):
        cases = {
            "I'm looking for Shoes Walking. A key requirement is: leather.": "Shoes Walking",
            "I'm looking for Women Dresses, but I'm still exploring.": "Women Dresses",
            "I'm looking for Watches Wrist Watches. I used to prefer black.": "Watches Wrist Watches",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(extract_category_phrase(message), expected)

    def test_handles_whitespace_and_rejects_unknown_messages(self):
        message = "  I'm   looking for   Shoes Walking, but I'm still exploring.  "
        self.assertEqual(extract_category_phrase(message), "Shoes Walking")
        self.assertIsNone(extract_category_phrase("completely unrelated message"))
        self.assertIsNone(extract_category_phrase(None))  # type: ignore[arg-type]


class BucketRouteTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "A1": product("A1", ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Walking"]),
            "A2": product("A2", ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Walking"]),
            "A3": product("A3", ["Clothing, Shoes & Jewelry", "Women", "Dresses"]),
        }
        self.route = BucketRoute(self.catalog)
        self.walking = "I'm looking for Shoes Walking, but I'm still exploring."

    def test_query_preserves_day_one_behavior(self):
        self.assertEqual(self.route.query(self.walking, 1), [("A1", 1.0)])
        self.assertEqual(self.route.bucket_for_message(self.walking), ("A1", "A2"))

    def test_filter_preserves_bm25_order(self):
        pool = ["A3", "A2", "unknown", "A1"]
        self.assertEqual(self.route.filter_by_category(pool, self.walking), ["A2", "A1"])

    def test_filter_never_empties_a_usable_pool(self):
        empty: list[str] = []
        self.assertIs(self.route.filter_by_category(empty, self.walking), empty)
        pool = ["A3", "unknown"]
        self.assertIs(self.route.filter_by_category(pool, self.walking), pool)
        self.assertIs(
            self.route.filter_by_category(pool, "completely unrelated message"), pool
        )
        missing = "I'm looking for Unknown Bucket, but I'm still exploring."
        self.assertIs(self.route.filter_by_category(pool, missing), pool)

    def test_filter_survivors_belong_to_bucket_and_keep_order(self):
        pool = ["A3", "A2", "A1"]
        filtered = self.route.filter_by_category(pool, self.walking)
        bucket = self.route.bucket_sets["shoes walking"]
        self.assertTrue(all(asin in bucket for asin in filtered))
        indices = [pool.index(asin) for asin in filtered]
        self.assertEqual(indices, sorted(indices))


class BucketBandTests(unittest.TestCase):
    def test_boundaries(self):
        expected = {
            0: "<=10", 10: "<=10", 11: "11-50", 50: "11-50",
            51: "51-500", 500: "51-500", 501: ">500",
        }
        for size, band in expected.items():
            with self.subTest(size=size):
                self.assertEqual(bucket_band(size), band)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            f"A{index}": product(
                f"A{index}",
                ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Walking"],
            )
            for index in range(1, 13)
        }
        self.catalog["D1"] = product(
            "D1", ["Clothing, Shoes & Jewelry", "Women", "Dresses"]
        )
        self.route = BucketRoute(self.catalog)

    def test_opening_message_scenarios(self):
        item = self.catalog["A1"]
        buying = {"scenario_type": "buying", "intent_card": {"hard_constraints": ["leather"], "soft_preferences": ["soft"]}}
        browsing = {"scenario_type": "browsing", "intent_card": {"hard_constraints": ["leather"], "soft_preferences": ["soft"]}}
        override = {"scenario_type": "intent_override", "intent_card": {"hard_constraints": ["leather"], "soft_preferences": ["old style"]}}
        self.assertIn("A key requirement is: leather", _opening_message(buying, item))
        self.assertTrue(_opening_message(browsing, item).endswith("still exploring."))
        self.assertTrue(_opening_message(override, item).endswith("old style"))

    def test_small_bucket_diagnostic(self):
        sessions = [
            {"scenario_type": "boundary", "ground_truth": {"parent_asin": "D1"}},
            {"scenario_type": "browsing", "ground_truth": {"parent_asin": "A1"}},
        ]
        result = analyze_sessions(sessions, self.catalog, self.route)
        self.assertEqual(result["small_bucket_sessions"], 1)
        self.assertEqual(result["small_bucket_targets"], 1)
        self.assertEqual(result["browsing_bands"]["11-50"], 1)

    def test_filter_metrics_count_preservation_and_improvement(self):
        sessions = [
            {"scenario_type": "browsing", "ground_truth": {"parent_asin": "A1"}},
            {"scenario_type": "buying", "ground_truth": {"parent_asin": "D1"}},
        ]
        results = ["D1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A1"]
        result = evaluate_browsing_filter(
            sessions, self.catalog, self.route, FakeBM25(results)
        )
        self.assertEqual(result["sessions"], 1)
        self.assertEqual(result["before_hits_at_10"], 0)
        self.assertEqual(result["after_hits_at_10"], 1)
        self.assertEqual(result["targets_in_pool"], 1)
        self.assertEqual(result["targets_preserved"], 1)
        self.assertEqual(result["targets_removed"], 0)


if __name__ == "__main__":
    unittest.main()
