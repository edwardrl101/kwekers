import json
import tempfile
import unittest
from pathlib import Path

from src.buckets import (
    BucketRoute,
    _opening_message,
    analyze_sessions,
    bucket_band,
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


class LoadCatalogTests(unittest.TestCase):
    def test_loads_jsonl_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [product("A1", ["Women", "Shoes"]), product("A2", ["Men", "Shirts"])]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")
            self.assertEqual(list(load_catalog(str(path))), ["A1", "A2"])

    def test_reports_malformed_json_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(json.dumps(product("A1", ["Women"])) + "\n{bad}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"line 2"):
                load_catalog(str(path))

    def test_rejects_duplicate_asins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            row = json.dumps(product("A1", ["Women"]))
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate parent_asin"):
                load_catalog(str(path))


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

    def test_rejects_unrecognized_or_non_string_message(self):
        self.assertIsNone(extract_category_phrase("Show me walking shoes"))
        self.assertIsNone(extract_category_phrase(None))  # type: ignore[arg-type]


class BucketRouteTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "A1": product("A1", ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Walking"]),
            "A2": product("A2", ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Walking"]),
            "A3": product("A3", ["Clothing, Shoes & Jewelry", "Women", "Dresses"]),
        }
        self.route = BucketRoute(self.catalog)

    def test_query_normalizes_case_preserves_order_and_honors_limit(self):
        message = "i'M looking for   shoes walking, but i'm still exploring."
        self.assertEqual(self.route.query(message, 1), [("A1", 1.0)])
        self.assertEqual(self.route.query(message, 10), [("A1", 1.0), ("A2", 1.0)])

    def test_query_rejects_invalid_limit_and_missing_bucket(self):
        self.assertEqual(self.route.query("I'm looking for Shoes Walking, but I'm still exploring.", 0), [])
        self.assertEqual(self.route.query("I'm looking for Shoes Walking, but I'm still exploring.", -1), [])
        self.assertEqual(self.route.query("I'm looking for Unknown Bucket, but I'm still exploring."), [])

    def test_full_bucket_is_available_for_analysis(self):
        message = "I'm looking for Shoes Walking, but I'm still exploring."
        self.assertEqual(self.route.bucket_for_message(message), ("A1", "A2"))


class BucketBandTests(unittest.TestCase):
    def test_boundaries(self):
        expected = {0: "<=10", 10: "<=10", 11: "11-50", 50: "11-50", 51: "51-500", 500: "51-500", 501: ">500"}
        for size, band in expected.items():
            with self.subTest(size=size):
                self.assertEqual(bucket_band(size), band)


class OpeningAndAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.item = product("A1", ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Walking"])

    def test_opening_message_scenarios(self):
        buying = {"scenario_type": "buying", "intent_card": {"hard_constraints": ["leather"], "soft_preferences": ["soft"]}}
        browsing = {"scenario_type": "browsing", "intent_card": {"hard_constraints": ["leather"], "soft_preferences": ["soft"]}}
        override = {"scenario_type": "intent_override", "intent_card": {"hard_constraints": ["leather"], "soft_preferences": ["old style"]}}
        self.assertIn("A key requirement is: leather", _opening_message(buying, self.item))
        self.assertTrue(_opening_message(browsing, self.item).endswith("still exploring."))
        self.assertTrue(_opening_message(override, self.item).endswith("old style"))

    def test_analysis_separates_missing_products_from_missing_buckets(self):
        catalog = {"A1": self.item}
        route = BucketRoute(catalog)
        route.buckets.clear()
        sessions = [
            {"scenario_type": "browsing", "ground_truth": {"parent_asin": "MISSING"}},
            {"scenario_type": "browsing", "ground_truth": {"parent_asin": "A1"}},
        ]
        result = analyze_sessions(sessions, catalog, route)
        self.assertEqual(result["missing_products"], 1)
        self.assertEqual(result["missing_buckets"], 1)

    def test_analysis_reports_actual_cutoff_coverage_and_scenarios(self):
        catalog = {
            f"A{index}": product(
                f"A{index}",
                ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Walking"],
            )
            for index in range(1, 13)
        }
        route = BucketRoute(catalog)
        sessions = [
            {
                "sample_id": "one",
                "scenario_type": "buying",
                "ground_truth": {"parent_asin": "A3"},
            },
            {
                "sample_id": "two",
                "scenario_type": "browsing",
                "ground_truth": {"parent_asin": "A12"},
            },
        ]
        result = analyze_sessions(sessions, catalog, route)
        self.assertEqual(result["target_coverage"], 2)
        self.assertEqual(result["coverage_at"], {10: 1, 50: 2, 200: 2})
        self.assertEqual(result["target_position_median"], 7.5)
        self.assertEqual(result["scenario_stats"]["buying"]["coverage_at"][10], 1)
        self.assertEqual(result["scenario_stats"]["browsing"]["coverage_at"][10], 0)


if __name__ == "__main__":
    unittest.main()
