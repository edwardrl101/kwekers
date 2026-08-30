import unittest

from src.dedup import (
    count_duplicate_pairs,
    deduplicated_top_k,
    diverse_top_k,
    normalize_title,
    suppress_duplicates,
    title_similarity,
    variant_tokens,
)


class DedupTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {
            "A": {"title": "Women's Classic Cotton T-Shirt - Black, Size M"},
            "B": {"title": "Women's Classic Cotton T Shirt Black Size M"},
            "C": {"title": "Leather Walking Boots"},
            "D": {"title": "Silk Evening Dress"},
            "E": {"title": "Another Unrelated Product"},
        }

    def test_normalize_title_is_conservative_and_deterministic(self):
        self.assertEqual(
            normalize_title(" Women's  Classic T-Shirt! "),
            "women s classic t shirt",
        )

    def test_jaccard_similarity_handles_empty_and_identical_titles(self):
        self.assertEqual(title_similarity("", ""), 0.0)
        self.assertEqual(title_similarity("Cotton Shirt", "cotton shirt"), 1.0)
        self.assertEqual(title_similarity("Cotton Shirt", "Leather Boots"), 0.0)

    def test_suppression_preserves_slot_one_and_order(self):
        pool = ["A", "B", "C", "D"]
        self.assertEqual(suppress_duplicates(pool, self.catalog, 0.8), ["A", "C", "D"])

    def test_missing_titles_are_not_collapsed_together(self):
        catalog = {"A": {}, "B": {}, "C": {"title": "Known"}}
        self.assertEqual(suppress_duplicates(["A", "B", "C"], catalog), ["A", "B", "C"])

    def test_top_k_backfills_and_returns_exactly_available_k(self):
        pool = ["A", "B", "C", "D", "E"]
        result = deduplicated_top_k(pool, self.catalog, k=5, threshold=0.8)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], "A")
        self.assertEqual(set(result), set(pool))

    def test_requested_diverse_top_k_api(self):
        self.assertEqual(
            diverse_top_k(["A", "B", "C", "D"], self.catalog, k=3),
            ["A", "C", "D"],
        )

    def test_top10_and_full_scopes_preserve_rank_one(self):
        pool = ["A", "B", "C", "D", "E"]
        for scope in ("top10", "full"):
            with self.subTest(scope=scope):
                result = deduplicated_top_k(
                    pool, self.catalog, k=3, threshold=0.8, scope=scope
                )
                self.assertEqual(result[0], "A")
                self.assertEqual(result, ["A", "C", "D"])

    def test_duplicate_pair_count(self):
        self.assertEqual(count_duplicate_pairs(["A", "B", "C"], self.catalog, 0.8), 1)

    def test_variant_sensitive_mode_preserves_different_variants(self):
        catalog = {
            "A": {"title": "Classic Cotton Shirt Black Size M"},
            "B": {"title": "Classic Cotton Shirt Blue Size L"},
        }
        self.assertNotEqual(variant_tokens(catalog["A"]["title"]), variant_tokens(catalog["B"]["title"]))
        self.assertEqual(
            suppress_duplicates(["A", "B"], catalog, 0.5, variant_sensitive=True),
            ["A", "B"],
        )


if __name__ == "__main__":
    unittest.main()
